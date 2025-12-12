import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

from raganything import RAGAnything, RAGAnythingConfig

# NOTE:
# `eval/` 不是 Python package（沒有 __init__.py），所以這裡用同資料夾匯入，
# 讓你能直接執行：python eval/kg_eval/run_kg_correctness_eval.py ...
from label_studio_to_ground_truth import parse_label_studio_export


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _relation_pair(head: str, tail: str) -> Tuple[str, str]:
    return (_norm(head), _norm(tail))


def _calc_prf(tp: int, pred_total: int, gt_total: int) -> Dict[str, float]:
    precision = tp / pred_total if pred_total else 0.0
    recall = tp / gt_total if gt_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


async def build_rag(file_path: str, working_dir: str):
    api_key = os.getenv("LLM_BINDING_API_KEY")
    base_url = os.getenv("LLM_BINDING_HOST")

    config = RAGAnythingConfig(
        working_dir=working_dir,
        parser=os.getenv("PARSER", "mineru"),
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            os.getenv("LLM_MODEL", "gpt-4o-mini"),
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    embedding_func = EmbeddingFunc(
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "1536")),
        max_token_size=int(os.getenv("EMBEDDING_MAX_TOKEN", "8192")),
        func=lambda texts: openai_embed(
            texts,
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            api_key=api_key,
            base_url=base_url,
        ),
    )

    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
    )

    # `.txt/.md` 走 LightRAG 的純文字 insert，避免 txt->pdf 轉換依賴（reportlab）。
    # 其他格式（pdf/docx/pptx...）維持原本的 process_document_complete。
    p = Path(file_path)
    if p.suffix.lower() in {".txt", ".md"}:
        init = await rag._ensure_lightrag_initialized()
        if not init.get("success"):
            raise RuntimeError(init.get("error", "Failed to initialize LightRAG"))

        text = p.read_text(encoding="utf-8")
        await rag.lightrag.ainsert(input=text, ids=p.name, file_paths=str(p))
    else:
        await rag.process_document_complete(
            file_path=file_path,
            output_dir=str(Path(working_dir) / "output"),
            parse_method="auto",
        )
    return rag


async def main():
    ap = argparse.ArgumentParser(
        description="KG correctness evaluation: Label Studio ground truth vs RAG-Anything extracted KG (loose matching)."
    )
    ap.add_argument(
        "--doc", required=True, help="Document path under ./eval/docs (e.g. eval/docs/test-01.txt)"
    )
    ap.add_argument(
        "--labelstudio-export",
        required=True,
        help="Label Studio export JSON path (Export -> JSON)",
    )
    ap.add_argument(
        "--working-dir",
        default="eval/kg_eval/_rag_storage/test-01",
        help="LightRAG working_dir (will be created if not exist)",
    )
    ap.add_argument(
        "--out",
        default="eval/kg_eval/_results/test-01",
        help="Output directory for evaluation artifacts",
    )
    args = ap.parse_args()

    doc_path = Path(args.doc)
    export_path = Path(args.labelstudio_export)
    working_dir = Path(args.working_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Ground truth from Label Studio export
    export_data = json.loads(export_path.read_text(encoding="utf-8"))
    gt_entities_dict, gt_relations_dict = parse_label_studio_export(
        export_data, relation_case="lower"
    )

    gt_entities: Set[str] = {_norm(k) for k in gt_entities_dict.keys()}
    gt_relation_pairs: Set[Tuple[str, str]] = {
        _relation_pair(v.get("head", ""), v.get("tail", ""))
        for v in gt_relations_dict.values()
        if v.get("head") and v.get("tail")
    }

    # 2) Run RAG-Anything to build KG
    rag = await build_rag(str(doc_path), str(working_dir))
    lightrag = rag.lightrag

    # 從 graph 對象獲取所有實體（節點）和關係（邊）
    graph = lightrag.chunk_entity_relation_graph
    # 獲取所有節點（實體）- graph.nodes() 返回節點 ID 列表
    predicted_entities: Dict[str, Any] = {}
    try:
        for node_id in graph.nodes():
            predicted_entities[node_id] = graph.nodes[node_id] if node_id in graph.nodes else {}
    except Exception as e:
        print(f"Warning: Failed to get entities from graph: {e}", file=sys.stderr)

    # 獲取所有邊（關係）- graph.edges() 返回 (head, tail) 元組列表
    predicted_relations: Dict[str, Any] = {}
    try:
        for edge in graph.edges():
            head, tail = edge[0], edge[1]
            edge_data = graph.edges[edge] if edge in graph.edges else {}
            # 使用 (head, tail) 作為 key，但為了與 ground truth 格式一致，我們只需要 head 和 tail
            relation_key = f"{head}::{edge_data.get('keywords', 'unknown')}::{tail}"
            predicted_relations[relation_key] = {"head": head, "tail": tail, **edge_data}
    except Exception as e:
        print(f"Warning: Failed to get relations from graph: {e}", file=sys.stderr)

    pred_entities: Set[str] = {_norm(k) for k in (predicted_entities or {}).keys()}

    pred_relation_pairs: Set[Tuple[str, str]] = set()
    for rel in (predicted_relations or {}).values():
        head = rel.get("head", "")
        tail = rel.get("tail", "")
        if head and tail:
            pred_relation_pairs.add(_relation_pair(head, tail))

    # 3) Loose matching metrics
    entity_matches = sorted(list(gt_entities & pred_entities))
    relation_matches = sorted(list(gt_relation_pairs & pred_relation_pairs))

    entity_metrics = _calc_prf(len(entity_matches), len(pred_entities), len(gt_entities))
    relation_metrics = _calc_prf(
        len(relation_matches), len(pred_relation_pairs), len(gt_relation_pairs)
    )

    report = {
        "doc": str(doc_path),
        "labelstudio_export": str(export_path),
        "working_dir": str(working_dir),
        "counts": {
            "gt_entities": len(gt_entities),
            "pred_entities": len(pred_entities),
            "gt_relations": len(gt_relation_pairs),
            "pred_relations": len(pred_relation_pairs),
        },
        "entity_metrics_loose": entity_metrics,
        "relation_metrics_loose": relation_metrics,
        "matches": {
            "entities": entity_matches[:200],
            "relations": [{"head": h, "tail": t} for (h, t) in relation_matches[:200]],
        },
        "diff": {
            "gt_only_entities": sorted(list(gt_entities - pred_entities))[:200],
            "pred_only_entities": sorted(list(pred_entities - gt_entities))[:200],
            "gt_only_relations": [
                {"head": h, "tail": t}
                for (h, t) in sorted(list(gt_relation_pairs - pred_relation_pairs))[:200]
            ],
            "pred_only_relations": [
                {"head": h, "tail": t}
                for (h, t) in sorted(list(pred_relation_pairs - gt_relation_pairs))[:200]
            ],
        },
        "notes": {
            "matching_policy": "loose (entities by name; relations by (head, tail) only; relation_type not required)",
        },
    }

    out_file = out_dir / "comparison_results.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"OK: wrote report -> {out_file}")
    print(
        f"Entity metrics (loose): P={entity_metrics['precision']:.2%}, R={entity_metrics['recall']:.2%}, F1={entity_metrics['f1']:.2%}"
    )
    print(
        f"Relation metrics (loose): P={relation_metrics['precision']:.2%}, R={relation_metrics['recall']:.2%}, F1={relation_metrics['f1']:.2%}"
    )


if __name__ == "__main__":
    asyncio.run(main())
