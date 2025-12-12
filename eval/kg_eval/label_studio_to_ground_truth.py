import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


RAG_ENTITY_TYPES = {
    "TECHNICAL_TERM",
    "SYSTEM_COMPONENT",
    "PERSON",
    "ORGANIZATION",
    "IMAGE",
    "TABLE",
    "EQUATION",
    "DATE",  # extension (方案 B)
}


_CJK_FALLBACK_LABEL_MAP = {
    # 讓 repo 內既有的 Label Studio 匯出（中文 label）也能直接轉換
    "模型名稱": "TECHNICAL_TERM",
    "核心機制": "TECHNICAL_TERM",
    "組件/架構": "SYSTEM_COMPONENT",
    "應用領域": "TECHNICAL_TERM",
    # "年份/人物" 需用啟發式判斷（PERSON vs DATE）
}


def _norm_text(s: str) -> str:
    # 寬鬆比對用的基礎正規化：去前後空白、合併多空白
    return re.sub(r"\s+", " ", (s or "").strip())


def _guess_entity_type(raw_label: str, entity_text: str) -> str:
    """
    把 Label Studio 的 entity label 轉成 ground truth 使用的 entity_type。
    - 優先支援 RAG-Anything 風格 label（TECHNICAL_TERM...）
    - 兼容中文 label（模型名稱...）做 mapping
    - 針對「年份/人物」用簡單啟發式：包含 4 位數年份 -> DATE，否則 PERSON
    """
    label = (raw_label or "").strip()
    if label in RAG_ENTITY_TYPES:
        return label

    if label == "年份/人物":
        if re.search(r"\b(19|20)\d{2}\b", entity_text):
            return "DATE"
        return "PERSON"

    if label in _CJK_FALLBACK_LABEL_MAP:
        return _CJK_FALLBACK_LABEL_MAP[label]

    # 最後 fallback：當成 TECHNICAL_TERM
    return "TECHNICAL_TERM"


def parse_label_studio_export(
    export_data: Any,
    *,
    relation_case: str = "lower",
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    解析 Label Studio 匯出的 JSON（Export -> JSON）。

    Returns:
        entities_by_name: { entity_name: {entity_type, source, occurrences...} }
        relations_by_key: { "head::relation_type::tail": {head, tail, relation_type, source} }
    """
    # Label Studio export 可能是 list of tasks
    tasks: List[Dict[str, Any]]
    if isinstance(export_data, list):
        tasks = export_data
    else:
        raise ValueError("Label Studio export JSON 必須是 task list (JSON array)")

    entities: Dict[str, Dict[str, Any]] = {}
    relations: Dict[str, Dict[str, Any]] = {}

    for task in tasks:
        annotations = task.get("annotations") or []
        if not annotations:
            continue

        # 只取第一份人工標註（多人的話可改成選 ground_truth==true 或最新）
        ann = annotations[0]
        results = ann.get("result") or []

        # 先收集所有 region(label span)
        region_by_id: Dict[str, Dict[str, Any]] = {}
        pending_relations: List[Dict[str, Any]] = []

        for r in results:
            r_type = r.get("type")
            if r_type == "labels":
                rid = r.get("id")
                val = r.get("value") or {}
                text = _norm_text(val.get("text", ""))
                labels = val.get("labels") or []
                raw_label = labels[0] if labels else ""
                region_by_id[rid] = {
                    "text": text,
                    "raw_label": raw_label,
                }
            elif r_type == "relation":
                pending_relations.append(r)

        # entities
        for rid, info in region_by_id.items():
            name = info["text"]
            if not name:
                continue
            entity_type = _guess_entity_type(info.get("raw_label", ""), name)

            if name not in entities:
                entities[name] = {
                    "entity_type": entity_type,
                    "source": "ground_truth",
                    "occurrences": 1,
                }
            else:
                # 若同名被標成不同類型，保留第一次，但累計 occurrences 方便人工檢查
                entities[name]["occurrences"] = int(entities[name].get("occurrences", 1)) + 1

        # relations
        for rel in pending_relations:
            from_id = rel.get("from_id")
            to_id = rel.get("to_id")
            labels = rel.get("labels") or []
            rel_type = labels[0] if labels else ""
            if relation_case == "lower":
                rel_type = (rel_type or "").lower()
            elif relation_case == "upper":
                rel_type = (rel_type or "").upper()

            head = _norm_text((region_by_id.get(from_id) or {}).get("text", ""))
            tail = _norm_text((region_by_id.get(to_id) or {}).get("text", ""))
            if not head or not tail:
                continue

            key = f"{head}::{rel_type}::{tail}"
            relations[key] = {
                "head": head,
                "tail": tail,
                "relation_type": rel_type,
                "source": "ground_truth",
            }

    return entities, relations


def main():
    ap = argparse.ArgumentParser(
        description="Convert Label Studio export JSON to a simple ground truth JSON for KG evaluation."
    )
    ap.add_argument(
        "--input",
        required=True,
        help="Label Studio export JSON path (Export -> JSON)",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output ground truth JSON path",
    )
    ap.add_argument(
        "--relation-case",
        choices=["lower", "upper", "keep"],
        default="lower",
        help="Normalize relation_type casing",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    export_data = json.loads(in_path.read_text(encoding="utf-8"))
    entities, relations = parse_label_studio_export(
        export_data,
        relation_case=args.relation_case,
    )

    ground_truth = {
        "entities": entities,
        "relations": relations,
    }
    out_path.write_text(json.dumps(ground_truth, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"OK: wrote ground truth -> {out_path}")
    print(f"  entities: {len(entities)}")
    print(f"  relations: {len(relations)}")


if __name__ == "__main__":
    main()


