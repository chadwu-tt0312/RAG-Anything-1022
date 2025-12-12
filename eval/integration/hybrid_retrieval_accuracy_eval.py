"""Hybrid retrieval accuracy evaluation (naive/local/global/hybrid/mix/bypass).

This script:
- Processes one or more documents into a LightRAG workspace via RAGAnything
- Runs a fixed query set across multiple modes
- Scores each answer via embedding cosine similarity against ground-truth answers
- Saves JSON + CSV into eval/integration/_results/

Assumptions (可自行調整):
- Accuracy metric uses embedding cosine similarity between predicted answer and ground truth.
- VLM-enhanced query is disabled for fair text-mode comparison.

Usage (PowerShell):
  python eval/integration/hybrid_retrieval_accuracy_eval.py `
    --docs eval/docs/test-01.txt `
    --query-set eval/integration/hybrid_eval_queries.jsonl `
    --working-dir eval/integration/_rag_storage_hybrid_eval `
    --modes naive local global hybrid `
    --out-dir eval/integration/_results

Environment:
- LLM_BINDING_API_KEY, LLM_BINDING_HOST, LLM_MODEL
- EMBEDDING_MODEL, EMBEDDING_DIM
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

from raganything import RAGAnything, RAGAnythingConfig


def _try_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=".env", override=False)
    except Exception:
        # Optional dependency
        return


def _cosine_sim(a: List[float], b: List[float]) -> float:
    # Convert to list if numpy array
    if isinstance(a, np.ndarray):
        a = a.tolist()
    if isinstance(b, np.ndarray):
        b = b.tolist()

    if len(a) == 0 or len(b) == 0 or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / ((na**0.5) * (nb**0.5))


@dataclass
class QueryCase:
    id: str
    query: str
    ground_truth: str
    query_type: str = ""
    expected_best_mode: str = ""


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_query_cases(path: Path) -> List[QueryCase]:
    raw = _read_jsonl(path)
    cases: List[QueryCase] = []
    for r in raw:
        cases.append(
            QueryCase(
                id=str(r["id"]),
                query=str(r["query"]),
                ground_truth=str(r["ground_truth"]),
                query_type=str(r.get("query_type", "")),
                expected_best_mode=str(r.get("expected_best_mode", "")),
            )
        )
    return cases


def _ensure_parent_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _build_llm_and_embedding():
    api_key = os.getenv("LLM_BINDING_API_KEY")
    base_url = os.getenv("LLM_BINDING_HOST")
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    if not api_key:
        raise RuntimeError("Missing env var: LLM_BINDING_API_KEY")

    def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        return openai_complete_if_cache(
            llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    emb_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    emb_dim = int(os.getenv("EMBEDDING_DIM", "1536"))

    embedding_func = EmbeddingFunc(
        embedding_dim=emb_dim,
        max_token_size=8192,
        func=lambda texts: openai_embed(
            texts,
            model=emb_model,
            api_key=api_key,
            base_url=base_url,
        ),
    )

    return llm_model_func, embedding_func


async def _process_docs_if_needed(rag: RAGAnything, docs: List[Path], output_dir: Path) -> None:
    # If working_dir already contains data, skip to keep evaluation repeatable.
    wd = Path(rag.config.working_dir)
    if wd.exists() and any(wd.iterdir()):
        return

    for doc in docs:
        await rag.process_document_complete(
            file_path=str(doc),
            output_dir=str(output_dir),
            parse_method="auto",
        )


async def main(args: argparse.Namespace) -> int:
    _try_load_dotenv()

    docs = [Path(p).resolve() for p in args.docs]
    for p in docs:
        if not p.exists():
            raise FileNotFoundError(str(p))

    query_set_path = Path(args.query_set).resolve()
    cases = _load_query_cases(query_set_path)

    llm_model_func, embedding_func = _build_llm_and_embedding()

    cfg = RAGAnythingConfig(
        working_dir=str(Path(args.working_dir).resolve()),
        parser=os.getenv("PARSER", "mineru"),
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    rag = RAGAnything(
        config=cfg,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        # For this accuracy test, force-disable VLM to avoid multimodal confound.
        vision_model_func=None,
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    await _process_docs_if_needed(rag, docs=docs, output_dir=output_dir)

    # Ensure LightRAG instance is initialized even if docs were skipped
    await rag._ensure_lightrag_initialized()

    modes: List[str] = list(args.modes)

    run_dir = Path(args.out_dir).resolve() / "hybrid_retrieval_accuracy" / _now_tag()
    run_dir.mkdir(parents=True, exist_ok=True)

    # Pre-embed ground truths (one-by-one to keep memory predictable)
    gt_embeddings: Dict[str, List[float]] = {}
    for c in cases:
        emb = (await embedding_func([c.ground_truth]))[0]
        gt_embeddings[c.id] = emb.tolist() if isinstance(emb, np.ndarray) else emb

    rows_for_csv: List[Dict[str, Any]] = []
    results: Dict[str, Any] = {
        "meta": {
            "docs": [str(p) for p in docs],
            "query_set": str(query_set_path),
            "modes": modes,
            "working_dir": str(Path(cfg.working_dir).resolve()),
            "llm_model": os.getenv("LLM_MODEL", ""),
            "embedding_model": os.getenv("EMBEDDING_MODEL", ""),
            "embedding_dim": int(os.getenv("EMBEDDING_DIM", "0") or "0"),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "vlm_enhanced": False,
        },
        "cases": [],
    }

    for c in cases:
        case_result: Dict[str, Any] = {
            "id": c.id,
            "query": c.query,
            "ground_truth": c.ground_truth,
            "query_type": c.query_type,
            "expected_best_mode": c.expected_best_mode,
            "by_mode": {},
        }

        for mode in modes:
            t0 = time.perf_counter()
            answer = await rag.aquery(c.query, mode=mode, vlm_enhanced=False)
            elapsed_s = time.perf_counter() - t0

            ans_emb_raw = (await embedding_func([answer]))[0]
            ans_emb = ans_emb_raw.tolist() if isinstance(ans_emb_raw, np.ndarray) else ans_emb_raw
            score = _cosine_sim(ans_emb, gt_embeddings[c.id])

            case_result["by_mode"][mode] = {
                "answer": answer,
                "elapsed_s": elapsed_s,
                "score": score,
            }

            rows_for_csv.append(
                {
                    "case_id": c.id,
                    "query_type": c.query_type,
                    "mode": mode,
                    "score": score,
                    "elapsed_s": elapsed_s,
                    "expected_best_mode": c.expected_best_mode,
                }
            )

        results["cases"].append(case_result)

    # Aggregate scores
    agg: Dict[str, Dict[str, float]] = {}
    for mode in modes:
        scores = [cr["by_mode"][mode]["score"] for cr in results["cases"] if mode in cr["by_mode"]]
        times = [
            cr["by_mode"][mode]["elapsed_s"] for cr in results["cases"] if mode in cr["by_mode"]
        ]
        agg[mode] = {
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "avg_elapsed_s": sum(times) / len(times) if times else 0.0,
        }

    results["aggregate"] = agg

    json_path = run_dir / "results.json"
    _ensure_parent_dir(json_path)
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = run_dir / "results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "query_type",
                "mode",
                "score",
                "elapsed_s",
                "expected_best_mode",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_for_csv)

    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    print(json.dumps({"aggregate": agg}, ensure_ascii=False, indent=2))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hybrid retrieval accuracy evaluation")
    p.add_argument(
        "--docs",
        nargs="+",
        required=True,
        help="Document paths to process (e.g., eval/docs/test-01.txt)",
    )
    p.add_argument(
        "--query-set",
        required=True,
        help="JSONL query set path (id/query/ground_truth)",
    )
    p.add_argument(
        "--working-dir",
        default="eval/integration/rag_storage_hybrid_eval",
        help="LightRAG working_dir for this evaluation",
    )
    p.add_argument(
        "--output-dir",
        default="eval/integration/output",
        help="Parser output directory",
    )
    p.add_argument(
        "--modes",
        nargs="+",
        default=["naive", "local", "global", "hybrid"],
        help="Modes to evaluate",
    )
    p.add_argument(
        "--out-dir",
        default="eval/integration/results",
        help="Output root directory",
    )
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args)))
