"""VLM enhanced query evaluation (on/off comparison).

Goal:
- Verify that when retrieved context contains `Image Path: ...`, the VLM-enhanced path:
  1) extracts image paths
  2) encodes images as base64
  3) calls vision_model_func with multimodal messages

This script runs the same queries twice:
- vlm_enhanced=False (text-only)
- vlm_enhanced=True  (multimodal, requires vision_model_func)

Usage:
  python eval/integration/vlm_enhanced_query_eval.py \
    --docs eval/docs/全球資訊服務暨軟體市場規模.png eval/docs/cat-01.jpg \
    --working-dir eval/integration/_rag_storage_vlm_eval \
    --out-dir eval/integration/_results

Env:
- LLM_BINDING_API_KEY, LLM_BINDING_HOST
- LLM_MODEL (text model, default gpt-4o-mini)
- VLM_MODEL (vision model, default gpt-4o)
- EMBEDDING_MODEL, EMBEDDING_DIM
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from lightrag import QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

from raganything import RAGAnything, RAGAnythingConfig


def _try_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=".env", override=False)
    except Exception:
        return


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _build_llm_vision_embedding():
    api_key = os.getenv("LLM_BINDING_API_KEY")
    base_url = os.getenv("LLM_BINDING_HOST")

    if not api_key:
        raise RuntimeError("Missing env var: LLM_BINDING_API_KEY")

    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    vlm_model = os.getenv("VLM_MODEL", "gpt-4o")

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

    def vision_model_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        image_data=None,
        messages=None,
        **kwargs,
    ):
        # VLM enhanced query will pass messages (multimodal format) when images are found.
        if messages:
            return openai_complete_if_cache(
                vlm_model,
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        if image_data:
            # Single image format (used by image processor)
            return openai_complete_if_cache(
                vlm_model,
                "",
                system_prompt=None,
                history_messages=[],
                messages=[
                    {"role": "system", "content": system_prompt} if system_prompt else None,
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                            },
                        ],
                    },
                ],
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        return llm_model_func(prompt, system_prompt, history_messages or [], **kwargs)

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

    return llm_model_func, vision_model_func, embedding_func


async def main(args: argparse.Namespace) -> int:
    _try_load_dotenv()

    docs = [Path(p).resolve() for p in args.docs]
    for p in docs:
        if not p.exists():
            raise FileNotFoundError(str(p))

    llm_model_func, vision_model_func, embedding_func = _build_llm_vision_embedding()

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
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip processing if storage exists
    wd = Path(cfg.working_dir)
    if not (wd.exists() and any(wd.iterdir())):
        for doc in docs:
            await rag.process_document_complete(
                file_path=str(doc),
                output_dir=str(output_dir),
                parse_method="auto",
            )

    queries = [
        "請描述文件/圖片中的主要內容重點。",
        "如果是圖表，請說明它呈現的主題與可能的趨勢。",
        "若畫面有標題或重要文字，請摘要出來。",
    ]

    run_dir = Path(args.out_dir).resolve() / "vlm_enhanced_query" / _now_tag()
    run_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "meta": {
            "docs": [str(p) for p in docs],
            "working_dir": str(Path(cfg.working_dir).resolve()),
            "mode": args.mode,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "llm_model": os.getenv("LLM_MODEL", ""),
            "vlm_model": os.getenv("VLM_MODEL", ""),
        },
        "queries": [],
    }

    for q in queries:
        # Capture retrieval prompt (for debugging image-path extraction)
        prompt = await rag.lightrag.aquery(
            q, param=QueryParam(mode=args.mode, only_need_prompt=True)
        )
        image_path_count = prompt.count("Image Path:") if isinstance(prompt, str) else 0

        item: Dict[str, Any] = {
            "query": q,
            "retrieval_prompt": prompt,
            "retrieval_prompt_image_path_count": image_path_count,
            "runs": {},
        }

        for flag in [False, True]:
            t0 = time.perf_counter()
            ans = await rag.aquery(q, mode=args.mode, vlm_enhanced=flag)
            elapsed = time.perf_counter() - t0
            item["runs"][str(flag)] = {"vlm_enhanced": flag, "answer": ans, "elapsed_s": elapsed}

        results["queries"].append(item)

    out_path = run_dir / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {out_path}")
    # Quick sanity: show prompt image-path counts
    summary = [
        {
            "query": q["query"],
            "image_path_count": q["retrieval_prompt_image_path_count"],
            "elapsed_vlm_off_s": q["runs"]["False"]["elapsed_s"],
            "elapsed_vlm_on_s": q["runs"]["True"]["elapsed_s"],
        }
        for q in results["queries"]
    ]
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VLM enhanced query evaluation")
    p.add_argument("--docs", nargs="+", required=True)
    p.add_argument("--working-dir", default="eval/integration/rag_storage_vlm_eval")
    p.add_argument("--output-dir", default="eval/integration/output")
    p.add_argument("--out-dir", default="eval/integration/results")
    p.add_argument("--mode", default="hybrid")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    raise SystemExit(asyncio.run(main(args)))
