"""Azure OpenAI integration smoketest.

What this verifies:
- Chat completions works via Azure OpenAI (deployment + api_version)
- Embeddings works via Azure OpenAI (embedding deployment + api_version)
- RAGAnything can run a small end-to-end flow using Azure for LLM + embeddings

Usage:
  python eval/integration/azure_openai_integration_smoketest.py \
    --doc eval/docs/test-01.txt \
    --working-dir eval/integration/_rag_storage_azure_eval \
    --mode hybrid

Required env (LLM):
- LLM_BINDING_HOST=https://{resource}.openai.azure.com
- LLM_BINDING_API_KEY=...
- AZURE_OPENAI_API_VERSION=2024-08-01-preview (example)
- AZURE_OPENAI_DEPLOYMENT=<your chat deployment name>

Required env (Embedding):
- AZURE_EMBEDDING_DEPLOYMENT=<your embedding deployment name>
- AZURE_EMBEDDING_API_VERSION=2023-05-15 (example)
- EMBEDDING_DIM=1536 (text-embedding-3-small) or 3072 (text-embedding-3-large)

Notes:
- In Azure OpenAI, you pass **deployment name** (not model name) to the client.
- This script uses lightrag.llm.openai wrappers used elsewhere in this repo.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

from raganything import RAGAnything, RAGAnythingConfig


def _try_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=".env", override=False)
    except Exception:
        return


def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v


def _build_azure_llm_and_embedding():
    api_key = _require_env("LLM_BINDING_API_KEY")
    base_url = _require_env("LLM_BINDING_HOST")

    chat_deployment = _require_env("AZURE_OPENAI_DEPLOYMENT")
    chat_api_version = _require_env("AZURE_OPENAI_API_VERSION")

    emb_deployment = _require_env("AZURE_EMBEDDING_DEPLOYMENT")
    emb_api_version = _require_env("AZURE_EMBEDDING_API_VERSION")
    emb_dim = int(_require_env("AZURE_EMBEDDING_DIM"))

    def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        return openai_complete_if_cache(
            chat_deployment,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=api_key,
            base_url=base_url,
            api_version=chat_api_version,
            **kwargs,
        )

    embedding_func = EmbeddingFunc(
        embedding_dim=emb_dim,
        max_token_size=8192,
        func=lambda texts: openai_embed(
            texts,
            model=emb_deployment,
            api_key=api_key,
            base_url=base_url,
            api_version=emb_api_version,
        ),
    )

    return llm_model_func, embedding_func


async def main(args: argparse.Namespace) -> int:
    _try_load_dotenv()

    doc_path = Path(args.doc).resolve()
    if not doc_path.exists():
        raise FileNotFoundError(str(doc_path))

    llm_model_func, embedding_func = _build_azure_llm_and_embedding()

    # 1) LLM quick check
    llm_out = await llm_model_func("請用一句話說明 Transformer 的核心創新是什麼？")
    print("[LLM OK]", str(llm_out)[:120])

    # 2) Embedding quick check
    emb = await embedding_func(["hello", "world"])
    print("[Embedding OK] dim=", len(emb[0]))

    # 3) End-to-end RAG
    cfg = RAGAnythingConfig(
        working_dir=str(Path(args.working_dir).resolve()),
        parser=os.getenv("PARSER", "mineru"),
        parse_method="auto",
        enable_image_processing=False,
        enable_table_processing=False,
        enable_equation_processing=False,
    )

    rag = RAGAnything(
        config=cfg,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        vision_model_func=None,
    )

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip re-processing if working_dir already has data
    wd = Path(cfg.working_dir)
    if not (wd.exists() and any(wd.iterdir())):
        await rag.process_document_complete(
            file_path=str(doc_path),
            output_dir=str(out_dir),
            parse_method="auto",
        )

    answer = await rag.aquery(
        "Transformer 相對於 RNN 的主要優點是什麼？", mode=args.mode, vlm_enhanced=False
    )
    print("[RAG OK]", answer[:300])

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Azure OpenAI integration smoketest")
    p.add_argument("--doc", default="eval/docs/test-01.txt")
    p.add_argument("--working-dir", default="eval/integration/rag_storage_azure_eval")
    p.add_argument("--output-dir", default="eval/integration/output")
    p.add_argument("--mode", default="hybrid")
    return p


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(build_arg_parser().parse_args())))
