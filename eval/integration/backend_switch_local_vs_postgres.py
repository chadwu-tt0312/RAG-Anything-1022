"""Backend switch test: local filesystem vs PostgreSQL.

Scope:
- Only tests these two backends.
- PostgreSQL mode selects LightRAG storages via env vars:
  LIGHTRAG_KV_STORAGE=PGKVStorage
  LIGHTRAG_VECTOR_STORAGE=PGVectorStorage
  LIGHTRAG_DOC_STATUS_STORAGE=PGDocStatusStorage
  plus POSTGRES_* connection env vars.

Usage:
  # 1) Start PostgreSQL (example using docker compose in deploy/docker)
  # 2) Export env vars: POSTGRES_HOST/PORT/USER/PASSWORD/DATABASE
  # 3) Run:
  python eval/integration/backend_switch_local_vs_postgres.py \
    --doc eval/docs/test-01.txt \
    --out-dir eval/integration/_results \
    --mode hybrid

Env (OpenAI-compatible for LLM+Embedding):
- LLM_BINDING_API_KEY, LLM_BINDING_HOST
- LLM_MODEL
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
from typing import Any, Dict, Optional

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


def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v


def _build_llm_and_embedding():
    api_key = _require_env("LLM_BINDING_API_KEY")
    base_url = os.getenv("LLM_BINDING_HOST")
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")

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


def _set_backend_local_env() -> None:
    # Unset PG storage selectors (fallback to local JSON storages)
    for k in [
        "LIGHTRAG_KV_STORAGE",
        "LIGHTRAG_VECTOR_STORAGE",
        "LIGHTRAG_DOC_STATUS_STORAGE",
        "LIGHTRAG_GRAPH_STORAGE",
    ]:
        if k in os.environ:
            os.environ.pop(k)


def _get_postgres_database() -> str:
    """取得 PostgreSQL 資料庫名稱，支援 POSTGRES_DATABASE 和 POSTGRES_DB"""
    db_name = os.getenv("POSTGRES_DATABASE") or os.getenv("POSTGRES_DB")
    if not db_name:
        raise RuntimeError("Missing env var: POSTGRES_DATABASE or POSTGRES_DB")
    return db_name


def _check_postgres_env() -> bool:
    """Check if all required PostgreSQL environment variables are set"""
    required_vars = [
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
    ]
    has_required = all(os.getenv(var) for var in required_vars)
    has_database = bool(os.getenv("POSTGRES_DATABASE") or os.getenv("POSTGRES_DB"))
    return has_required and has_database


def _set_backend_postgres_env() -> None:
    # Select PG storages
    os.environ["LIGHTRAG_KV_STORAGE"] = "PGKVStorage"
    os.environ["LIGHTRAG_VECTOR_STORAGE"] = "PGVectorStorage"
    os.environ["LIGHTRAG_DOC_STATUS_STORAGE"] = "PGDocStatusStorage"

    # Require connection info
    _require_env("POSTGRES_HOST")
    _require_env("POSTGRES_PORT")
    _require_env("POSTGRES_USER")
    _require_env("POSTGRES_PASSWORD")
    # 確保至少有一個資料庫名稱環境變數
    db_name = _get_postgres_database()
    # 同時設定兩個環境變數以確保相容性
    os.environ["POSTGRES_DATABASE"] = db_name
    os.environ["POSTGRES_DB"] = db_name


def _safe_cls_name(obj: Any) -> str:
    try:
        return obj.__name__  # class
    except Exception:
        try:
            return obj.__class__.__name__
        except Exception:
            return str(obj)


async def _clear_postgres_doc_status(rag: RAGAnything, doc_id: str) -> None:
    """Clear document status from PostgreSQL storage to allow reprocessing"""
    try:
        await rag._ensure_lightrag_initialized()
        lightrag = rag.lightrag
        if hasattr(lightrag, "doc_status_storage") and lightrag.doc_status_storage:
            # Try to delete the document status
            if hasattr(lightrag.doc_status_storage, "delete"):
                await lightrag.doc_status_storage.delete(doc_id)
            elif hasattr(lightrag.doc_status_storage, "remove"):
                await lightrag.doc_status_storage.remove(doc_id)
    except Exception as e:
        # If clearing fails, log but don't fail the entire process
        print(f"Warning: Failed to clear PostgreSQL doc_status for {doc_id}: {e}")


async def _run_once(
    *,
    backend_name: str,
    working_dir: Path,
    doc_path: Path,
    output_dir: Path,
    mode: str,
    llm_model_func,
    embedding_func,
    force_reprocess: bool = False,
) -> Dict[str, Any]:
    cfg = RAGAnythingConfig(
        working_dir=str(working_dir.resolve()),
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

    # For PostgreSQL backend, always process the document since storage is in DB, not local files
    # For local filesystem backend, check if working_dir is empty
    wd = Path(cfg.working_dir)
    is_postgres_backend = backend_name == "postgresql"
    
    if is_postgres_backend:
        # PostgreSQL backend: always process (storage is in DB, not local files)
        # First, ensure LightRAG is initialized to check which storage is actually being used
        await rag._ensure_lightrag_initialized()
        lightrag = rag.lightrag
        
        # Debug: Print actual storage classes being used
        print(f"INFO: PostgreSQL backend storage classes:")
        print(f"  KV Storage: {_safe_cls_name(getattr(lightrag, 'key_string_value_json_storage_cls', None))}")
        print(f"  Vector Storage: {_safe_cls_name(getattr(lightrag, 'vector_db_storage_cls', None))}")
        print(f"  Doc Status Storage: {_safe_cls_name(getattr(lightrag, 'doc_status_storage_cls', None))}")
        
        # Check if doc_status_storage is actually PostgreSQL
        doc_status_storage = getattr(lightrag, 'doc_status_storage', None)
        if doc_status_storage:
            storage_cls_name = _safe_cls_name(doc_status_storage)
            print(f"  Doc Status Storage instance: {storage_cls_name}")
            
            # Try to check if there's existing doc_status for this document
            from lightrag.utils import compute_mdhash_id
            with open(doc_path, "rb") as f:
                content = f.read()
            doc_id = compute_mdhash_id(content, prefix="doc-")
            print(f"  Document ID: {doc_id}")
            
            # Try to read existing doc_status to verify which storage is being checked
            try:
                if hasattr(doc_status_storage, "get"):
                    existing_status = await doc_status_storage.get(doc_id)
                    if existing_status:
                        print(f"  WARNING: Found existing doc_status in {storage_cls_name}: {existing_status}")
                    else:
                        print(f"  INFO: No existing doc_status found in {storage_cls_name}")
                elif hasattr(doc_status_storage, "load"):
                    existing_status = await doc_status_storage.load(doc_id)
                    if existing_status:
                        print(f"  WARNING: Found existing doc_status in {storage_cls_name}: {existing_status}")
                    else:
                        print(f"  INFO: No existing doc_status found in {storage_cls_name}")
            except Exception as e:
                print(f"  INFO: Could not check existing doc_status: {e}")
            
            # Clear doc_status if force_reprocess is True or if we want to ensure fresh processing
            if force_reprocess:
                await _clear_postgres_doc_status(rag, doc_id)
        else:
            print(f"  WARNING: doc_status_storage is None!")
        
        # The doc_status check will be done by LightRAG internally
        await rag.process_document_complete(
            file_path=str(doc_path),
            output_dir=str(output_dir),
            parse_method="auto",
        )
    elif not (wd.exists() and any(wd.iterdir())):
        # Local filesystem backend: process if working_dir is empty
        await rag.process_document_complete(
            file_path=str(doc_path),
            output_dir=str(output_dir),
            parse_method="auto",
        )
    else:
        # Ensure LightRAG is initialized even if working_dir is not empty
        await rag._ensure_lightrag_initialized()

    q = "Transformer 相對於 RNN 的主要優點是什麼？"
    t0 = time.perf_counter()
    ans = await rag.aquery(q, mode=mode, vlm_enhanced=False)
    elapsed = time.perf_counter() - t0

    # Best-effort introspection of selected storage classes
    lightrag = rag.lightrag
    storage_info = {
        "key_string_value_json_storage_cls": _safe_cls_name(
            getattr(lightrag, "key_string_value_json_storage_cls", None)
        ),
        "vector_db_storage_cls": _safe_cls_name(getattr(lightrag, "vector_db_storage_cls", None)),
        "doc_status_storage_cls": _safe_cls_name(getattr(lightrag, "doc_status_storage_cls", None)),
    }

    return {
        "backend": backend_name,
        "working_dir": str(wd.resolve()),
        "query": q,
        "answer": ans,
        "elapsed_s": elapsed,
        "storage_info": storage_info,
    }


async def main(args: argparse.Namespace) -> int:
    _try_load_dotenv()

    doc_path = Path(args.doc).resolve()
    if not doc_path.exists():
        raise FileNotFoundError(str(doc_path))

    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    llm_model_func, embedding_func = _build_llm_and_embedding()

    # Local
    _set_backend_local_env()
    local = await _run_once(
        backend_name="local_filesystem",
        working_dir=Path(args.local_working_dir),
        doc_path=doc_path,
        output_dir=output_dir,
        mode=args.mode,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
    )

    # PostgreSQL (optional - skip if env vars not set)
    pg: Optional[Dict[str, Any]] = None
    if _check_postgres_env():
        try:
            _set_backend_postgres_env()
            pg = await _run_once(
                backend_name="postgresql",
                working_dir=Path(args.postgres_working_dir),
                doc_path=doc_path,
                output_dir=output_dir,
                mode=args.mode,
                llm_model_func=llm_model_func,
                embedding_func=embedding_func,
                force_reprocess=args.force_reprocess,
            )
        except Exception as e:
            print(f"Warning: PostgreSQL test failed: {e}")
            pg = {"error": str(e), "skipped": True}
    else:
        print("Warning: PostgreSQL environment variables not set, skipping PostgreSQL test")
        pg = {"skipped": True, "reason": "Missing PostgreSQL environment variables"}

    results: Dict[str, Any] = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "mode": args.mode,
            "doc": str(doc_path),
        },
        "local": local,
        "postgres": pg,
    }

    run_dir = out_root / "backend_switch_local_vs_postgres" / _now_tag()
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {out_path}")
    elapsed_info = {"local_elapsed_s": local["elapsed_s"]}
    if pg and "elapsed_s" in pg:
        elapsed_info["pg_elapsed_s"] = pg["elapsed_s"]
    print(json.dumps(elapsed_info, ensure_ascii=False, indent=2))

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Backend switch test: local vs PostgreSQL")
    p.add_argument("--doc", default="eval/docs/test-01.txt")
    p.add_argument("--mode", default="hybrid")
    p.add_argument("--output-dir", default="eval/integration/output")
    p.add_argument("--out-dir", default="eval/integration/results")
    p.add_argument(
        "--local-working-dir",
        default="eval/integration/rag_storage_backend_local",
    )
    p.add_argument(
        "--postgres-working-dir",
        default="eval/integration/rag_storage_backend_postgres",
    )
    p.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Force reprocessing of documents even if they exist in PostgreSQL doc_status",
    )
    return p


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(build_arg_parser().parse_args())))
