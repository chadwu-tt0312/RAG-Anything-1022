"""FastAPI service example for integrating RAG-Anything with existing systems.

This service exposes a minimal API:
- GET  /health
- POST /v1/insert_text : insert raw text content (no parser dependency; container-friendly)
- POST /v1/process_file: process a local file path (requires parser installed & file accessible)
- POST /v1/query       : query the knowledge base

Auth:
- Uses X-API-Key header. Set RAG_API_KEY in env.

Run (dev):
  pip install -r eval/integration/requirements-integration.txt
  uvicorn eval.integration.api_service_fastapi:app --host 0.0.0.0 --port 8000

Env (OpenAI-compatible):
- LLM_BINDING_API_KEY
- LLM_BINDING_HOST (optional)
- LLM_MODEL (default gpt-4o-mini)
- EMBEDDING_MODEL (default text-embedding-3-small)
- EMBEDDING_DIM (default 1536)

Storage:
- WORKING_DIR (default ./rag_storage)
- OUTPUT_DIR (default ./output)
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

from raganything import RAGAnything, RAGAnythingConfig

try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=".env", override=False)
except Exception:
    pass


app = FastAPI(title="RAG-Anything Integration API", version="0.1")


class InsertTextRequest(BaseModel):
    doc_id: Optional[str] = Field(default=None, description="Optional custom document id")
    file_path: Optional[str] = Field(default=None, description="Optional citation file path/name")
    text: str = Field(min_length=1)


class ProcessFileRequest(BaseModel):
    file_path: str
    parse_method: str = "auto"


class QueryRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    vlm_enhanced: bool = False


class QueryResponse(BaseModel):
    result: str


class StatusResponse(BaseModel):
    status: str


_rag: Optional[RAGAnything] = None
_rag_lock = asyncio.Lock()


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


async def _get_rag() -> RAGAnything:
    global _rag
    if _rag is not None:
        return _rag

    llm_model_func, embedding_func = _build_llm_and_embedding()

    cfg = RAGAnythingConfig(
        working_dir=os.getenv("WORKING_DIR", "./rag_storage"),
        parser=os.getenv("PARSER", "mineru"),
        parse_method=os.getenv("PARSE_METHOD", "auto"),
        enable_image_processing=os.getenv("ENABLE_IMAGE_PROCESSING", "true").lower() == "true",
        enable_table_processing=os.getenv("ENABLE_TABLE_PROCESSING", "true").lower() == "true",
        enable_equation_processing=os.getenv("ENABLE_EQUATION_PROCESSING", "true").lower()
        == "true",
    )

    _rag = RAGAnything(
        config=cfg,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        vision_model_func=None,  # API sample keeps VLM optional; enable if needed.
    )
    return _rag


async def _auth(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
    expected = os.getenv("RAG_API_KEY")
    if not expected:
        raise HTTPException(
            status_code=500,
            detail="Server misconfigured: missing RAG_API_KEY",
        )
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health", response_model=StatusResponse)
async def health() -> StatusResponse:
    return StatusResponse(status="ok")


@app.post("/v1/insert_text", response_model=StatusResponse, dependencies=[Depends(_auth)])
async def insert_text(req: InsertTextRequest) -> StatusResponse:
    rag = await _get_rag()
    async with _rag_lock:
        await rag._ensure_lightrag_initialized()
        await rag.lightrag.ainsert(
            input=req.text,
            ids=req.doc_id,
            file_paths=req.file_path,
        )
        await rag.lightrag.finalize_storages()
    return StatusResponse(status="success")


@app.post("/v1/process_file", response_model=StatusResponse, dependencies=[Depends(_auth)])
async def process_file(req: ProcessFileRequest) -> StatusResponse:
    rag = await _get_rag()
    file_path = Path(req.file_path).resolve()
    if not file_path.exists():
        raise HTTPException(status_code=400, detail=f"File not found: {file_path}")

    out_dir = Path(os.getenv("OUTPUT_DIR", "./output")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    async with _rag_lock:
        await rag.process_document_complete(
            file_path=str(file_path),
            output_dir=str(out_dir),
            parse_method=req.parse_method,
        )
    return StatusResponse(status="success")


@app.post("/v1/query", response_model=QueryResponse, dependencies=[Depends(_auth)])
async def query(req: QueryRequest) -> QueryResponse:
    rag = await _get_rag()
    async with _rag_lock:
        result = await rag.aquery(req.query, mode=req.mode, vlm_enhanced=req.vlm_enhanced)
    return QueryResponse(result=result)
