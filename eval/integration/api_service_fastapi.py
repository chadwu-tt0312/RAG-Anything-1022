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
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

from raganything import RAGAnything, RAGAnythingConfig

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=".env", override=False)
except Exception:
    pass


app = FastAPI(title="RAG-Anything Integration API", version="0.1")

# 啟用 CORS（可根據需求調整）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生產環境應限制特定來源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 效能監控：追蹤請求統計
_request_stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_duration": 0.0,
}
_stats_lock = asyncio.Lock()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """記錄請求並追蹤效能"""
    start_time = time.time()
    global _request_stats

    # 記錄請求
    logger.info(f"{request.method} {request.url.path}")

    try:
        response = await call_next(request)
        duration = time.time() - start_time

        async with _stats_lock:
            _request_stats["total_requests"] += 1
            if response.status_code < 400:
                _request_stats["successful_requests"] += 1
            else:
                _request_stats["failed_requests"] += 1
            _request_stats["total_duration"] += duration

        # 記錄響應時間
        logger.info(
            f"{request.method} {request.url.path} - {response.status_code} ({duration:.3f}s)"
        )

        return response
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"{request.method} {request.url.path} - Error: {str(e)} ({duration:.3f}s)")
        async with _stats_lock:
            _request_stats["total_requests"] += 1
            _request_stats["failed_requests"] += 1
            _request_stats["total_duration"] += duration
        raise


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全域異常處理器"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


class InsertTextRequest(BaseModel):
    doc_id: Optional[str] = Field(default=None, description="Optional custom document id")
    file_path: Optional[str] = Field(default=None, description="Optional citation file path/name")
    text: str = Field(min_length=1)


class ProcessFileRequest(BaseModel):
    file_path: str
    parse_method: str = "auto"


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, description="查詢問題")
    mode: str = Field(default="hybrid", description="查詢模式: naive, local, global, hybrid")
    vlm_enhanced: bool = Field(default=False, description="是否使用 VLM 增強")

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        valid_modes = ["naive", "local", "global", "hybrid"]
        if v not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {v}")
        return v


class QueryResponse(BaseModel):
    result: str


class StatusResponse(BaseModel):
    status: str


class StatsResponse(BaseModel):
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_duration: float
    success_rate: float


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
    """健康檢查端點（不需要認證）"""
    return StatusResponse(status="ok")


@app.get("/v1/stats", response_model=StatsResponse, dependencies=[Depends(_auth)])
async def get_stats() -> StatsResponse:
    """取得 API 統計資訊（需要認證）"""
    async with _stats_lock:
        total = _request_stats["total_requests"]
        if total == 0:
            avg_duration = 0.0
            success_rate = 0.0
        else:
            avg_duration = _request_stats["total_duration"] / total
            success_rate = (_request_stats["successful_requests"] / total) * 100

        return StatsResponse(
            total_requests=_request_stats["total_requests"],
            successful_requests=_request_stats["successful_requests"],
            failed_requests=_request_stats["failed_requests"],
            average_duration=avg_duration,
            success_rate=success_rate,
        )


@app.post("/v1/insert_text", response_model=StatusResponse, dependencies=[Depends(_auth)])
async def insert_text(req: InsertTextRequest) -> StatusResponse:
    """插入文字內容到知識庫"""
    try:
        rag = await _get_rag()
        async with _rag_lock:
            await rag._ensure_lightrag_initialized()
            await rag.lightrag.ainsert(
                input=req.text,
                ids=req.doc_id,
                file_paths=req.file_path,
            )
            await rag.lightrag.finalize_storages()
        logger.info(f"Successfully inserted text (doc_id: {req.doc_id})")
        return StatusResponse(status="success")
    except Exception as e:
        logger.error(f"Error inserting text: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to insert text: {str(e)}")


@app.post("/v1/process_file", response_model=StatusResponse, dependencies=[Depends(_auth)])
async def process_file(req: ProcessFileRequest) -> StatusResponse:
    """處理檔案並插入到知識庫"""
    try:
        rag = await _get_rag()
        file_path = Path(req.file_path).resolve()
        if not file_path.exists():
            raise HTTPException(status_code=400, detail=f"File not found: {file_path}")

        # 驗證 parse_method
        valid_methods = ["auto", "layout", "ocr"]
        if req.parse_method not in valid_methods:
            raise HTTPException(
                status_code=400,
                detail=f"parse_method must be one of {valid_methods}, got {req.parse_method}",
            )

        out_dir = Path(os.getenv("OUTPUT_DIR", "./output")).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing file: {file_path} (method: {req.parse_method})")
        async with _rag_lock:
            await rag.process_document_complete(
                file_path=str(file_path),
                output_dir=str(out_dir),
                parse_method=req.parse_method,
            )
        logger.info(f"Successfully processed file: {file_path}")
        return StatusResponse(status="success")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@app.post("/v1/query", response_model=QueryResponse, dependencies=[Depends(_auth)])
async def query(req: QueryRequest) -> QueryResponse:
    """查詢知識庫"""
    try:
        rag = await _get_rag()
        logger.info(f"Query: {req.query[:100]}... (mode: {req.mode}, vlm: {req.vlm_enhanced})")
        async with _rag_lock:
            result = await rag.aquery(req.query, mode=req.mode, vlm_enhanced=req.vlm_enhanced)

        # 處理 None 或空字串的情況 - 確保 result 始終是有效的字串
        default_message = (
            "抱歉，無法找到相關的答案。請確認知識庫中已包含相關內容，或嘗試使用不同的查詢方式。"
        )

        # 統一處理：將 result 轉換為有效的字串
        if result is None:
            logger.warning(
                "Query returned None - knowledge base may be empty or no relevant content found"
            )
            result = default_message
        elif not isinstance(result, str):
            logger.warning(f"Query returned unexpected type: {type(result)}, converting to string")
            try:
                result = str(result) if result is not None else default_message
            except Exception:
                result = default_message

        # 確保 result 是字串類型
        if not isinstance(result, str):
            result = default_message

        # 處理空字串或只有空白字元的情況
        if not result or not result.strip():
            logger.warning("Query returned empty or whitespace-only string")
            result = default_message

        # 最終驗證：確保 result 是有效的非空字串
        if not isinstance(result, str) or not result.strip():
            logger.error(
                f"Failed to ensure valid result string, result type: {type(result)}, value: {result}"
            )
            result = default_message

        logger.info(f"Query completed (result length: {len(result)})")
        return QueryResponse(result=result)
    except ValueError as e:
        # LightRAG 未初始化或知識庫為空
        error_msg = str(e)
        logger.error(f"Query validation error: {error_msg}")
        raise HTTPException(
            status_code=400,
            detail=f"Query failed: {error_msg}. Please ensure documents have been processed first.",
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")
