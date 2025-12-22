"""儲存後端綜合測試

目標：
- 驗證系統能正確切換不同的儲存後端
- 確認資料遷移正確
- 驗證各後端的效能差異
- 確認功能在各後端下一致

測試範圍：
1. 後端切換測試：驗證環境變數切換能正確選擇儲存後端
2. 資料遷移測試：從來源後端讀取資料，寫入目標後端，驗證資料完整性
3. 效能測試：比較不同後端的查詢、插入效能
4. 功能一致性測試：確保所有後端都能正確執行相同的操作

支援的後端：
- Local filesystem (預設，JSON files)
- PostgreSQL (PGKVStorage, PGVectorStorage, PGDocStatusStorage)

Usage:
  # 1) 確保 PostgreSQL 正在運行（如使用 docker compose）
  # 2) 設定環境變數：POSTGRES_HOST/PORT/USER/PASSWORD/DATABASE
  # 3) 執行測試：
  python eval/integration/storage_backend_comprehensive_test.py \
    --doc eval/docs/test-01.txt \
    --out-dir eval/integration/results \
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
import shutil
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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


async def _validate_api_connection(api_key: str, base_url: Optional[str], llm_model: str, emb_model: str) -> bool:
    """驗證 API 連接是否可用"""
    print("\n[驗證] 檢查 API 連接...")
    
    try:
        # 測試 LLM API
        print(f"  - 測試 LLM API (model: {llm_model})...")
        test_result = await openai_complete_if_cache(
            llm_model,
            "test",
            system_prompt=None,
            history_messages=[],
            api_key=api_key,
            base_url=base_url,
        )
        if not test_result:
            print(f"    ✗ LLM API 測試失敗：無回應")
            return False
        print(f"    ✓ LLM API 連接成功")
        
        # 測試 Embedding API
        print(f"  - 測試 Embedding API (model: {emb_model})...")
        emb_result = await openai_embed(
            ["test"],
            model=emb_model,
            api_key=api_key,
            base_url=base_url,
        )
        if not emb_result or len(emb_result) == 0:
            print(f"    ✗ Embedding API 測試失敗：無回應")
            return False
        print(f"    ✓ Embedding API 連接成功 (維度: {len(emb_result[0])})")
        
        return True
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        
        # 提供更詳細的錯誤診斷
        print(f"    ✗ API 連接失敗: {error_type}: {error_msg}")
        
        if "404" in error_msg or "NotFound" in error_type:
            print("\n[診斷] 404 錯誤可能原因：")
            if base_url:
                print(f"  - 檢查 LLM_BINDING_HOST: {base_url}")
                print(f"  - 確認 endpoint 格式正確（例如：https://api.openai.com/v1）")
            print(f"  - 檢查模型名稱是否存在：")
            print(f"    * LLM_MODEL: {llm_model}")
            print(f"    * EMBEDDING_MODEL: {emb_model}")
            print(f"  - 確認 API key 有權限存取這些模型")
        elif "401" in error_msg or "Unauthorized" in error_type:
            print("\n[診斷] 401 錯誤可能原因：")
            print(f"  - 檢查 LLM_BINDING_API_KEY 是否正確")
            print(f"  - 確認 API key 未過期")
        elif "429" in error_msg:
            print("\n[診斷] 429 錯誤：請求過多，請稍後再試")
        
        return False


def _build_llm_and_embedding():
    api_key = _require_env("LLM_BINDING_API_KEY")
    base_url = os.getenv("LLM_BINDING_HOST")
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        try:
            return openai_complete_if_cache(
                llm_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            # 提供更清晰的錯誤訊息
            if "404" in error_msg or "NotFound" in error_type:
                raise RuntimeError(
                    f"LLM API 404 錯誤：模型 '{llm_model}' 不存在或 endpoint 不正確。"
                    f"請檢查 LLM_BINDING_HOST={base_url} 和 LLM_MODEL={llm_model}"
                ) from e
            raise

    emb_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    emb_dim = int(os.getenv("EMBEDDING_DIM", "1536"))

    def embedding_wrapper(texts):
        try:
            return openai_embed(
                texts,
                model=emb_model,
                api_key=api_key,
                base_url=base_url,
            )
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            # 提供更清晰的錯誤訊息
            if "404" in error_msg or "NotFound" in error_type:
                raise RuntimeError(
                    f"Embedding API 404 錯誤：模型 '{emb_model}' 不存在或 endpoint 不正確。"
                    f"請檢查 LLM_BINDING_HOST={base_url} 和 EMBEDDING_MODEL={emb_model}"
                ) from e
            raise

    embedding_func = EmbeddingFunc(
        embedding_dim=emb_dim,
        max_token_size=8192,
        func=embedding_wrapper,
    )

    return llm_model_func, embedding_func


def _set_backend_local_env() -> None:
    """設定為本地檔案系統後端"""
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


def _set_backend_postgres_env() -> None:
    """設定為 PostgreSQL 後端"""
    os.environ["LIGHTRAG_KV_STORAGE"] = "PGKVStorage"
    os.environ["LIGHTRAG_VECTOR_STORAGE"] = "PGVectorStorage"
    os.environ["LIGHTRAG_DOC_STATUS_STORAGE"] = "PGDocStatusStorage"

    # 驗證連線資訊
    _require_env("POSTGRES_HOST")
    _require_env("POSTGRES_PORT")
    _require_env("POSTGRES_USER")
    _require_env("POSTGRES_PASSWORD")
    # 確保至少有一個資料庫名稱環境變數
    _get_postgres_database()
    # 同時設定兩個環境變數以確保相容性
    db_name = _get_postgres_database()
    os.environ["POSTGRES_DATABASE"] = db_name
    os.environ["POSTGRES_DB"] = db_name


def _safe_cls_name(obj: Any) -> str:
    """安全取得類別名稱"""
    try:
        return obj.__name__  # class
    except Exception:
        try:
            return obj.__class__.__name__
        except Exception:
            return str(obj)


def _get_storage_info(rag: RAGAnything) -> Dict[str, str]:
    """取得儲存後端資訊"""
    await rag._ensure_lightrag_initialized()
    lightrag = rag.lightrag
    
    # 取得實際使用的儲存實例類型
    kv_storage = getattr(lightrag, "key_string_value_json_storage", None)
    vector_storage = getattr(lightrag, "vector_db_storage", None)
    doc_status_storage = getattr(lightrag, "doc_status_storage", None)
    graph_storage = getattr(lightrag, "graph_storage", None)
    
    # 優先使用實例類型，否則使用類別
    return {
        "kv_storage": _safe_cls_name(kv_storage) if kv_storage else _safe_cls_name(
            getattr(lightrag, "key_string_value_json_storage_cls", None)
        ),
        "vector_storage": _safe_cls_name(vector_storage) if vector_storage else _safe_cls_name(
            getattr(lightrag, "vector_db_storage_cls", None)
        ),
        "doc_status_storage": _safe_cls_name(doc_status_storage) if doc_status_storage else _safe_cls_name(
            getattr(lightrag, "doc_status_storage_cls", None)
        ),
        "graph_storage": _safe_cls_name(graph_storage) if graph_storage else _safe_cls_name(
            getattr(lightrag, "graph_storage_cls", None)
        ),
    }


@dataclass
class BackendTestResult:
    """單一後端測試結果"""

    backend_name: str
    storage_info: Dict[str, str]
    operations: Dict[str, Any] = field(default_factory=dict)
    performance: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


@dataclass
class MigrationTestResult:
    """資料遷移測試結果"""

    source_backend: str
    target_backend: str
    success: bool
    data_integrity: Dict[str, Any] = field(default_factory=dict)
    migration_time: float = 0.0
    errors: List[str] = field(default_factory=list)


async def test_backend_switch(
    backend_name: str,
    working_dir: Path,
    doc_path: Path,
    output_dir: Path,
    llm_model_func,
    embedding_func,
) -> BackendTestResult:
    """測試後端切換：驗證環境變數切換能正確選擇儲存後端"""

    print(f"\n[後端切換測試] {backend_name}")

    # 設定後端環境變數
    if backend_name == "local_filesystem":
        _set_backend_local_env()
    elif backend_name == "postgresql":
        _set_backend_postgres_env()
    else:
        raise ValueError(f"Unknown backend: {backend_name}")

    # 確保工作目錄是空的（用於乾淨測試）
    if working_dir.exists():
        shutil.rmtree(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

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

    # 驗證儲存後端類型
    storage_info = _get_storage_info(rag)
    print(f"  儲存後端類型: {storage_info}")

    result = BackendTestResult(backend_name=backend_name, storage_info=storage_info)

    # 驗證後端類型是否符合預期
    if backend_name == "local_filesystem":
        expected_kv = "JSONStorage"
        expected_doc_status = "JsonDocStatusStorage"
        if expected_kv not in storage_info["kv_storage"]:
            result.errors.append(
                f"KV storage 類型不符合預期: 期望包含 {expected_kv}, 實際為 {storage_info['kv_storage']}"
            )
        if expected_doc_status not in storage_info["doc_status_storage"]:
            result.errors.append(
                f"DocStatus storage 類型不符合預期: 期望包含 {expected_doc_status}, 實際為 {storage_info['doc_status_storage']}"
            )
    elif backend_name == "postgresql":
        expected_kv = "PGKVStorage"
        expected_vector = "PGVectorStorage"
        expected_doc_status = "PGDocStatusStorage"
        if expected_kv not in storage_info["kv_storage"]:
            result.errors.append(
                f"KV storage 類型不符合預期: 期望包含 {expected_kv}, 實際為 {storage_info['kv_storage']}"
            )
        if expected_vector not in storage_info["vector_storage"]:
            result.errors.append(
                f"Vector storage 類型不符合預期: 期望包含 {expected_vector}, 實際為 {storage_info['vector_storage']}"
            )
        if expected_doc_status not in storage_info["doc_status_storage"]:
            result.errors.append(
                f"DocStatus storage 類型不符合預期: 期望包含 {expected_doc_status}, 實際為 {storage_info['doc_status_storage']}"
            )

    return result


async def test_backend_functionality(
    backend_name: str,
    working_dir: Path,
    doc_path: Path,
    output_dir: Path,
    mode: str,
    llm_model_func,
    embedding_func,
) -> BackendTestResult:
    """測試後端功能：驗證所有後端都能正確執行相同的操作"""

    print(f"\n[功能一致性測試] {backend_name}")

    # 設定後端環境變數
    if backend_name == "local_filesystem":
        _set_backend_local_env()
    elif backend_name == "postgresql":
        _set_backend_postgres_env()
    else:
        raise ValueError(f"Unknown backend: {backend_name}")

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

    storage_info = _get_storage_info(rag)
    result = BackendTestResult(backend_name=backend_name, storage_info=storage_info)

    # 測試 1: 處理文件
    print(f"  測試 1: 處理文件...")
    try:
        start = time.perf_counter()
        await rag.process_document_complete(
            file_path=str(doc_path),
            output_dir=str(output_dir),
            parse_method="auto",
        )
        process_time = time.perf_counter() - start
        result.operations["process_document"] = {"success": True, "time": process_time}
        result.performance["process_document"] = process_time
        print(f"    ✓ 成功 ({process_time:.2f}s)")
    except Exception as e:
        result.operations["process_document"] = {"success": False, "error": str(e)}
        result.errors.append(f"處理文件失敗: {str(e)}")
        print(f"    ✗ 失敗: {str(e)}")

    # 測試 2: 插入文字
    print(f"  測試 2: 插入文字...")
    test_text = "這是一個測試文字，用於驗證後端功能。"
    try:
        start = time.perf_counter()
        await rag._ensure_lightrag_initialized()
        await rag.lightrag.ainsert(input=test_text, ids="test-doc-1", file_paths="test.txt")
        await rag.lightrag.finalize_storages()
        insert_time = time.perf_counter() - start
        result.operations["insert_text"] = {"success": True, "time": insert_time}
        result.performance["insert_text"] = insert_time
        print(f"    ✓ 成功 ({insert_time:.2f}s)")
    except Exception as e:
        result.operations["insert_text"] = {"success": False, "error": str(e)}
        result.errors.append(f"插入文字失敗: {str(e)}")
        print(f"    ✗ 失敗: {str(e)}")

    # 測試 3: 查詢（多種模式）
    print(f"  測試 3: 查詢測試...")
    queries = [
        ("Transformer 的核心創新是什麼？", "hybrid"),
        ("Transformer 相對於 RNN 的主要優點是什麼？", "naive"),
        ("Transformer 的優點有哪些？", "local"),
    ]

    for query, query_mode in queries:
        try:
            start = time.perf_counter()
            answer = await rag.aquery(query, mode=query_mode, vlm_enhanced=False)
            query_time = time.perf_counter() - start

            op_key = f"query_{query_mode}"
            result.operations[op_key] = {
                "success": True,
                "time": query_time,
                "answer_length": len(answer) if answer else 0,
            }
            result.performance[op_key] = query_time
            print(f"    ✓ {query_mode} 模式成功 ({query_time:.2f}s, 答案長度: {len(answer) if answer else 0})")
        except Exception as e:
            op_key = f"query_{query_mode}"
            result.operations[op_key] = {"success": False, "error": str(e)}
            result.errors.append(f"查詢 ({query_mode}) 失敗: {str(e)}")
            print(f"    ✗ {query_mode} 模式失敗: {str(e)}")

    return result


async def test_backend_performance(
    backend_name: str,
    working_dir: Path,
    doc_path: Path,
    output_dir: Path,
    mode: str,
    num_iterations: int,
    llm_model_func,
    embedding_func,
) -> BackendTestResult:
    """測試後端效能：比較不同後端的查詢、插入效能"""

    print(f"\n[效能測試] {backend_name} ({num_iterations} 次迭代)")

    # 設定後端環境變數
    if backend_name == "local_filesystem":
        _set_backend_local_env()
    elif backend_name == "postgresql":
        _set_backend_postgres_env()
    else:
        raise ValueError(f"Unknown backend: {backend_name}")

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

    storage_info = _get_storage_info(rag)
    result = BackendTestResult(backend_name=backend_name, storage_info=storage_info)

    # 確保資料已載入
    wd = Path(cfg.working_dir)
    if not (wd.exists() and any(wd.iterdir())):
        print(f"  載入測試資料...")
        await rag.process_document_complete(
            file_path=str(doc_path),
            output_dir=str(output_dir),
            parse_method="auto",
        )

    # 效能測試 1: 查詢效能
    print(f"  測試查詢效能...")
    query = "Transformer 相對於 RNN 的主要優點是什麼？"
    query_times = []

    for i in range(num_iterations):
        try:
            start = time.perf_counter()
            await rag.aquery(query, mode=mode, vlm_enhanced=False)
            elapsed = time.perf_counter() - start
            query_times.append(elapsed)
        except Exception as e:
            result.errors.append(f"查詢迭代 {i+1} 失敗: {str(e)}")

    if query_times:
        sorted_times = sorted(query_times)
        result.performance["query"] = {
            "mean": statistics.mean(query_times),
            "median": statistics.median(query_times),
            "min": min(query_times),
            "max": max(query_times),
            "stdev": statistics.stdev(query_times) if len(query_times) > 1 else 0.0,
            "p95": sorted_times[int(len(query_times) * 0.95)] if len(query_times) > 1 else query_times[0],
            "p99": sorted_times[int(len(query_times) * 0.99)] if len(query_times) > 1 else query_times[0],
            "total": sum(query_times),
            "count": len(query_times),
            "qps": len(query_times) / sum(query_times) if sum(query_times) > 0 else 0,  # Queries per second
        }
        print(
            f"    平均: {result.performance['query']['mean']:.3f}s, "
            f"中位數: {result.performance['query']['median']:.3f}s, "
            f"P95: {result.performance['query']['p95']:.3f}s, "
            f"QPS: {result.performance['query']['qps']:.2f}"
        )

    # 效能測試 2: 插入效能
    print(f"  測試插入效能...")
    insert_times = []
    test_texts = [
        f"效能測試文字 {i}。這是一個用於測試插入效能的範例文字。"
        for i in range(min(num_iterations, 5))  # 限制插入次數以避免過多資料
    ]

    for i, text in enumerate(test_texts):
        try:
            start = time.perf_counter()
            await rag._ensure_lightrag_initialized()
            await rag.lightrag.ainsert(
                input=text, ids=f"perf-test-{i}", file_paths=f"perf-test-{i}.txt"
            )
            await rag.lightrag.finalize_storages()
            elapsed = time.perf_counter() - start
            insert_times.append(elapsed)
        except Exception as e:
            result.errors.append(f"插入迭代 {i+1} 失敗: {str(e)}")

    if insert_times:
        result.performance["insert"] = {
            "mean": statistics.mean(insert_times),
            "median": statistics.median(insert_times),
            "min": min(insert_times),
            "max": max(insert_times),
            "stdev": statistics.stdev(insert_times) if len(insert_times) > 1 else 0.0,
            "total": sum(insert_times),
            "count": len(insert_times),
            "throughput": len(insert_times) / sum(insert_times) if sum(insert_times) > 0 else 0,  # Inserts per second
        }
        print(
            f"    平均: {result.performance['insert']['mean']:.3f}s, "
            f"中位數: {result.performance['insert']['median']:.3f}s, "
            f"吞吐量: {result.performance['insert']['throughput']:.2f} ops/s"
        )

    return result


async def test_data_migration(
    source_backend: str,
    target_backend: str,
    source_working_dir: Path,
    target_working_dir: Path,
    doc_path: Path,
    output_dir: Path,
    mode: str,
    llm_model_func,
    embedding_func,
) -> MigrationTestResult:
    """測試資料遷移：從來源後端讀取資料，寫入目標後端，驗證資料完整性"""

    print(f"\n[資料遷移測試] {source_backend} -> {target_backend}")

    result = MigrationTestResult(
        source_backend=source_backend, target_backend=target_backend, success=False
    )

    # 步驟 1: 在來源後端建立資料
    print(f"  步驟 1: 在來源後端 ({source_backend}) 建立資料...")
    if source_backend == "local_filesystem":
        _set_backend_local_env()
    elif source_backend == "postgresql":
        _set_backend_postgres_env()
    else:
        result.errors.append(f"未知的來源後端: {source_backend}")
        return result

    source_cfg = RAGAnythingConfig(
        working_dir=str(source_working_dir.resolve()),
        parser=os.getenv("PARSER", "mineru"),
        parse_method="auto",
        enable_image_processing=False,
        enable_table_processing=False,
        enable_equation_processing=False,
    )

    source_rag = RAGAnything(
        config=source_cfg,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        vision_model_func=None,
    )

    # 確保來源後端有資料
    source_wd = Path(source_cfg.working_dir)
    if not (source_wd.exists() and any(source_wd.iterdir())):
        await source_rag.process_document_complete(
            file_path=str(doc_path),
            output_dir=str(output_dir),
            parse_method="auto",
        )

    # 確保 LightRAG 已初始化
    await source_rag._ensure_lightrag_initialized()
    source_lightrag = source_rag.lightrag
    
    # 收集來源後端的資料統計
    source_stats = {}
    try:
        # 統計 KV storage 的 key 數量（如果可存取）
        kv_storage = getattr(source_lightrag, "key_string_value_json_storage", None)
        if kv_storage and hasattr(kv_storage, "keys"):
            try:
                keys = await kv_storage.keys() if asyncio.iscoroutinefunction(kv_storage.keys) else kv_storage.keys()
                source_stats["kv_keys_count"] = len(list(keys)) if keys else 0
            except Exception:
                source_stats["kv_keys_count"] = "N/A"
        
        # 統計 Vector storage 的記錄數量（如果可存取）
        vector_storage = getattr(source_lightrag, "vector_db_storage", None)
        if vector_storage:
            try:
                # 嘗試取得向量數量（方法可能因實作而異）
                if hasattr(vector_storage, "count"):
                    count = await vector_storage.count() if asyncio.iscoroutinefunction(vector_storage.count) else vector_storage.count()
                    source_stats["vector_count"] = count
                else:
                    source_stats["vector_count"] = "N/A"
            except Exception:
                source_stats["vector_count"] = "N/A"
        
        # 統計 DocStatus 記錄數量
        doc_status_storage = getattr(source_lightrag, "doc_status_storage", None)
        if doc_status_storage:
            try:
                if hasattr(doc_status_storage, "keys"):
                    doc_keys = await doc_status_storage.keys() if asyncio.iscoroutinefunction(doc_status_storage.keys) else doc_status_storage.keys()
                    source_stats["doc_status_count"] = len(list(doc_keys)) if doc_keys else 0
                else:
                    source_stats["doc_status_count"] = "N/A"
            except Exception:
                source_stats["doc_status_count"] = "N/A"
    except Exception as e:
        print(f"    警告: 無法收集來源後端統計資料: {e}")
    
    result.data_integrity["source_stats"] = source_stats
    print(f"    來源後端資料統計: {source_stats}")
    
    # 在來源後端執行查詢，記錄結果
    source_query = "Transformer 相對於 RNN 的主要優點是什麼？"
    try:
        source_answer = await source_rag.aquery(source_query, mode=mode, vlm_enhanced=False)
        result.data_integrity["source_answer"] = source_answer
        result.data_integrity["source_answer_length"] = len(source_answer) if source_answer else 0
        print(f"    來源後端查詢成功，答案長度: {result.data_integrity['source_answer_length']}")
    except Exception as e:
        result.errors.append(f"來源後端查詢失敗: {str(e)}")
        return result

    # 步驟 2: 在目標後端建立相同資料
    print(f"  步驟 2: 在目標後端 ({target_backend}) 建立資料...")
    if target_backend == "local_filesystem":
        _set_backend_local_env()
    elif target_backend == "postgresql":
        _set_backend_postgres_env()
    else:
        result.errors.append(f"未知的目標後端: {target_backend}")
        return result

    # 確保目標工作目錄是空的
    if target_working_dir.exists():
        shutil.rmtree(target_working_dir)
    target_working_dir.mkdir(parents=True, exist_ok=True)

    target_cfg = RAGAnythingConfig(
        working_dir=str(target_working_dir.resolve()),
        parser=os.getenv("PARSER", "mineru"),
        parse_method="auto",
        enable_image_processing=False,
        enable_table_processing=False,
        enable_equation_processing=False,
    )

    start_migration = time.perf_counter()
    target_rag = RAGAnything(
        config=target_cfg,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        vision_model_func=None,
    )

    # 在目標後端處理相同文件
    await target_rag.process_document_complete(
        file_path=str(doc_path),
        output_dir=str(output_dir),
        parse_method="auto",
    )
    result.migration_time = time.perf_counter() - start_migration

    # 確保目標 LightRAG 已初始化
    await target_rag._ensure_lightrag_initialized()
    target_lightrag = target_rag.lightrag
    
    # 收集目標後端的資料統計
    target_stats = {}
    try:
        # 統計 KV storage 的 key 數量
        kv_storage = getattr(target_lightrag, "key_string_value_json_storage", None)
        if kv_storage and hasattr(kv_storage, "keys"):
            try:
                keys = await kv_storage.keys() if asyncio.iscoroutinefunction(kv_storage.keys) else kv_storage.keys()
                target_stats["kv_keys_count"] = len(list(keys)) if keys else 0
            except Exception:
                target_stats["kv_keys_count"] = "N/A"
        
        # 統計 Vector storage 的記錄數量
        vector_storage = getattr(target_lightrag, "vector_db_storage", None)
        if vector_storage:
            try:
                if hasattr(vector_storage, "count"):
                    count = await vector_storage.count() if asyncio.iscoroutinefunction(vector_storage.count) else vector_storage.count()
                    target_stats["vector_count"] = count
                else:
                    target_stats["vector_count"] = "N/A"
            except Exception:
                target_stats["vector_count"] = "N/A"
        
        # 統計 DocStatus 記錄數量
        doc_status_storage = getattr(target_lightrag, "doc_status_storage", None)
        if doc_status_storage:
            try:
                if hasattr(doc_status_storage, "keys"):
                    doc_keys = await doc_status_storage.keys() if asyncio.iscoroutinefunction(doc_status_storage.keys) else doc_status_storage.keys()
                    target_stats["doc_status_count"] = len(list(doc_keys)) if doc_keys else 0
                else:
                    target_stats["doc_status_count"] = "N/A"
            except Exception:
                target_stats["doc_status_count"] = "N/A"
    except Exception as e:
        print(f"    警告: 無法收集目標後端統計資料: {e}")
    
    result.data_integrity["target_stats"] = target_stats
    print(f"    目標後端資料統計: {target_stats}")
    
    # 步驟 3: 驗證資料完整性
    print(f"  步驟 3: 驗證資料完整性...")
    try:
        target_answer = await target_rag.aquery(source_query, mode=mode, vlm_enhanced=False)
        result.data_integrity["target_answer"] = target_answer
        result.data_integrity["target_answer_length"] = len(target_answer) if target_answer else 0

        # 比較答案長度（因為 LLM 可能產生略有不同的答案，所以只比較長度）
        source_len = result.data_integrity["source_answer_length"]
        target_len = result.data_integrity["target_answer_length"]
        length_diff = abs(source_len - target_len)
        length_diff_percent = (length_diff / source_len * 100) if source_len > 0 else 0

        result.data_integrity["length_diff"] = length_diff
        result.data_integrity["length_diff_percent"] = length_diff_percent
        
        # 比較資料統計
        stats_match = True
        if "source_stats" in result.data_integrity and "target_stats" in result.data_integrity:
            source_stats = result.data_integrity["source_stats"]
            target_stats = result.data_integrity["target_stats"]
            
            # 比較各項統計（允許一定誤差）
            for key in ["kv_keys_count", "vector_count", "doc_status_count"]:
                if key in source_stats and key in target_stats:
                    src_val = source_stats[key]
                    tgt_val = target_stats[key]
                    if isinstance(src_val, (int, float)) and isinstance(tgt_val, (int, float)):
                        if abs(src_val - tgt_val) > max(1, src_val * 0.1):  # 允許 10% 誤差或至少 1 個差異
                            stats_match = False
                            result.errors.append(
                                f"{key} 數量不匹配: 來源={src_val}, 目標={tgt_val}"
                            )

        # 如果長度差異小於 20% 且統計資料匹配，認為遷移成功
        if length_diff_percent < 20 and stats_match:
            result.success = True
            print(
                f"    ✓ 遷移成功 (長度差異: {length_diff_percent:.1f}%, "
                f"遷移時間: {result.migration_time:.2f}s)"
            )
        else:
            if length_diff_percent >= 20:
                result.errors.append(
                    f"答案長度差異過大: {length_diff_percent:.1f}% (來源: {source_len}, 目標: {target_len})"
                )
            if not stats_match:
                result.errors.append("資料統計不匹配")
            print(
                f"    ✗ 遷移可能失敗 (長度差異: {length_diff_percent:.1f}%, 統計匹配: {stats_match})"
            )
    except Exception as e:
        result.errors.append(f"目標後端查詢失敗: {str(e)}")
        print(f"    ✗ 目標後端查詢失敗: {str(e)}")

    return result


async def main(args: argparse.Namespace) -> int:
    _try_load_dotenv()

    doc_path = Path(args.doc).resolve()
    if not doc_path.exists():
        raise FileNotFoundError(str(doc_path))

    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 驗證 API 連接
    api_key = _require_env("LLM_BINDING_API_KEY")
    base_url = os.getenv("LLM_BINDING_HOST")
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    emb_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    
    api_valid = await _validate_api_connection(api_key, base_url, llm_model, emb_model)
    if not api_valid:
        print("\n[錯誤] API 連接驗證失敗，請檢查環境變數設定後再試")
        print("\n建議檢查項目：")
        print(f"  1. LLM_BINDING_API_KEY: {'已設定' if api_key else '未設定'}")
        print(f"  2. LLM_BINDING_HOST: {base_url if base_url else '未設定（使用預設）'}")
        print(f"  3. LLM_MODEL: {llm_model}")
        print(f"  4. EMBEDDING_MODEL: {emb_model}")
        return 1

    llm_model_func, embedding_func = _build_llm_and_embedding()

    # 準備工作目錄
    base_working_dir = Path(args.base_working_dir).resolve()
    local_working_dir = base_working_dir / "local"
    postgres_working_dir = base_working_dir / "postgres"

    results: Dict[str, Any] = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "mode": args.mode,
            "doc": str(doc_path),
            "num_performance_iterations": args.num_performance_iterations,
        },
        "backend_switch": {},
        "functionality": {},
        "performance": {},
        "migration": {},
    }

    # 測試 1: 後端切換測試
    print("\n" + "=" * 80)
    print("測試 1: 後端切換測試")
    print("=" * 80)

    for backend in ["local_filesystem", "postgresql"]:
        try:
            working_dir = local_working_dir if backend == "local_filesystem" else postgres_working_dir
            result = await test_backend_switch(
                backend_name=backend,
                working_dir=working_dir / "switch_test",
                doc_path=doc_path,
                output_dir=output_dir,
                llm_model_func=llm_model_func,
                embedding_func=embedding_func,
            )
            results["backend_switch"][backend] = {
                "storage_info": result.storage_info,
                "errors": result.errors,
                "success": len(result.errors) == 0,
            }
        except Exception as e:
            results["backend_switch"][backend] = {
                "success": False,
                "error": str(e),
            }

    # 測試 2: 功能一致性測試
    print("\n" + "=" * 80)
    print("測試 2: 功能一致性測試")
    print("=" * 80)

    for backend in ["local_filesystem", "postgresql"]:
        try:
            working_dir = local_working_dir if backend == "local_filesystem" else postgres_working_dir
            result = await test_backend_functionality(
                backend_name=backend,
                working_dir=working_dir / "functionality_test",
                doc_path=doc_path,
                output_dir=output_dir,
                mode=args.mode,
                llm_model_func=llm_model_func,
                embedding_func=embedding_func,
            )
            results["functionality"][backend] = {
                "storage_info": result.storage_info,
                "operations": result.operations,
                "errors": result.errors,
                "success": len(result.errors) == 0,
            }
        except Exception as e:
            results["functionality"][backend] = {
                "success": False,
                "error": str(e),
            }

    # 測試 3: 效能測試
    print("\n" + "=" * 80)
    print("測試 3: 效能測試")
    print("=" * 80)

    for backend in ["local_filesystem", "postgresql"]:
        try:
            working_dir = local_working_dir if backend == "local_filesystem" else postgres_working_dir
            result = await test_backend_performance(
                backend_name=backend,
                working_dir=working_dir / "performance_test",
                doc_path=doc_path,
                output_dir=output_dir,
                mode=args.mode,
                num_iterations=args.num_performance_iterations,
                llm_model_func=llm_model_func,
                embedding_func=embedding_func,
            )
            results["performance"][backend] = {
                "storage_info": result.storage_info,
                "performance": result.performance,
                "errors": result.errors,
            }
        except Exception as e:
            results["performance"][backend] = {
                "success": False,
                "error": str(e),
            }

    # 測試 4: 資料遷移測試
    print("\n" + "=" * 80)
    print("測試 4: 資料遷移測試")
    print("=" * 80)

    migration_pairs = [
        ("local_filesystem", "postgresql"),
        ("postgresql", "local_filesystem"),
    ]

    for source, target in migration_pairs:
        try:
            source_working_dir = (
                local_working_dir if source == "local_filesystem" else postgres_working_dir
            )
            target_working_dir = (
                local_working_dir if target == "local_filesystem" else postgres_working_dir
            )

            result = await test_data_migration(
                source_backend=source,
                target_backend=target,
                source_working_dir=source_working_dir / "migration_source",
                target_working_dir=target_working_dir / "migration_target",
                doc_path=doc_path,
                output_dir=output_dir,
                mode=args.mode,
                llm_model_func=llm_model_func,
                embedding_func=embedding_func,
            )

            results["migration"][f"{source}_to_{target}"] = {
                "success": result.success,
                "migration_time": result.migration_time,
                "data_integrity": result.data_integrity,
                "errors": result.errors,
            }
        except Exception as e:
            results["migration"][f"{source}_to_{target}"] = {
                "success": False,
                "error": str(e),
            }

    # 儲存結果
    run_dir = out_root / "storage_backend_comprehensive" / _now_tag()
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # 生成報告
    report_path = run_dir / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("儲存後端綜合測試報告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"測試時間: {results['meta']['timestamp']}\n")
        f.write(f"測試文件: {results['meta']['doc']}\n")
        f.write(f"查詢模式: {results['meta']['mode']}\n\n")

        # 後端切換測試結果
        f.write("-" * 80 + "\n")
        f.write("1. 後端切換測試\n")
        f.write("-" * 80 + "\n")
        for backend, result in results["backend_switch"].items():
            f.write(f"\n{backend}:\n")
            f.write(f"  成功: {result.get('success', False)}\n")
            if "storage_info" in result:
                f.write(f"  儲存類型: {result['storage_info']}\n")
            if result.get("errors"):
                f.write(f"  錯誤: {result['errors']}\n")

        # 功能一致性測試結果
        f.write("\n" + "-" * 80 + "\n")
        f.write("2. 功能一致性測試\n")
        f.write("-" * 80 + "\n")
        for backend, result in results["functionality"].items():
            f.write(f"\n{backend}:\n")
            f.write(f"  成功: {result.get('success', False)}\n")
            if "operations" in result:
                for op_name, op_result in result["operations"].items():
                    status = "✓" if op_result.get("success") else "✗"
                    time_str = f" ({op_result.get('time', 0):.2f}s)" if "time" in op_result else ""
                    f.write(f"  {status} {op_name}{time_str}\n")
            if result.get("errors"):
                f.write(f"  錯誤: {result['errors']}\n")

        # 效能測試結果
        f.write("\n" + "-" * 80 + "\n")
        f.write("3. 效能測試\n")
        f.write("-" * 80 + "\n")
        for backend, result in results["performance"].items():
            f.write(f"\n{backend}:\n")
            if "performance" in result:
                for perf_name, perf_data in result["performance"].items():
                    if isinstance(perf_data, dict):
                        f.write(f"  {perf_name}:\n")
                        f.write(f"    平均: {perf_data.get('mean', 0):.3f}s\n")
                        f.write(f"    中位數: {perf_data.get('median', 0):.3f}s\n")
                        if "p95" in perf_data:
                            f.write(f"    P95: {perf_data.get('p95', 0):.3f}s\n")

        # 資料遷移測試結果
        f.write("\n" + "-" * 80 + "\n")
        f.write("4. 資料遷移測試\n")
        f.write("-" * 80 + "\n")
        for migration_name, result in results["migration"].items():
            f.write(f"\n{migration_name}:\n")
            f.write(f"  成功: {result.get('success', False)}\n")
            if "migration_time" in result:
                f.write(f"  遷移時間: {result['migration_time']:.2f}s\n")
            if "data_integrity" in result:
                integrity = result["data_integrity"]
                f.write(f"  來源答案長度: {integrity.get('source_answer_length', 0)}\n")
                f.write(f"  目標答案長度: {integrity.get('target_answer_length', 0)}\n")
                if "length_diff_percent" in integrity:
                    f.write(f"  長度差異: {integrity['length_diff_percent']:.1f}%\n")
            if result.get("errors"):
                f.write(f"  錯誤: {result['errors']}\n")

    print(f"\n結果已儲存至: {out_path}")
    print(f"報告已儲存至: {report_path}")

    # 輸出摘要
    print("\n" + "=" * 80)
    print("測試摘要")
    print("=" * 80)

    # 後端切換測試摘要
    switch_success = sum(
        1 for r in results["backend_switch"].values() if r.get("success", False)
    )
    print(f"後端切換測試: {switch_success}/{len(results['backend_switch'])} 通過")

    # 功能一致性測試摘要
    func_success = sum(1 for r in results["functionality"].values() if r.get("success", False))
    print(f"功能一致性測試: {func_success}/{len(results['functionality'])} 通過")

    # 效能測試摘要
    print("效能測試:")
    for backend, result in results["performance"].items():
        if "performance" in result:
            if "query" in result["performance"]:
                query_perf = result["performance"]["query"]
                print(f"  {backend} 查詢:")
                print(f"    平均時間: {query_perf.get('mean', 0):.3f}s")
                print(f"    QPS: {query_perf.get('qps', 0):.2f}")
            if "insert" in result["performance"]:
                insert_perf = result["performance"]["insert"]
                print(f"  {backend} 插入:")
                print(f"    平均時間: {insert_perf.get('mean', 0):.3f}s")
                print(f"    吞吐量: {insert_perf.get('throughput', 0):.2f} ops/s")
    
    # 效能比較
    if len(results["performance"]) == 2:
        backends = list(results["performance"].keys())
        if all("performance" in results["performance"][b] and "query" in results["performance"][b]["performance"] 
               for b in backends):
            local_perf = results["performance"].get("local_filesystem", {}).get("performance", {}).get("query", {})
            pg_perf = results["performance"].get("postgresql", {}).get("performance", {}).get("query", {})
            if local_perf and pg_perf:
                local_mean = local_perf.get("mean", 0)
                pg_mean = pg_perf.get("mean", 0)
                if local_mean > 0 and pg_mean > 0:
                    speedup = local_mean / pg_mean
                    print(f"\n  效能比較 (查詢):")
                    print(f"    Local 平均: {local_mean:.3f}s")
                    print(f"    PostgreSQL 平均: {pg_mean:.3f}s")
                    print(f"    速度比: {speedup:.2f}x ({'Local 較快' if speedup > 1 else 'PostgreSQL 較快'})")

    # 資料遷移測試摘要
    migration_success = sum(1 for r in results["migration"].values() if r.get("success", False))
    print(f"資料遷移測試: {migration_success}/{len(results['migration'])} 通過")

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="儲存後端綜合測試")
    p.add_argument("--doc", default="eval/docs/test-01.txt", help="測試文件路徑")
    p.add_argument("--mode", default="hybrid", help="查詢模式")
    p.add_argument("--output-dir", default="eval/integration/output", help="輸出目錄")
    p.add_argument("--out-dir", default="eval/integration/results", help="結果儲存目錄")
    p.add_argument(
        "--base-working-dir",
        default="eval/integration/rag_storage_backend_test",
        help="基礎工作目錄",
    )
    p.add_argument(
        "--num-performance-iterations",
        type=int,
        default=10,
        help="效能測試迭代次數",
    )
    return p


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(build_arg_parser().parse_args())))

