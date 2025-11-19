# LightRAG 使用指南

## 概述

RAG-Anything 是基於 **LightRAG** 構建的多模態文件處理系統。LightRAG 提供核心的知識圖譜（Knowledge Graph）和檢索增強生成（RAG）功能，而 RAG-Anything 在此基礎上擴展了多模態內容處理能力。

## LightRAG 在 RAG-Anything 中的角色

### 1. 核心功能提供者

LightRAG 為 RAG-Anything 提供以下核心功能：

- **知識圖譜管理**：儲存和管理實體（Entities）與關係（Relations）
- **向量檢索**：基於 embedding 的相似度搜尋
- **文件處理**：文字內容的實體關係提取與合併
- **查詢引擎**：支援多種查詢模式（naive, local, global, hybrid）

### 2. 儲存後端

LightRAG 管理所有資料儲存：

- **KV Storage**：鍵值儲存（文件、chunks、實體、關係、快取）
- **Vector Storage**：向量資料庫（embedding 向量）
- **Graph Storage**：圖譜儲存（GraphML 格式）
- **Doc Status Storage**：文件處理狀態追蹤

## 使用方式

### 方式一：自動初始化（推薦）

RAG-Anything 會自動建立和管理 LightRAG 實例：

```python
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

# 配置
config = RAGAnythingConfig(
    working_dir="./rag_storage",  # LightRAG 工作目錄
    parser="mineru",
    enable_image_processing=True,
    enable_table_processing=True,
    enable_equation_processing=True,
)

# 定義模型函數
def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )

embedding_func = EmbeddingFunc(
    embedding_dim=1536,  # 或 3072
    max_token_size=8192,
    func=lambda texts: openai_embed(
        texts,
        model="text-embedding-3-small",  # 或 text-embedding-3-large
        api_key=api_key,
        base_url=base_url,
    ),
)

# 初始化 RAGAnything（自動建立 LightRAG 實例）
rag = RAGAnything(
    config=config,
    llm_model_func=llm_model_func,
    embedding_func=embedding_func,
)

# 處理文件（自動使用 LightRAG）
await rag.process_document_complete(
    file_path="document.pdf",
    output_dir="./output"
)

# 查詢（使用 LightRAG 的查詢功能）
result = await rag.aquery("What is the main content?", mode="hybrid")
```

### 方式二：使用現有 LightRAG 實例

如果您已經有 LightRAG 實例，可以直接傳入：

```python
from lightrag import LightRAG
from raganything import RAGAnything

# 建立或載入現有的 LightRAG 實例
lightrag_instance = LightRAG(
    working_dir="./rag_storage",
    llm_model_func=llm_model_func,
    embedding_func=embedding_func,
)

# 初始化儲存
await lightrag_instance.initialize_storages()

# 使用現有 LightRAG 實例初始化 RAGAnything
rag = RAGAnything(
    lightrag=lightrag_instance,  # 傳入現有實例
    vision_model_func=vision_model_func,
    # 注意：working_dir、llm_model_func、embedding_func 等從 lightrag_instance 繼承
)

# 現在可以使用 RAGAnything 的多模態功能
await rag.process_document_complete(
    file_path="multimodal_document.pdf",
    output_dir="./output"
)
```

### 方式三：自訂 LightRAG 參數

透過 `lightrag_kwargs` 傳遞額外的 LightRAG 配置：

```python
rag = RAGAnything(
    config=config,
    llm_model_func=llm_model_func,
    embedding_func=embedding_func,
    lightrag_kwargs={
        # 檢索參數
        "top_k": 60,
        "chunk_top_k": 5,
        "cosine_threshold": 0.2,
        
        # Token 限制
        "max_entity_tokens": 4000,
        "max_relation_tokens": 4000,
        "max_total_tokens": 32768,
        
        # Chunk 設定
        "chunk_token_size": 1200,
        "chunk_overlap_token_size": 100,
        
        # 並行處理
        "max_parallel_insert": 2,
        "embedding_func_max_async": 16,
        "llm_model_max_async": 4,
        
        # 圖譜限制
        "max_graph_nodes": 1000,
        
        # 快取設定
        "enable_llm_cache": True,
    }
)
```

## LightRAG 查詢模式

RAG-Anything 支援 LightRAG 的所有查詢模式：

### 1. Naive 模式

純文字查詢，不進行知識圖譜檢索：

```python
result = await rag.aquery("簡單問題", mode="naive")
```

### 2. Local 模式

基於實體節點的局部查詢：

```python
result = await rag.aquery("查詢特定實體", mode="local")
```

### 3. Global 模式

基於關係邊的全域查詢：

```python
result = await rag.aquery("查詢關係", mode="global")
```

### 4. Hybrid 模式（推薦）

結合 local 和 global 的混合查詢：

```python
result = await rag.aquery("複雜查詢", mode="hybrid")
```

## 直接使用 LightRAG API

如果需要直接存取 LightRAG 的功能：

```python
# 存取 LightRAG 實例
lightrag = rag.lightrag

# 直接使用 LightRAG 的方法
# 插入純文字
await lightrag.insert(
    text="純文字內容",
    doc_id="doc-123"
)

# 直接查詢
result = await lightrag.aquery(
    "查詢問題",
    param=QueryParam(mode="hybrid")
)

# 存取儲存
entities = await lightrag.kg.get_all_entities()
relations = await lightrag.kg.get_all_relations()
```

## 儲存結構

LightRAG 在 `working_dir` 中使用扁平化的 JSON 檔案結構（而非目錄結構）：

```
rag_storage/
├── graph_chunk_entity_relation.graphml  # 知識圖譜（GraphML 格式）
│
├── kv_store_full_docs.json              # 完整文件內容
├── kv_store_text_chunks.json            # 文字區塊
├── kv_store_full_entities.json          # 完整實體資訊
├── kv_store_full_relations.json         # 完整關係資訊
├── kv_store_entity_chunks.json          # 實體相關區塊
├── kv_store_relation_chunks.json        # 關係相關區塊
├── kv_store_llm_response_cache.json     # LLM 回應快取
├── kv_store_doc_status.json             # 文件處理狀態
├── kv_store_parse_cache.json            # 解析結果快取（RAG-Anything 擴展）
│
├── vdb_chunks.json                      # Chunks 向量資料庫
├── vdb_entities.json                     # Entities 向量資料庫
└── vdb_relationships.json                # Relationships 向量資料庫
```

### 檔案說明

**KV Storage（鍵值儲存）**：
- `kv_store_*.json`：所有鍵值資料以 JSON 格式儲存
- 每個檔案包含一個 JSON 物件，鍵為 ID，值為對應的資料

**Vector Database（向量資料庫）**：
- `vdb_*.json`：向量資料以 JSON 格式儲存
- 包含 `embedding_dim`（向量維度）和 `data`（向量資料陣列）
- `matrix`：向量矩陣（用於快速相似度計算）

**Graph Storage（圖譜儲存）**：
- `graph_chunk_entity_relation.graphml`：知識圖譜的 GraphML 格式檔案
- 包含所有節點（nodes）和邊（edges）的結構化資訊

**注意**：實際儲存使用扁平化的 JSON 檔案，而非目錄結構。這種設計簡化了檔案管理，所有相關資料都在同一目錄中。

## 環境變數配置

LightRAG 支援透過環境變數配置，這些變數也會被 RAG-Anything 使用：

```bash
# LLM 配置
LLM_BINDING=openai
LLM_MODEL=gpt-4o
LLM_BINDING_HOST=https://api.openai.com/v1
LLM_BINDING_API_KEY=your_api_key

# Embedding 配置
EMBEDDING_BINDING=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
EMBEDDING_BINDING_HOST=https://api.openai.com/v1
EMBEDDING_BINDING_API_KEY=your_api_key

# 查詢參數
TOP_K=60
COSINE_THRESHOLD=0.2
MAX_TOKEN_TEXT_CHUNK=4000

# Chunk 設定
CHUNK_SIZE=1200
CHUNK_OVERLAP_SIZE=100

# 並行處理
MAX_PARALLEL_INSERT=2
MAX_ASYNC=4
```

## 最佳實踐

### 1. 工作目錄管理

- 每個專案使用獨立的 `working_dir`
- 定期備份 `rag_storage` 目錄
- 更換 embedding 模型時，刪除舊的 `rag_storage` 重新開始

### 2. Embedding 維度一致性

確保 `EMBEDDING_DIM` 與實際使用的 embedding 模型匹配：

- `text-embedding-3-small` → 1536 維度
- `text-embedding-3-large` → 3072 維度

### 3. 查詢模式選擇

- **簡單問題**：使用 `naive` 或 `local`
- **複雜查詢**：使用 `hybrid`（推薦）
- **關係查詢**：使用 `global`

### 4. 效能優化

- 啟用 LLM 快取：`enable_llm_cache=True`
- 調整並行度：`max_parallel_insert`、`llm_model_max_async`
- 使用適當的 `chunk_token_size`（建議 500-1500）

## 故障排除

### 問題：向量維度不匹配

**錯誤訊息**：
```
ValueError: all the input array dimensions except for the concatenation axis must match exactly
```

**解決方案**：
1. 檢查 `.env` 中的 `EMBEDDING_DIM` 設定
2. 確保與實際使用的 embedding 模型匹配
3. 如果仍有問題，刪除 `rag_storage` 目錄重新開始

### 問題：LightRAG 未初始化

**錯誤訊息**：
```
No LightRAG instance available
```

**解決方案**：
1. 確保提供了 `llm_model_func` 和 `embedding_func`
2. 或傳入已初始化的 `lightrag` 實例
3. 檢查 `working_dir` 權限

### 問題：查詢結果為空

**可能原因**：
1. 文件尚未處理完成
2. `cosine_threshold` 設定過高
3. `top_k` 設定過小

**解決方案**：
1. 確認文件處理完成
2. 降低 `cosine_threshold`（例如 0.1）
3. 增加 `top_k`（例如 100）

## 參考資源

- [LightRAG GitHub](https://github.com/HKUDS/LightRAG)
- [RAG-Anything 文檔](../README.md)
- [架構關係圖](./lightrag_architecture.md)

