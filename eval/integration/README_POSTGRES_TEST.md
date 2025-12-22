# PostgreSQL 儲存測試說明

## 概述

`test_postgres_storage_simple.py` 是一個專門用於驗證資料是否能正確儲存到 PostgreSQL 的測試程式。

## 前置需求

### 1. PostgreSQL 資料庫

確保 PostgreSQL 正在運行，並且已建立必要的表格：

```bash
# 初始化 PostgreSQL 表格
python eval/integration/init_postgres_tables.py
```

### 2. 環境變數設定

設定以下環境變數：

```bash
# PostgreSQL 連線資訊
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_USER=rag_user
export POSTGRES_PASSWORD=rag_password
export POSTGRES_DATABASE=rag_db  # 或使用 POSTGRES_DB

# LightRAG Storage 選擇（必須設定為 PostgreSQL）
export LIGHTRAG_KV_STORAGE=PGKVStorage
export LIGHTRAG_VECTOR_STORAGE=PGVectorStorage
export LIGHTRAG_DOC_STATUS_STORAGE=PGDocStatusStorage

# LLM 和 Embedding API
export LLM_BINDING_API_KEY=your_api_key
export LLM_BINDING_HOST=https://api.openai.com/v1
export LLM_MODEL=gpt-4o-mini
export EMBEDDING_MODEL=text-embedding-3-small
export EMBEDDING_DIM=1536
```

或者使用 `.env` 檔案：

```bash
# 從 .env 載入環境變數
cp env.example .env
# 編輯 .env 檔案填入正確的值
```

## 執行測試

```bash
python eval/integration/test_postgres_storage_simple.py \
    --doc eval/docs/test-01.txt \
    --working-dir eval/integration/rag_storage_postgres_test \
    --output-dir eval/integration/output
```

## 測試步驟

程式會依序執行以下步驟：

1. **測試 PostgreSQL 連線**
   - 驗證能否連接到 PostgreSQL 資料庫

2. **驗證環境變數設定**
   - 檢查所有必要的環境變數是否已設定
   - 如果未設定 PostgreSQL storage 環境變數，會自動設定

3. **驗證儲存後端類型**
   - 確認 RAGAnything 實際使用的 storage 類別
   - 驗證是否真的使用 PostgreSQL storage（PGKVStorage, PGVectorStorage, PGDocStatusStorage）

4. **處理文件**
   - 處理指定的文件並生成 doc_id

5. **驗證資料是否寫入 PostgreSQL**
   - 直接查詢 PostgreSQL 資料庫
   - 檢查 `lightrag_doc_status_storage` 表格
   - 檢查 `lightrag_kv_storage` 表格
   - 檢查 `lightrag_vector_storage_*` 表格

6. **測試查詢功能**
   - 執行查詢驗證資料是否能正確檢索

## 常見問題

### 問題 1: 顯示使用 JsonDocStatusStorage 而非 PGDocStatusStorage

**原因**：環境變數未正確設定或未被 LightRAG 讀取

**解決方法**：
1. 確認環境變數已正確設定：
   ```bash
   echo $LIGHTRAG_DOC_STATUS_STORAGE  # 應該顯示 PGDocStatusStorage
   ```
2. 確認在執行 Python 程式前已設定環境變數
3. 如果使用 `.env` 檔案，確認 `load_dotenv()` 已正確載入

### 問題 2: PostgreSQL 連線失敗

**檢查項目**：
1. PostgreSQL 是否正在運行
2. 連線資訊（host, port, user, password, database）是否正確
3. 防火牆設定是否允許連線
4. 使用者是否有權限存取資料庫

### 問題 3: 資料未寫入 PostgreSQL

**檢查項目**：
1. 確認 storage 後端類型正確（步驟 3）
2. 檢查 PostgreSQL 表格是否存在：
   ```bash
   python eval/integration/check_postgres_storage.py
   ```
3. 檢查是否有錯誤訊息

### 問題 4: 查詢返回 "[no-context]"

**原因**：資料未正確寫入或索引未建立

**解決方法**：
1. 確認資料已寫入 PostgreSQL（步驟 5）
2. 確認 vector storage 中有資料
3. 檢查 embedding 是否正確生成

## 驗證資料

可以使用 `check_postgres_storage.py` 來檢查 PostgreSQL 中的資料：

```bash
python eval/integration/check_postgres_storage.py
```

或直接連接到 PostgreSQL：

```bash
psql -h localhost -U rag_user -d rag_db

# 查詢 doc_status
SELECT * FROM lightrag_doc_status_storage;

# 查詢 KV storage
SELECT COUNT(*) FROM lightrag_kv_storage;

# 查詢 vector storage
SELECT COUNT(*) FROM lightrag_vector_storage_chunks;
```

## 輸出說明

測試成功時會顯示：

```
================================================================================
PostgreSQL 儲存測試
================================================================================

[步驟 1] 測試 PostgreSQL 連線...
  ✓ PostgreSQL 連線成功
    主機: localhost:5432
    資料庫: rag_db
    使用者: rag_user

[步驟 2] 驗證環境變數設定...
  ✓ LIGHTRAG_KV_STORAGE: PGKVStorage
  ✓ LIGHTRAG_VECTOR_STORAGE: PGVectorStorage
  ✓ LIGHTRAG_DOC_STATUS_STORAGE: PGDocStatusStorage

[步驟 3] 驗證儲存後端類型...
  KV Storage: PGKVStorage
  Vector Storage: PGVectorStorage
  DocStatus Storage: PGDocStatusStorage
  ✓ 確認使用 PostgreSQL storage

[步驟 4] 處理文件...
  ✓ 文件處理完成

[步驟 5] 驗證資料是否寫入 PostgreSQL...
  ✓ 找到 doc_status 記錄
  ✓ 找到 KV 記錄
  ✓ 找到 vector 記錄
  ✓ 驗證成功：資料已寫入 PostgreSQL

[步驟 6] 測試查詢功能...
  ✓ 查詢成功

================================================================================
✅ 所有測試通過！PostgreSQL 儲存功能正常
================================================================================
```

## 相關檔案

- `test_postgres_storage_simple.py` - 主要測試程式
- `init_postgres_tables.py` - 初始化 PostgreSQL 表格
- `check_postgres_storage.py` - 檢查 PostgreSQL 儲存狀態
- `storage_backend_comprehensive_test.py` - 完整的後端測試（包含效能測試）

