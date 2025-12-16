# RAG-Anything API 驗證測試指南

本指南說明如何驗證 RAG-Anything 作為 API 服務的完整性、穩定性與效能。

## 測試目標

- ✅ 驗證 RAG-Anything 能作為服務提供 API
- ✅ 確認與現有系統的整合介面正確
- ✅ 驗證 API 的穩定性與效能
- ✅ 確認認證與授權機制正確

## 前置需求

### 1. 安裝依賴

```bash
pip install -r eval/integration/requirements-integration.txt
```

### 2. 環境變數設定

建立 `.env` 檔案或設定以下環境變數：

```bash
# LLM 配置
LLM_BINDING_API_KEY=your_api_key
LLM_BINDING_HOST=https://api.openai.com/v1  # 可選
LLM_MODEL=gpt-4o-mini  # 可選，預設值

# Embedding 配置
EMBEDDING_MODEL=text-embedding-3-small  # 可選，預設值
EMBEDDING_DIM=1536  # 可選，預設值

# 儲存配置
WORKING_DIR=./rag_storage  # 可選，預設值
OUTPUT_DIR=./output  # 可選，預設值

# API 認證
RAG_API_KEY=your_secure_api_key  # 必填
```

### 3. 啟動 API 服務

```bash
uvicorn eval.integration.api_service_fastapi:app --host 0.0.0.0 --port 8000
```

服務啟動後，可透過以下端點存取：

- `GET /health` - 健康檢查（不需要認證）
- `GET /v1/stats` - API 統計資訊（需要認證）
- `POST /v1/insert_text` - 插入文字內容（需要認證）
- `POST /v1/process_file` - 處理檔案（需要認證）
- `POST /v1/query` - 查詢知識庫（需要認證）

## 執行測試

### 基本測試

```bash
uv run eval/integration/api_comprehensive_test.py \
  --base-url http://localhost:8000 \
  --api-key your_secure_api_key
```

### 完整測試（包含穩定性測試）

```bash
uv run eval/integration/api_comprehensive_test.py \
  --base-url http://localhost:8000 \
  --api-key your_secure_api_key \
  --test-file eval/docs/test-01.txt \
  --concurrent-requests 20 \
  --duration 600 \
  --output api_test_report.txt
```

### 參數說明

- `--base-url`: API 服務的基礎 URL（預設: <http://localhost:8000）>
- `--api-key`: API 金鑰（必填）
- `--test-file`: 用於測試的檔案路徑（可選）
- `--concurrent-requests`: 並發測試的請求數（預設: 10）
- `--duration`: 穩定性測試持續時間，單位：秒（預設: 300）
- `--output`: 測試報告輸出檔案路徑（可選）
- `--skip-stability`: 跳過穩定性測試（長時間運行）

## 測試內容

### 1. 健康檢查測試

驗證 API 服務是否正常運行。

### 2. 認證與授權測試

- ✅ 正確的 API key 應能正常存取
- ✅ 錯誤的 API key 應返回 401 Unauthorized
- ✅ 缺失的 API key 應返回 401 或 422

### 3. 錯誤處理測試

- ✅ 缺失必要參數應返回 422 Validation Error
- ✅ 無效的查詢模式應返回 400/422/500
- ✅ 不存在的檔案路徑應返回 400

### 4. 整合測試

完整的端到端流程：

1. 插入文字內容到知識庫
2. 執行多種模式的查詢（naive, local, global, hybrid）
3. 處理檔案（如果提供了測試檔案）

### 5. 並發測試

同時發送多個請求，驗證：

- 並發處理能力
- 響應時間分佈
- 成功率

### 6. 效能測試

測量：

- 平均響應時間
- 中位數響應時間
- P95 響應時間
- 標準差

### 7. 穩定性測試（可選）

長時間運行測試，檢查：

- 錯誤率趨勢
- 記憶體使用情況
- 響應時間穩定性

## 測試報告

測試完成後會生成：

1. **文字報告** (`api_test_report_*.txt`): 人類可讀的測試摘要
2. **JSON 報告** (`api_test_report_*.json`): 機器可讀的詳細結果

報告包含：

- 總測試數與成功率
- 各測試套件的詳細結果
- 失敗測試的錯誤訊息
- 效能指標（響應時間、吞吐量等）

## 範例輸出

```
[測試 1/6] 健康檢查測試
結果: ✓ 通過

[測試 2/6] 認證與授權測試
結果: 3/3 通過

[測試 3/6] 錯誤處理測試
結果: 3/3 通過

[測試 4/6] 整合測試
  ✓ 健康檢查: 成功 (0.012s)
  ✓ 插入文字: 成功 (2.345s)
  ✓ 查詢 (hybrid): 成功 (3.456s, 答案長度: 234)
  ✓ 查詢 (naive): 成功 (2.123s, 答案長度: 198)
  ✓ 查詢 (local): 成功 (2.567s, 答案長度: 201)
結果: 5/5 通過

[測試 5/6] 並發測試
  完成時間: 15.234s
  成功率: 100.0%
  平均響應時間: 1.523s
  P95 響應時間: 2.345s
結果: 10/10 通過

[測試 6/6] 效能測試
  平均響應時間: 1.456s
  中位數響應時間: 1.423s
  最小響應時間: 1.234s
  最大響應時間: 2.123s
  標準差: 0.234s
  P95 響應時間: 1.987s
結果: 20/20 通過

================================================================================
整體測試結果: 61/61 通過 (100.0%)
================================================================================
```

## 故障排除

### API 服務無法啟動

1. 檢查環境變數是否正確設定
2. 確認端口 8000 未被占用
3. 檢查依賴是否完整安裝

### 認證失敗

1. 確認 `RAG_API_KEY` 環境變數已設定
2. 確認測試腳本使用的 API key 與服務端一致
3. 檢查請求標頭是否正確包含 `X-API-Key`

### 查詢超時

1. 增加 `--timeout` 參數值
2. 檢查 LLM API 是否正常
3. 檢查網路連線

### 記憶體不足

1. 減少 `--concurrent-requests` 數量
2. 縮短 `--duration` 時間
3. 檢查系統資源使用情況

## 進階使用

### 自訂測試

可以修改 `api_comprehensive_test.py` 以加入自訂測試案例。

### 整合到 CI/CD

```yaml
# GitHub Actions 範例
- name: Run API Tests
  run: |
    uvicorn eval.integration.api_service_fastapi:app --host 0.0.0.0 --port 8000 &
    sleep 5
    python eval/integration/api_comprehensive_test.py \
      --base-url http://localhost:8000 \
      --api-key ${{ secrets.RAG_API_KEY }} \
      --skip-stability
```

### 監控與告警

測試報告的 JSON 格式可整合到監控系統，設定告警規則：

- 成功率 < 95%
- 平均響應時間 > 5s
- P95 響應時間 > 10s

## 相關文件

- [API 服務實作](api_service_fastapi.py)
- [簡單客戶端範例](api_client_example.py)
- [容器化測試](container_api_smoketest.py)
