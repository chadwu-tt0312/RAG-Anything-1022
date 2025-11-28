# RAG-Anything 使用說明

**專案網頁**：<https://github.com/HKUDS/RAG-Anything>  
**參考網頁**：<https://deepwiki.com/HKUDS/RAG-Anything>  
**目標讀者**：負責部署與維護的技術人員
**最後更新**：2025年11月

---

## 1. 簡介

RAG-Anything 是一個綜合性多模態文件處理 RAG 系統，旨在無縫處理和查詢包含文本、圖像、表格、公式等多模態內容的複雜文件。它基於 LightRAG 架構，結合了 MinerU 的高精度解析能力，提供從文件解析、知識圖譜構建到多模態檢索的完整解決方案。

**核心架構**：
- **文件解析**：整合 MinerU，支援 PDF、Office、圖片等格式。
- **知識圖譜**：自動提取實體與關係，建立跨模態語義連接。
- **檢索生成**：支援向量與圖譜混合檢索，並具備 VLM 增強查詢能力。

---

## 2. 環境變數詳解 (Environment Variables)

以下為 RAG-Anything 的主要環境變數配置，請參考 `.env.example` 建立您的 `.env` 檔案。

| 變數名稱 | 預設值 | 必填 | 說明 |
|---------|--------|------|------|
| **Server Configuration** | | | |
| `HOST` | `0.0.0.0` | 否 | 服務監聽位址 |
| `PORT` | `9621` | 否 | 服務埠號 |
| **LLM Configuration** | | | |
| `LLM_BINDING` | `openai` | 是 | LLM 服務提供者 (openai, azure_openai, ollama 等) |
| `LLM_MODEL` | `gpt-4o` | 是 | 使用的模型名稱 |
| `LLM_BINDING_API_KEY` | - | 是 | LLM API 金鑰 |
| `LLM_BINDING_HOST` | `https://api.openai.com/v1` | 否 | LLM API 基礎 URL |
| `MAX_TOKENS` | `32768` | 否 | 最大 Token 數 |
| **Azure OpenAI Specific** | | | |
| `AZURE_OPENAI_API_VERSION` | - | 若用 Azure 則必填 | Azure OpenAI API 版本 (如 `2024-08-01-preview`) |
| `AZURE_OPENAI_DEPLOYMENT` | - | 若用 Azure 則必填 | Azure OpenAI 部署名稱 |
| **Embedding Configuration** | | | |
| `EMBEDDING_BINDING` | `ollama` | 是 | Embedding 服務提供者 |
| `EMBEDDING_MODEL` | `bge-m3:latest` | 是 | Embedding 模型名稱 |
| `EMBEDDING_BINDING_API_KEY`| - | 視情況 | Embedding API 金鑰 |
| `EMBEDDING_BINDING_HOST` | `http://localhost:11434` | 否 | Embedding API URL |
| **Storage Configuration** | | | |
| `INPUT_DIR` | `./inputs` | 否 | 輸入文件目錄 |
| `POSTGRES_HOST` | `localhost` | 否 | PostgreSQL 主機 (若使用 PG) |
| `NEO4J_URI` | - | 否 | Neo4j 連線 URI (若使用 Neo4j) |

---

## 3. 安裝與部署 (Installation & Deployment)

### 本機安裝 (Local Installation)

由於專案包含 `setup.py` 與 `requirements.txt`，建議使用 `pip` 進行安裝。

**前置需求**：
- Python >= 3.9
- LibreOffice (若需處理 Office 文件)

**安裝步驟**：

1. **複製專案代碼**：
    ```bash
    git clone https://github.com/HKUDS/RAG-Anything.git
    cd RAG-Anything
    ```

2. **建立虛擬環境 (建議)**：
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate
    ```

3. **安裝依賴套件**：
    ```bash
    # 基礎安裝
    pip install -e .

    # 安裝所有選用功能 (包含圖片與文字處理)
    pip install -e ".[all]"
    ```

4. **驗證 MinerU 安裝**：
    ```bash
    mineru --version
    ```

### 容器化部署 (Docker)

目前專案根目錄未提供 `Dockerfile`，若需容器化部署，建議自行建立 Docker 映像檔，或等待官方更新。

---

## 4. 操作指南 (Operations)

### 基本操作

**啟動服務**：
雖然專案主要作為 Python 函式庫使用，但若有 WebUI 或 API Server (參考 `.env` 中的 `PORT` 設定)，通常透過以下方式啟動 (需確認具體入口文件，如 `main.py` 或 `server.py`)：

```bash
# 假設入口為 raganything 模組或範例腳本
python examples/your_script.py
```

**範例：端到端文件處理**：

```python
import asyncio
from raganything import RAGAnything, RAGAnythingConfig

# ... (配置 LLM 與 Embedding 函數，參考 README)

async def main():
    # 初始化
    rag = RAGAnything(config=config, ...)
    
    # 處理文件
    await rag.process_document_complete(
        file_path="path/to/your/document.pdf",
        output_dir="./output"
    )
    
    # 查詢
    result = await rag.aquery("文件內容是什麼？", mode="hybrid")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### 進階設定 (LLM Integration)

#### Azure OpenAI 整合

若需使用 Azure OpenAI，請在 `.env` 檔案或環境變數中進行以下設定：

1. **修改 `LLM_BINDING`**：
    ```env
    LLM_BINDING=azure_openai
    ```

2. **設定 Azure 專屬變數**：
    ```env
    LLM_BINDING_HOST=https://<your-resource-name>.openai.azure.com
    LLM_BINDING_API_KEY=<your-azure-api-key>
    AZURE_OPENAI_API_VERSION=2024-08-01-preview
    AZURE_OPENAI_DEPLOYMENT=<your-deployment-name>
    ```

3. **程式碼調用確認**：
    確保在初始化 `llm_model_func` 時，正確傳遞 `base_url` (對應 Endpoint) 與 `api_key`。

### 故障排除 (Troubleshooting)

1. **Azure 連線錯誤 (401/404)**：
    - 檢查 `LLM_BINDING_HOST` 是否包含完整的 Endpoint URL。
    - 確認 `AZURE_OPENAI_DEPLOYMENT` 名稱是否與 Azure Portal 上的一致。
    - 確認 `AZURE_OPENAI_API_VERSION` 是否為支援的版本。

2. **MinerU 解析失敗**：
    - 確保已安裝 `magic-pdf` (MinerU 的核心套件)。
    - 若處理 Office 文件失敗，請確認系統路徑中可找到 `soffice` (LibreOffice) 指令。

3. **Python 套件相依性衝突**：
    - 建議使用 `uv` 或 `poetry` 進行依賴管理，以鎖定版本。
    - 檢查 `setup.py` 中的 `install_requires` 列表，確認是否有版本衝突。

---

## 5. 範例與截圖 (Examples)

> [圖片說明：此處應顯示 RAG-Anything 處理多模態文件後的知識圖譜視覺化結果，展示圖片與文字實體的關聯]

**多模態查詢範例**：

當查詢包含圖片或表格數據時，RAG-Anything 可進行聯合分析：

**輸入**：
```python
multimodal_result = await rag.aquery_with_multimodal(
    "分析這個性能數據並解釋與現有文件內容的關係",
    multimodal_content=[{
        "type": "table",
        "table_data": """系統,准确率
                        RAGAnything,95.2%
                        基准,87.3%""",
        "table_caption": "性能对比结果"
    }],
    mode="hybrid"
)
```

**輸出**：
> 根據提供的性能對比表格，RAGAnything 的準確率達到 95.2%，顯著優於基準方法的 87.3%。這與文件中提到的「高精度解析平台」描述相符...
