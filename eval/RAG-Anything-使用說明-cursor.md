# RAG-Anything 使用說明

**文件版本**：v1.2.8  
**最後更新**：2025年11月
**專案網頁**：<https://github.com/HKUDS/RAG-Anything>  
**參考網頁**：<https://deepwiki.com/HKUDS/RAG-Anything>  
**目標讀者**：負責部署與維護的技術人員

---

## 1. 簡介

### 專案概述

RAG-Anything 是一個基於 [LightRAG](https://github.com/HKUDS/LightRAG) 構建的綜合性多模態文件處理 RAG（Retrieval-Augmented Generation）系統。它能夠無縫處理包含文字、圖片、表格、數學公式等多模態內容的複雜文件，並提供完整的檢索增強生成解決方案。

### 核心功能

- **端到端多模態處理流水線**：從文件解析到多模態查詢響應的完整處理鏈
- **多格式文件支援**：PDF、Office 文件（DOC/DOCX/PPT/PPTX/XLS/XLSX）、圖片（JPG/PNG/BMP/TIFF/GIF/WebP）、文字檔（TXT/MD）
- **多模態內容分析**：針對圖片、表格、公式部署專用處理器
- **知識圖譜索引**：自動實體提取與跨模態關係建立
- **智慧檢索**：支援向量相似度搜尋、圖譜遍歷、混合檢索模式

### 系統架構

RAG-Anything 採用分層架構：

1. **文件解析層**：使用 MinerU 或 Docling 解析器提取文件內容
2. **多模態處理層**：透過專用處理器分析圖片、表格、公式
3. **知識圖譜層**：基於 LightRAG 建立實體與關係圖譜
4. **檢索層**：提供向量檢索與圖譜檢索能力
5. **查詢層**：支援純文字查詢、VLM 增強查詢、多模態查詢

---

## 2. 環境變數詳解 (Environment Variables)

### RAG-Anything 專用環境變數

| 變數名稱 | 預設值 | 必填 | 說明 |
|---------|--------|------|------|
| `WORKING_DIR` | `./rag_storage` | 否 | RAG 儲存與快取檔案目錄 |
| `PARSE_METHOD` | `auto` | 否 | 解析方法：`auto`、`ocr`、`txt` |
| `OUTPUT_DIR` | `./output` | 否 | 解析內容的預設輸出目錄 |
| `PARSER` | `mineru` | 否 | 解析器選擇：`mineru` 或 `docling` |
| `DISPLAY_CONTENT_STATS` | `true` | 否 | 是否顯示內容統計資訊 |
| `ENABLE_IMAGE_PROCESSING` | `true` | 否 | 啟用圖片內容處理 |
| `ENABLE_TABLE_PROCESSING` | `true` | 否 | 啟用表格內容處理 |
| `ENABLE_EQUATION_PROCESSING` | `true` | 否 | 啟用公式內容處理 |
| `MAX_CONCURRENT_FILES` | `1` | 否 | 批次處理的最大並發檔案數 |
| `SUPPORTED_FILE_EXTENSIONS` | `.pdf,.jpg,...` | 否 | 支援的檔案副檔名（逗號分隔） |
| `RECURSIVE_FOLDER_PROCESSING` | `true` | 否 | 批次處理時是否遞迴掃描子資料夾 |
| `CONTEXT_WINDOW` | `1` | 否 | 上下文視窗大小（頁數或 chunk 數） |
| `CONTEXT_MODE` | `page` | 否 | 上下文模式：`page` 或 `chunk` |
| `MAX_CONTEXT_TOKENS` | `2000` | 否 | 提取上下文的最大 token 數 |
| `INCLUDE_HEADERS` | `true` | 否 | 是否包含文件標題與標頭 |
| `INCLUDE_CAPTIONS` | `true` | 否 | 是否包含圖片/表格標題 |
| `CONTEXT_FILTER_CONTENT_TYPES` | `text` | 否 | 上下文過濾的內容類型（逗號分隔） |
| `CONTENT_FORMAT` | `minerU` | 否 | 內容格式：`minerU`、`text_chunks`、`auto` |

### LLM 整合環境變數

#### OpenAI API

| 變數名稱 | 預設值 | 必填 | 說明 |
|---------|--------|------|------|
| `LLM_BINDING` | `openai` | 是 | LLM 綁定類型 |
| `LLM_MODEL` | `gpt-4o` | 是 | LLM 模型名稱 |
| `LLM_BINDING_HOST` | `https://api.openai.com/v1` | 是 | API 端點 URL |
| `LLM_BINDING_API_KEY` | - | 是 | OpenAI API Key |
| `TEMPERATURE` | `0` | 否 | 模型溫度參數 |
| `MAX_TOKENS` | `32768` | 否 | 最大 token 數 |
| `MAX_ASYNC` | `4` | 否 | 最大並發請求數 |
| `TIMEOUT` | `240` | 否 | 請求超時時間（秒） |
| `ENABLE_LLM_CACHE` | `true` | 否 | 啟用 LLM 快取 |

#### Azure OpenAI 整合

| 變數名稱 | 預設值 | 必填 | 說明 |
|---------|--------|------|------|
| `LLM_BINDING` | `azure_openai` | 是 | 設定為 `azure_openai` |
| `LLM_BINDING_HOST` | - | 是 | Azure OpenAI 端點（格式：`https://<resource-name>.openai.azure.com`） |
| `LLM_BINDING_API_KEY` | - | 是 | Azure OpenAI API Key |
| `AZURE_OPENAI_API_VERSION` | `2024-08-01-preview` | 是 | Azure OpenAI API 版本 |
| `AZURE_OPENAI_DEPLOYMENT` | - | 是 | Azure OpenAI Deployment Name（例如：`gpt-4o`） |
| `LLM_MODEL` | - | 否 | 模型名稱（通常與 Deployment Name 相同） |

**Azure OpenAI 設定範例**：

```bash
LLM_BINDING=azure_openai
LLM_BINDING_HOST=https://your-resource.openai.azure.com
LLM_BINDING_API_KEY=your-azure-api-key
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o
LLM_MODEL=gpt-4o
```

#### Embedding 整合

| 變數名稱 | 預設值 | 必填 | 說明 |
|---------|--------|------|------|
| `EMBEDDING_BINDING` | `ollama` | 是 | Embedding 綁定類型：`openai`、`ollama`、`azure_openai`、`lmstudio` |
| `EMBEDDING_MODEL` | `bge-m3:latest` | 是 | Embedding 模型名稱 |
| `EMBEDDING_DIM` | `1024` | 是 | Embedding 維度 |
| `EMBEDDING_BINDING_HOST` | `http://localhost:11434` | 是 | Embedding 服務端點 |
| `EMBEDDING_BINDING_API_KEY` | - | 否 | Embedding API Key（視服務而定） |
| `AZURE_EMBEDDING_DEPLOYMENT` | - | 否 | Azure Embedding Deployment Name |
| `AZURE_EMBEDDING_API_VERSION` | `2023-05-15` | 否 | Azure Embedding API 版本 |

### 資料庫後端環境變數

#### PostgreSQL

| 變數名稱 | 預設值 | 必填 | 說明 |
|---------|--------|------|------|
| `POSTGRES_HOST` | `localhost` | 是 | PostgreSQL 主機位址 |
| `POSTGRES_PORT` | `5432` | 是 | PostgreSQL 埠號 |
| `POSTGRES_USER` | - | 是 | 資料庫使用者名稱 |
| `POSTGRES_PASSWORD` | - | 是 | 資料庫密碼 |
| `POSTGRES_DATABASE` | - | 是 | 資料庫名稱 |
| `POSTGRES_MAX_CONNECTIONS` | `12` | 否 | 最大連線數 |

#### Neo4j

| 變數名稱 | 預設值 | 必填 | 說明 |
|---------|--------|------|------|
| `NEO4J_URI` | - | 是 | Neo4j 連線 URI（例如：`neo4j+s://xxx.databases.neo4j.io`） |
| `NEO4J_USERNAME` | `neo4j` | 是 | Neo4j 使用者名稱 |
| `NEO4J_PASSWORD` | - | 是 | Neo4j 密碼 |

### 其他環境變數

| 變數名稱 | 預設值 | 必填 | 說明 |
|---------|--------|------|------|
| `LOG_LEVEL` | `INFO` | 否 | 日誌等級：`DEBUG`、`INFO`、`WARNING`、`ERROR` |
| `LOG_DIR` | 當前目錄 | 否 | 日誌檔案目錄 |
| `LOG_MAX_BYTES` | `10485760` | 否 | 日誌檔案最大大小（位元組） |
| `LOG_BACKUP_COUNT` | `5` | 否 | 日誌備份檔案數量 |
| `TIKTOKEN_CACHE_DIR` | - | 否 | Tiktoken 快取目錄（離線部署用） |

---

## 3. 安裝與部署 (Installation & Deployment)

### 前置需求

- **Python**：3.10 或更高版本
- **作業系統**：Windows、macOS、Linux
- **LibreOffice**（處理 Office 文件時需要）：
  - Windows：從[官方網站](https://www.libreoffice.org/download/download/)下載安裝程式
  - macOS：`brew install --cask libreoffice`
  - Ubuntu/Debian：`sudo apt-get install libreoffice`
  - CentOS/RHEL：`sudo yum install libreoffice`

### 套件安裝

#### 方法一：從 PyPI 安裝（推薦）

```bash
# 基本安裝
pip install raganything

# 安裝所有選用功能
pip install 'raganything[all]'

# 僅安裝特定功能
pip install 'raganything[image]'      # 圖片格式轉換（BMP, TIFF, GIF, WebP）
pip install 'raganything[text]'       # 文字檔處理（TXT, MD）
pip install 'raganything[image,text]' # 多個功能
```

#### 方法二：從原始碼安裝

```bash
# 安裝 uv（如果尚未安裝）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 複製專案
git clone https://github.com/HKUDS/RAG-Anything.git
cd RAG-Anything

# 安裝套件與依賴（在虛擬環境中）
uv sync

# 安裝選用依賴
uv sync --extra image --extra text  # 特定功能
uv sync --all-extras                 # 所有選用功能

# 使用 uv 執行命令（推薦方式）
uv run python examples/raganything_example.py --help
```

#### 方法三：使用 setup.py

```bash
git clone https://github.com/HKUDS/RAG-Anything.git
cd RAG-Anything
pip install -e .
```

### 驗證安裝

```bash
# 檢查 MinerU 安裝
mineru --version

# 檢查 Python 套件
python -c "from raganything import RAGAnything; rag = RAGAnything(); print('✅ MinerU 安裝正確' if rag.check_parser_installation() else '❌ MinerU 安裝問題')"
```

### 環境變數設定

建立 `.env` 檔案（參考 `env.example`）：

```bash
# 複製範例檔案
cp env.example .env

# 編輯 .env 檔案，設定必要的環境變數
# 至少需要設定：
# - LLM_BINDING_API_KEY
# - EMBEDDING_BINDING_HOST（如果使用外部服務）
```

---

## 4. 操作指南 (Operations)

### 基本操作

#### 啟動服務

RAG-Anything 是一個 Python 函式庫，不需要單獨啟動服務。使用方式如下：

```python
import asyncio
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

async def main():
    # 設定 API 配置
    api_key = "your-api-key"
    base_url = "your-base-url"  # 可選

    # 建立 RAGAnything 配置
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    # 定義 LLM 模型函數
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

    # 定義視覺模型函數
    def vision_model_func(prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs):
        if messages:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        elif image_data:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=[
                    {"role": "system", "content": system_prompt} if system_prompt else None,
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                        ],
                    },
                ],
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        else:
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    # 定義 Embedding 函數
    embedding_func = EmbeddingFunc(
        embedding_dim=3072,
        max_token_size=8192,
        func=lambda texts: openai_embed(
            texts,
            model="text-embedding-3-large",
            api_key=api_key,
            base_url=base_url,
        ),
    )

    # 初始化 RAGAnything
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )

    # 處理文件
    await rag.process_document_complete(
        file_path="path/to/your/document.pdf",
        output_dir="./output",
        parse_method="auto"
    )

    # 查詢處理後的內容
    result = await rag.aquery(
        "文件中的主要發現是什麼？",
        mode="hybrid"
    )
    print("查詢結果:", result)

if __name__ == "__main__":
    asyncio.run(main())
```

#### 執行範例

```bash
# 端到端處理（含解析器選擇）
python examples/raganything_example.py path/to/document.pdf --api-key YOUR_API_KEY --parser mineru

# 直接多模態處理
python examples/modalprocessors_example.py --api-key YOUR_API_KEY

# Office 文件解析測試（僅測試 MinerU，無需 API Key）
python examples/office_document_test.py --file path/to/document.docx

# 圖片格式解析測試（僅測試 MinerU，無需 API Key）
python examples/image_format_test.py --file path/to/image.bmp
```

### 進階設定 (LLM Integration)

#### Azure OpenAI 整合設定

RAG-Anything 透過 LightRAG 的 `openai_complete_if_cache` 與 `openai_embed` 函數支援 Azure OpenAI。設定方式如下：

**方法一：透過環境變數**

在 `.env` 檔案中設定：

```bash
LLM_BINDING=azure_openai
LLM_BINDING_HOST=https://your-resource.openai.azure.com
LLM_BINDING_API_KEY=your-azure-api-key
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o
LLM_MODEL=gpt-4o

# Embedding 設定（如果使用 Azure OpenAI Embedding）
EMBEDDING_BINDING=azure_openai
EMBEDDING_BINDING_HOST=https://your-resource.openai.azure.com
EMBEDDING_BINDING_API_KEY=your-azure-api-key
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-large
AZURE_EMBEDDING_API_VERSION=2023-05-15
EMBEDDING_DIM=3072
```

**方法二：在程式碼中設定**

```python
import os
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

# Azure OpenAI 配置
azure_endpoint = "https://your-resource.openai.azure.com"
azure_api_key = "your-azure-api-key"
azure_api_version = "2024-08-01-preview"
azure_deployment = "gpt-4o"

# LLM 模型函數（使用 Azure OpenAI）
def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return openai_complete_if_cache(
        azure_deployment,  # 使用 Deployment Name
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=azure_api_key,
        base_url=azure_endpoint,
        api_version=azure_api_version,  # 指定 API 版本
        **kwargs,
    )

# Embedding 函數（使用 Azure OpenAI）
embedding_func = EmbeddingFunc(
    embedding_dim=3072,
    max_token_size=8192,
    func=lambda texts: openai_embed(
        texts,
        model="text-embedding-3-large",  # 或使用 AZURE_EMBEDDING_DEPLOYMENT
        api_key=azure_api_key,
        base_url=azure_endpoint,
        api_version="2023-05-15",  # Embedding API 版本
    ),
)
```

**重要注意事項**：

1. **Deployment Name**：Azure OpenAI 使用 Deployment Name 而非模型名稱，需在 Azure Portal 中建立 Deployment
2. **API Version**：不同功能可能需要不同的 API 版本
   - Chat Completions：通常使用 `2024-08-01-preview` 或更新版本
   - Embeddings：通常使用 `2023-05-15` 或更新版本
3. **端點格式**：Azure OpenAI 端點格式為 `https://<resource-name>.openai.azure.com`
4. **驗證連線**：可使用以下程式碼測試連線：

```python
# 測試 Azure OpenAI 連線
try:
    result = llm_model_func("Hello, world!")
    print("✅ Azure OpenAI 連線成功")
except Exception as e:
    print(f"❌ Azure OpenAI 連線失敗: {e}")
```

### 故障排除 (Troubleshooting)

#### Azure 連線錯誤

**錯誤 401（未授權）**：
- 檢查 `LLM_BINDING_API_KEY` 是否正確
- 確認 API Key 是否已啟用且未過期
- 驗證 API Key 是否對應到正確的 Azure 資源

**錯誤 404（找不到資源）**：
- 檢查 `LLM_BINDING_HOST` 是否正確（格式：`https://<resource-name>.openai.azure.com`）
- 確認 `AZURE_OPENAI_DEPLOYMENT` 是否存在且已部署
- 驗證 Deployment 是否在正確的資源中

**錯誤 429（請求過多）**：
- 檢查 Azure OpenAI 的 Rate Limit 設定
- 降低 `MAX_ASYNC` 環境變數值
- 增加請求間隔時間

#### Python 套件相依性問題

**MinerU 安裝失敗**：
```bash
# 檢查 Python 版本（需 >= 3.10）
python --version

# 重新安裝 MinerU
pip install --upgrade mineru[core]

# 或使用 uv
uv pip install mineru[core]
```

**依賴衝突**：
```bash
# 使用虛擬環境隔離依賴
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate  # Windows

# 重新安裝
pip install raganything
```

#### NFS 掛載失敗（Kubernetes 部署）

如果使用 Kubernetes 部署並需要 NFS 儲存：

1. **檢查 NFS 伺服器連線**：
```bash
# 測試 NFS 連線
showmount -e <nfs-server-ip>
```

2. **檢查 PersistentVolume 設定**：
```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: rag-storage-pv
spec:
  capacity:
    storage: 100Gi
  accessModes:
    - ReadWriteMany
  nfs:
    server: <nfs-server-ip>
    path: /path/to/nfs/share
  persistentVolumeReclaimPolicy: Retain
```

3. **檢查 PersistentVolumeClaim**：
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: rag-storage-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
```

#### 其他常見問題

**記憶體不足**：
- 降低 `MAX_CONCURRENT_FILES` 值
- 減少批次處理的檔案數量
- 增加系統記憶體或使用交換空間

**處理速度慢**：
- 啟用 GPU 加速（如果使用 MinerU）：設定 `device="cuda:0"`
- 增加 `MAX_ASYNC` 值（但需注意 Rate Limit）
- 使用更快的儲存後端（如 PostgreSQL 而非本地檔案）

**日誌檔案過大**：
- 調整 `LOG_MAX_BYTES` 與 `LOG_BACKUP_COUNT`
- 定期清理舊日誌檔案
- 使用日誌輪轉工具

---

## 5. 範例與截圖 (Examples)

### 基本使用範例

#### 範例 1：處理單一 PDF 文件

```python
import asyncio
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

async def process_pdf():
    api_key = "your-api-key"
    
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="mineru",
        parse_method="auto",
    )
    
    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            **kwargs,
        )
    
    embedding_func = EmbeddingFunc(
        embedding_dim=3072,
        max_token_size=8192,
        func=lambda texts: openai_embed(
            texts,
            model="text-embedding-3-large",
            api_key=api_key,
        ),
    )
    
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
    )
    
    # 處理文件
    await rag.process_document_complete(
        file_path="document.pdf",
        output_dir="./output"
    )
    
    # 查詢
    result = await rag.aquery("文件的主要內容是什麼？", mode="hybrid")
    print(result)

asyncio.run(process_pdf())
```

#### 範例 2：批次處理資料夾

```python
# 批次處理資料夾中的所有 PDF 文件
await rag.process_folder_complete(
    folder_path="./documents",
    output_dir="./output",
    file_extensions=[".pdf", ".docx"],
    recursive=True,
    max_workers=4
)
```

#### 範例 3：多模態查詢

```python
# 查詢包含表格的問題
table_result = await rag.aquery_with_multimodal(
    "比較這些效能指標與文件內容",
    multimodal_content=[{
        "type": "table",
        "table_data": """Method,Accuracy,Speed
                        RAGAnything,95.2%,120ms
                        Traditional,87.3%,180ms""",
        "table_caption": "效能比較"
    }],
    mode="hybrid"
)

# 查詢包含公式的問題
equation_result = await rag.aquery_with_multimodal(
    "解釋這個公式及其與文件內容的關聯",
    multimodal_content=[{
        "type": "equation",
        "latex": "P(d|q) = \\frac{P(q|d) \\cdot P(d)}{P(q)}",
        "equation_caption": "文件相關性機率"
    }],
    mode="hybrid"
)
```

### 截圖位置標註

> [圖片說明：此處應顯示 RAG-Anything 系統架構圖，展示文件解析、多模態處理、知識圖譜、檢索與查詢的完整流程]
> [圖片說明：此處應顯示 Azure OpenAI 設定畫面，包含 Deployment Name、API Version、端點 URL 等配置選項]
> [圖片說明：此處應顯示批次處理進度畫面，包含已處理檔案數、處理時間、錯誤統計等資訊]
> [圖片說明：此處應顯示查詢結果畫面，包含問題、答案、引用來源等資訊]
