## Role
你是一位資深的 DevOps 工程師與技術文件撰寫專家（Technical Writer）。你正處於一個名為「<<專案名稱>>」的專案 Workspace 中，可以直接讀取所有程式碼檔案。

## Task
請分析當前目錄下的檔案，撰寫一份 Markdown 格式的技術文件，檔名為 `<<專案名稱>>-使用說明.md`。

## Critical Logic: Installation & Deployment Analysis
在撰寫「安裝與部署」章節時，請執行以下邏輯判斷：
1. **條件式 K8s Helm 建置**：
    - 檢查是否存在 `Dockerfile` 或 `docker-compose.yml`。
    - **只有在**確認此專案具備容器化服務特徵時，才額外編寫 Kubernetes Helm 部署指引。
    - **若需編寫 Helm Chart，請遵守以下規格**：
        - **Service Type**：預設使用 `NodePort`。
        - **Storage**：配置 NFS Persistent Volume (PV/PVC) 範例。
        - **Source**：若無現成 Chart，請根據 `docker-compose.yml` 的邏輯反推生成。
2. **檢查套件安裝**：若專案中包含 `setup.py`, `pyproject.toml` 或 `requirements.txt`，請說明如何使用 `pip` 或 `poetry` 進行本機安裝與執行。

## Report Structure & Content Requirements

### 1. 簡介
- 基於代碼分析，簡述專案功能與架構。
  - 專案網頁：<https://github.com/HKUDS/RAG-Anything>
  - 參考網頁：<https://deepwiki.com/HKUDS/RAG-Anything>
- 目標讀者：負責部署與維護的技術人員。

### 2. 環境變數詳解 (Environment Variables)
- 掃描 `config.py`, `.env.example` 或程式碼中的 `os.getenv`。
- 製作表格：變數名稱 | 預設值 | 必填 | 說明。
- **特別注意**：針對 LLM 整合部分，請列出 **Azure OpenAI** 相關的標準變數（如 `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `OPENAI_API_VERSION` 等）。

### 3. 安裝與部署 (Installation & Deployment)
- (根據上述邏輯，撰寫「套件安裝」或/及「Kubernetes Helm 部署」)

### 4. 操作指南 (Operations)
- **基本操作**：啟動服務、執行基本任務的指令。
- **進階設定 (LLM Integration)**：
  - 專注於 **Azure OpenAI** 的整合設定。
  - 說明如何在設定檔或環境變數中指定 Deployment Name 與 API Version。
- **故障排除 (Troubleshooting)**：
  - 針對 Azure 連線錯誤（401/404）、NFS 掛載失敗或 Python 套件相依性問題提供排查建議。

### 5. 範例與截圖 (Examples)
- 使用 `> [圖片說明：此處應顯示...]` 標註需要截圖的位置。
- 若代碼中有範例輸入/輸出，請直接引用為文字範例。

## Constraints
1. **語言**：繁體中文。
2. **格式**：結構清晰的 Markdown。

---
**請現在開始分析當前 Workspace 的檔案，並撰寫報告。**
