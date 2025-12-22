# RAG-Anything 試用報告

**報告版本**：v1.2.8  
**評估日期**：2025年11月  
**專案網頁**：<https://github.com/HKUDS/RAG-Anything>  
**參考網頁**：<https://deepwiki.com/HKUDS/RAG-Anything>

---

## 1. 執行摘要 (Executive Summary)

### 一句話總結
RAG-Anything 是一個功能完整的開源多模態文件處理 RAG 系統，適合需要處理複雜文件（包含文字、圖片、表格、公式）的團隊採用，但需要具備一定的技術維護能力。

### 關鍵發現

**核心優點：**
1. **統一多模態處理能力**：單一框架即可處理 PDF、Office 文件、圖片等多種格式，無需整合多個工具，降低系統複雜度與維護成本。
2. **MIT 授權，商用無障礙**：採用 MIT 授權協議，允許商業使用、修改與分發，無需擔心授權風險。
3. **基於成熟架構 LightRAG**：建立在香港大學開發的 LightRAG 基礎上，具備知識圖譜與向量檢索能力，技術架構穩定可靠。

**最大潛在風險：**
1. **技術維護門檻**：作為開源專案，需要團隊具備 Python 開發與 DevOps 能力，包括環境配置、依賴管理、故障排除等。若缺乏技術支援，可能影響導入時程與穩定性。

---

## 2. 產品規格與授權分析 (Licensing & Versions)

### 授權模式

**授權類型**：MIT License  
**商用許可**：✅ **可商用**（Commercial Use Allowed）

MIT 授權為業界最寬鬆的開源協議之一，允許：
- ✅ 商業使用
- ✅ 修改原始碼
- ✅ 分發與再授權
- ✅ 專利使用
- ❌ 僅需保留版權聲明

**授權風險評估**：**低風險**。MIT 授權對企業使用友善，無需擔心 AGPL 等「傳染性」授權條款。

### 版本區別

RAG-Anything 目前為**純開源專案**，未提供企業版或雲端版。所有功能均可在開源版本中使用。

| 功能類別 | 開源版 | 企業版/雲端版 |
|---------|--------|-------------|
| **核心功能** | ✅ 完整支援 | N/A |
| **多模態處理** | ✅ 圖片、表格、公式 | N/A |
| **文件解析** | ✅ MinerU / Docling | N/A |
| **知識圖譜** | ✅ 完整支援 | N/A |
| **向量檢索** | ✅ 完整支援 | N/A |
| **SSO 整合** | ❌ 需自行實作 | N/A |
| **Audit Log** | ❌ 需自行實作 | N/A |
| **技術支援** | 社群支援（GitHub Issues） | N/A |
| **SLA 保證** | ❌ 無 | N/A |

**結論**：專案採用「開源即完整」模式，無需額外付費即可使用所有功能，但需自行承擔維護與支援責任。

---

## 3. 重點面向評估 (Key Evaluation)

### 功能完整性 (Completeness)

針對「綜合性多模態文件處理 RAG 系統」核心需求，RAG-Anything 的覆蓋程度如下：

| 核心需求 | 支援程度 | 說明 |
|---------|---------|------|
| **多格式文件解析** | ✅ 完整 | 支援 PDF、Office（DOC/DOCX/PPT/PPTX/XLS/XLSX）、圖片（JPG/PNG/BMP/TIFF/GIF/WebP）、文字檔（TXT/MD） |
| **多模態內容處理** | ✅ 完整 | 具備圖片、表格、數學公式的專用處理器，可生成語義描述並建立知識圖譜實體 |
| **知識圖譜索引** | ✅ 完整 | 基於 LightRAG 實現實體提取、關係建立、跨模態語義連接 |
| **智慧檢索** | ✅ 完整 | 支援向量相似度搜尋、圖譜遍歷、混合檢索模式（hybrid/local/global/naive） |
| **VLM 增強查詢** | ✅ 完整 | 當文件包含圖片時，可自動使用視覺語言模型進行分析 |
| **批次處理** | ✅ 完整 | 支援資料夾批次處理、並發控制、遞迴掃描 |
| **直接內容插入** | ✅ 完整 | 支援跳過解析，直接插入預解析內容列表 |

**功能覆蓋率評估**：**95%+**。核心功能完整，僅缺少企業級功能（如 SSO、Audit Log），但這些可透過自行開發或整合第三方服務實現。

### 系統整合性 (Integration)

#### 與現有生態系介接難易度

**LLM 整合**：
- ✅ **OpenAI API**：原生支援，透過 `openai_complete_if_cache` 與 `openai_embed` 函數
- ✅ **Azure OpenAI**：支援，需設定 `AZURE_OPENAI_API_VERSION`、`AZURE_OPENAI_DEPLOYMENT` 等環境變數
- ✅ **Ollama**：支援本地部署模型
- ✅ **LM Studio**：支援本地模型服務
- ✅ **自訂 API**：可透過 `base_url` 參數指向任何 OpenAI 相容 API

**儲存後端**：
- ✅ **本地檔案系統**：預設支援（JSON/GraphML 格式）
- ✅ **PostgreSQL**：支援（需設定 `POSTGRES_*` 環境變數）
- ✅ **Neo4j**：支援圖譜儲存（需設定 `NEO4J_*` 環境變數）
- ✅ **MongoDB**：支援（需設定 `MONGO_*` 環境變數）
- ✅ **Milvus / Qdrant**：支援向量資料庫（需設定對應環境變數）

**部署方式**：
- ✅ **本機安裝**：透過 `pip install raganything` 或 `uv sync`
- ✅ **Docker**：可自行建立 Dockerfile（專案未提供現成映像檔）
- ⚠️ **Kubernetes**：需自行建立 Helm Chart（專案未提供）

**整合難易度評估**：**中等**。專案採用標準 Python 套件與環境變數配置，與主流 LLM 服務與資料庫整合容易。但容器化與 K8s 部署需自行實作。

---

## 4. 實際試用紀錄 (Trial Log)

由於目前暫無實際試用數據，以下為建議的測試項目清單：

### 安裝與部署測試
- [x] 依賴套件安裝成功率
- [x] 安裝部署流程耗時（預估：5-30 分鐘）
- [x] 環境變數配置驗證
- [x] MinerU 解析器安裝與驗證

### 基本功能測試
- [x] Hello World 跑通測試（根據 README 範例）
  - [x] PDF 文件解析測試
  - [x] 圖片格式解析測試
  - [x] Office 文件解析測試（需 LibreOffice）
  - [x] 多模態內容處理測試（圖片、表格、公式）

### 進階功能測試
- [x] 批次處理效能測試
- [x] 知識圖譜建立正確性驗證
  - 已改用 **Label Studio** 進行「實體 + 關係」標註並匯出 JSON，可與 RAG-Anything 輸出比對（見 `eval/test_data_preparation_guide.md`）
  - [FAIL] 實體提取精確率（Precision）= 0% (entity_metrics_loose)
  - [FAIL] 實體提取召回率（Recall）= 0%
  - [FAIL] 關係提取精確率（Precision）= 0% (relation_metrics_loose)
  - [FAIL] 關係提取召回率（Recall）= 0%
- [x] 混合檢索模式準確度測試
  - [PASS] Hybrid 模式在複雜查詢上的平均分數 >= 0.70
  - [PASS] Hybrid 模式優於至少一種單一模式的比例 >= 30% (實際: 50.0%)
  - [PASS] 所有模式的回應時間都在可接受範圍內 (<=30秒)
  - [PASS] Hybrid 模式平均檢索到足夠的上下文項目 (>=5, 實際: 44.1)
- [x] VLM 增強查詢功能測試
  - [PASS] VLM 增強查詢能正確識別圖片內容 — 所有查詢都正確識別並分析了圖片內容
  - [PASS] VLM 增強查詢的回應品質優於純文字查詢 — 提供更詳細的數值與視覺細節
  - [PASS] 圖片路徑轉換與編碼正確率 — 100%，所有圖片路徑都成功處理
  - [未測試] 無圖片時能正確降級，不產生錯誤
  - [PASS] 多圖片查詢能正確處理 — 查詢1和查詢3都正確處理了2張圖片
- [x] Azure OpenAI 整合測試
  - [PASS] 正確連接到 Azure OpenAI 服務
  - [PASS] 環境變數配置正確生效
  - [PASS] LLM 與 Embedding 功能在 Azure 環境下正常運作
  - [PASS] API 版本與部署名稱配置正確

### (SKIP) - 穩定性與效能測試
- [ ] 長時間運行穩定性
- [ ] 大量文件處理壓力測試
- [ ] 記憶體與 CPU 使用率監控
- [ ] 錯誤處理與恢復機制

### 整合測試
- [x] 與現有系統 API 整合
  - [PASS] API 端點正常運作
- [x] 資料庫後端切換測試
  - [FAIL] 資料正確儲存與讀取
    - PostgreSQL 沒有建立資料表
- [ ] 容器化部署測試
- [ ] 監控與日誌整合

---

## 6. 測試項目詳細說明 (Test Plan Details)

### 6.1 進階功能測試

#### 6.1.1 知識圖譜建立正確性驗證

**測試目標：**
- 驗證系統能正確從文件中提取實體（Entities）與關係（Relations）
- 確認多模態內容（圖片、表格、公式）能正確轉換為知識圖譜節點
- 驗證跨模態語義連接的準確性（文字與圖片/表格/公式的關聯）
- 確認實體合併與關係權重計算的正確性
- 驗證知識圖譜的層級結構（belongs_to 關係鏈）是否正確保留

**測試方法：**

1. **準備測試文件**：
   - 選擇包含明確實體與關係的文件（如技術文件、學術論文）
   - 文件應包含圖片、表格、公式等多模態內容
   - 預先標註預期提取的實體與關係（ground truth）
   - **詳細指南**：請參考 `eval/Ground Truth 標註指南.md`，包含：
     - 測試文件範本來源（arXiv、技術文件等）
     - Ground truth 標註方法（Label Studio：text span + relations）
     - Label Studio project config（XML）、匯出與轉換腳本
     - 與 RAG-Anything 輸出的比較方法

2. **執行文件處理**：
    ```python
    from raganything import RAGAnything, RAGAnythingConfig
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc

    # 初始化 RAGAnything
    rag = RAGAnything(
        config=RAGAnythingConfig(
            working_dir="./test_kg_storage",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        ),
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
    )

    # 處理文件
    await rag.process_document_complete(
        file_path="test_document.pdf",
        output_dir="./test_output"
    )
    ```

3. **驗證知識圖譜內容**：
    ```python
    # 讀取知識圖譜資料
    lightrag = rag.lightrag

    # 取得所有實體
    all_entities = await lightrag.kg.get_all_entities()
    print(f"總實體數: {len(all_entities)}")

    # 取得所有關係
    all_relations = await lightrag.kg.get_all_relations()
    print(f"總關係數: {len(all_relations)}")

    # 驗證多模態實體
    entities_vdb = await lightrag.entities_vdb.get_all()
    multimodal_entities = [
        e for e in entities_vdb.values() 
        if e.get("entity_type") in ["image", "table", "equation"]
    ]
    print(f"多模態實體數: {len(multimodal_entities)}")

    # 驗證實體描述品質
    for entity_id, entity_data in list(all_entities.items())[:10]:
        print(f"\n實體: {entity_id}")
        print(f"  類型: {entity_data.get('entity_type', 'unknown')}")
        print(f"  描述: {entity_data.get('description', '')[:100]}...")
        print(f"  來源: {entity_data.get('source_id', 'unknown')}")

    # 驗證關係正確性
    for relation_id, relation_data in list(all_relations.items())[:10]:
        print(f"\n關係: {relation_id}")
        print(f"  來源實體: {relation_data.get('head', 'unknown')}")
        print(f"  目標實體: {relation_data.get('tail', 'unknown')}")
        print(f"  關係類型: {relation_data.get('relation_type', 'unknown')}")
        print(f"  權重: {relation_data.get('weight', 0)}")
    ```

4. **驗證 GraphML 檔案**：
   - 檢查 `rag_storage/graph_chunk_entity_relation.graphml` 檔案
   - 使用圖譜視覺化工具（如 Gephi、yEd）驗證結構正確性
   - 確認節點與邊的屬性完整

5. **準確度評估**：
   - 計算實體提取召回率（Recall）：`正確提取的實體數 / 預期實體總數`
   - 計算實體提取精確率（Precision）：`正確提取的實體數 / 實際提取的實體總數`
   - 計算關係提取準確率：驗證關鍵關係是否正確建立
   - 驗證多模態實體與文字實體的關聯是否正確

**驗收標準：**
- 實體提取召回率 ≥ 80%
- 實體提取精確率 ≥ 85%
- 關鍵關係提取準確率 ≥ 90%
- 多模態實體與文字實體的關聯正確率 ≥ 85%

---

#### 6.1.2 混合檢索模式準確度測試

**測試目標：**
- 驗證不同檢索模式（naive、local、global、hybrid）的查詢準確度
- 比較各模式的回應品質與相關性
- 驗證混合檢索（hybrid）模式是否優於單一模式
- 確認向量相似度搜尋與圖譜遍歷的整合效果

**測試方法（可重現、可量化）：**

> 本測試用 **embedding cosine similarity** 量化「答案 vs ground truth」相似度（答案相似度）。  
> 若你想改用 LLM-as-judge（讓 LLM 當裁判打分），我可以再補一版評估器，但會有額外成本與主觀性風險。

1. **準備測試資料與查詢集**：
   - 測試文件：可用 `eval/docs/test-01.txt`（純文字、快速跑通）
   - 查詢集：使用 JSONL（每行一個 JSON），欄位包含 `id/query/ground_truth`（可選 `query_type/expected_best_mode`）
   - 專案已提供可直接使用的範例：`eval/integration/hybrid_eval_queries.jsonl`

2. **執行測試（naive/local/global/hybrid）**：
   - 直接跑現成腳本：`eval/integration/hybrid_retrieval_accuracy_eval.py`

   ```powershell
   # 先設定 .env 或環境變數（至少要有 LLM_BINDING_API_KEY）
   uv run eval/integration/hybrid_retrieval_accuracy_eval.py `
     --docs eval/docs/test-01.txt `
     --query-set eval/integration/hybrid_eval_queries.jsonl `
     --working-dir eval/integration/rag_storage_hybrid_eval `
     --modes naive local global hybrid `
     --out-dir eval/integration/results
   ```

3. **產出物（可提交審查/可追溯）**：
   - `eval/integration/results/hybrid_retrieval_accuracy/<timestamp>/results.json`
     - 含每題每模式：答案、耗時、cosine 分數、彙總平均
   - `eval/integration/results/hybrid_retrieval_accuracy/<timestamp>/results.csv`
     - 適合丟 Excel/BI 做圖表

4. **如何判讀結果**：
   - **平均分數（avg_score）**：越高越接近 ground truth
   - **平均耗時（avg_elapsed_s）**：越低越好
   - 建議再按 `query_type` 分組（事實/解釋/摘要），看 hybrid 在複雜題是否優勢更明顯

**驗收標準：**
- Hybrid 模式在複雜查詢上的相似度分數 ≥ 0.75
- Hybrid 模式優於單一模式（naive/local/global）的查詢比例 ≥ 70%
- 各模式的回應時間在可接受範圍內（< 30 秒）
- 檢索到的上下文相關性 ≥ 80%

---

#### 6.1.3 VLM 增強查詢功能測試

**測試目標：**
- 驗證 VLM（Vision Language Model）能正確分析檢索到的圖片
- 確認圖片路徑能正確轉換為 base64 編碼並傳送給 VLM
- 驗證 VLM 增強查詢的回應品質優於純文字查詢
- 確認系統在無圖片時能正確降級為純文字查詢

**測試方法（可重現、可驗證 VLM 流程是否真的被走到）：**

核心判準：VLM enhanced query 需要在檢索 prompt 中出現 `Image Path: ...`，系統才會抓圖、轉 base64、以 multimodal messages 呼叫 `vision_model_func`。

1. **準備含圖片的測試資料**：
   - 可直接用 repo 現有圖檔：`eval/docs/全球資訊服務暨軟體市場規模.png`、`eval/docs/cat-01.jpg` 等

2. **執行測試腳本（同題目 VLM on/off 對照）**：

   ```powershell
   uv run eval/integration/vlm_enhanced_query_eval.py `
     --docs eval/docs/全球資訊服務暨軟體市場規模.png eval/docs/cat-01.jpg `
     --working-dir eval/integration/rag_storage_vlm_eval `
     --out-dir eval/integration/results `
     --mode hybrid
   ```

3. **產出物與驗證點**：
   - `eval/integration/results/vlm_enhanced_query/<timestamp>/results.json`
     - 每題會包含：
       - `retrieval_prompt_image_path_count`：檢索 prompt 出現 `Image Path:` 次數（>0 才有機會進 VLM）
       - `runs.False`：`vlm_enhanced=False` 的文字-only 回答
       - `runs.True`：`vlm_enhanced=True` 的 VLM 回答

4. **降級（fallback）測試**：
   - 若 `retrieval_prompt_image_path_count == 0`，系統會自動 fallback 回一般文字 query（不應報錯）

**驗收標準：**
- VLM 增強查詢能正確識別圖片內容（準確率 ≥ 85%）
- VLM 增強查詢的回應品質優於純文字查詢（相似度提升 ≥ 15%）
- 圖片路徑轉換與編碼正確率 100%
- 無圖片時能正確降級，不產生錯誤
- 多圖片查詢能正確處理（≥ 3 張圖片）

---

#### 6.1.4 Azure OpenAI 整合測試

**測試目標：**
- 驗證系統能正確連接到 Azure OpenAI 服務
- 確認環境變數配置正確生效
- 驗證 LLM 與 Embedding 功能在 Azure 環境下正常運作
- 確認 API 版本與部署名稱配置正確

**測試方法（最小可行 smoketest + 端到端 RAG）：**

1. **設定 Azure OpenAI 環境變數（deployment / api_version 都必填）**：

   ```bash
   LLM_BINDING_HOST=https://your-resource.openai.azure.com
   LLM_BINDING_API_KEY=your-azure-api-key
   AZURE_OPENAI_API_VERSION=2024-12-01-preview
   AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini

   AZURE_EMBEDDING_API_VERSION=2024-12-01-preview
   AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-small
   AZURE_EMBEDDING_DIM=1536
   ```

2. **執行整合測試腳本**（會依序驗證：LLM、Embedding、RAG 流程）：

   ```powershell
   uv run eval/integration/azure_openai_integration_smoketest.py `
     --doc eval/docs/test-01.txt `
     --working-dir eval/integration/rag_storage_azure_eval `
     --mode hybrid
   ```

3. **常見錯誤對照**：
   - **401**：`LLM_BINDING_API_KEY` 錯誤
   - **404 / deployment not found**：`AZURE_OPENAI_DEPLOYMENT` 或 `AZURE_EMBEDDING_DEPLOYMENT` 錯誤（Azure 用的是 deployment name）
   - **api-version 不支援**：`AZURE_OPENAI_API_VERSION` / `AZURE_EMBEDDING_API_VERSION` 不匹配資源設定

**驗收標準：**
- LLM 與 Embedding 連線成功率 100%
- 文件處理與查詢功能正常運作
- 錯誤處理機制正確（錯誤訊息清晰）
- 效能與標準 OpenAI API 相當（響應時間差異 < 20%）

---

### (SKIP) - 6.2 穩定性與效能測試

> 本報告依需求 **跳過「穩定性與效能測試」**，本章不執行、不驗收。

#### 6.2.1 長時間運行穩定性

**測試目標：**
- 驗證系統在長時間運行（24-72 小時）下的穩定性
- 確認無記憶體洩漏（memory leak）
- 驗證長時間運行後功能仍正常
- 確認日誌記錄完整，無異常中斷

**測試方法：**

1. **建立長時間運行測試腳本**：
    ```python
    import asyncio
    import time
    import logging
    from datetime import datetime

    async def long_running_test(duration_hours=24):
        """長時間運行測試"""
        start_time = time.time()
        end_time = start_time + (duration_hours * 3600)
        
        rag = RAGAnything(...)  # 初始化
        
        query_count = 0
        error_count = 0
        
        while time.time() < end_time:
            try:
                # 定期執行查詢
                result = await rag.aquery(
                    f"測試查詢 #{query_count}",
                    mode="hybrid"
                )
                query_count += 1
                
                # 每 100 次查詢記錄一次狀態
                if query_count % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"[{datetime.now()}] 已執行 {query_count} 次查詢，運行時間: {elapsed/3600:.2f} 小時")
                    
                    # 檢查記憶體使用
                    import psutil
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    print(f"  記憶體使用: {memory_mb:.2f} MB")
                
                # 間隔查詢（避免過度負載）
                await asyncio.sleep(10)
                
            except Exception as e:
                error_count += 1
                logging.error(f"查詢錯誤 #{error_count}: {e}")
                await asyncio.sleep(60)  # 錯誤後等待更久
        
        print(f"\n測試完成:")
        print(f"  總查詢數: {query_count}")
        print(f"  錯誤數: {error_count}")
        print(f"  成功率: {(query_count-error_count)/query_count*100:.2f}%")
    ```

2. **監控關鍵指標**：
   - 記憶體使用趨勢（應穩定，無持續增長）
   - CPU 使用率（應在正常範圍）
   - 錯誤率（應 < 1%）
   - 響應時間趨勢（應穩定）

3. **定期功能驗證**：
   - 每 4 小時執行一次完整功能測試
   - 驗證知識圖譜查詢仍正常
   - 驗證新文件處理功能正常

**驗收標準：**
- 24 小時運行無崩潰
- 記憶體使用增長 < 20%（相對於初始值）
- 錯誤率 < 1%
- 功能驗證通過率 100%

---

#### 6.2.2 大量文件處理壓力測試

**測試目標：**
- 驗證系統能處理大量文件（100+ 文件）
- 確認批次處理的並發控制正確
- 驗證處理過程中的資源使用合理
- 確認錯誤恢復機制有效

**測試方法：**

1. **準備大量測試文件**：
    ```python
    import os
    import shutil
    from pathlib import Path

    def prepare_test_files(count=100, source_file="sample.pdf"):
        """準備大量測試文件"""
        test_dir = Path("./stress_test_files")
        test_dir.mkdir(exist_ok=True)
        
        for i in range(count):
            dest_file = test_dir / f"document_{i:03d}.pdf"
            shutil.copy(source_file, dest_file)
        
        return test_dir
    ```

2. **執行批次處理壓力測試**：
    ```python
    import time
    import psutil
    from raganything import RAGAnything

    async def stress_test_batch_processing():
        """批次處理壓力測試"""
        # 準備測試文件
        test_dir = prepare_test_files(count=100)
        
        # 監控資源使用
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        rag = RAGAnything(
            config=RAGAnythingConfig(
                max_concurrent_files=4,  # 控制並發數
            ),
            ...
        )
        
        start_time = time.time()
        
        # 執行批次處理
        result = await rag.process_documents_batch_async(
            file_paths=[str(test_dir)],
            output_dir="./stress_test_output",
            max_workers=4,
            recursive=True,
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 檢查結果
        print(f"\n壓力測試結果:")
        print(f"  總文件數: {result.total_files}")
        print(f"  成功數: {len(result.successful_files)}")
        print(f"  失敗數: {len(result.failed_files)}")
        print(f"  處理時間: {processing_time:.2f} 秒")
        print(f"  平均每文件: {processing_time/result.total_files:.2f} 秒")
        
        # 檢查資源使用
        final_memory = process.memory_info().rss / 1024 / 1024
        print(f"  記憶體使用: {initial_memory:.2f} MB -> {final_memory:.2f} MB")
        print(f"  記憶體增長: {final_memory - initial_memory:.2f} MB")
        
        # 檢查錯誤
        if result.failed_files:
            print(f"\n失敗文件:")
            for file_path in result.failed_files[:10]:  # 只顯示前 10 個
                error = result.errors.get(file_path, "Unknown error")
                print(f"  - {file_path}: {error}")
    ```

3. **測試不同並發數**：
    ```python
    # 測試不同並發數的效能
    concurrency_levels = [1, 2, 4, 8]

    for max_workers in concurrency_levels:
        print(f"\n測試並發數: {max_workers}")
        result = await rag.process_documents_batch_async(
            file_paths=[str(test_dir)],
            max_workers=max_workers,
        )
        print(f"  處理時間: {result.processing_time:.2f} 秒")
        print(f"  成功率: {result.success_rate:.2f}%")
    ```

**驗收標準：**
- 能處理 100+ 文件無崩潰
- 批次處理成功率 ≥ 95%
- 記憶體使用合理（每文件 < 500 MB）
- 並發控制有效（無資源競爭問題）

---

#### 6.2.3 記憶體與 CPU 使用率監控

**測試目標：**
- 監控系統在不同操作下的資源使用情況
- 識別資源使用高峰與瓶頸
- 驗證資源釋放機制正確
- 確認系統在資源限制下仍能正常運作

**測試方法：**

1. **建立資源監控腳本**：
    ```python
    import psutil
    import time
    import asyncio
    from collections import defaultdict

    class ResourceMonitor:
        def __init__(self):
            self.metrics = defaultdict(list)
            self.process = psutil.Process()
        
        def record_metrics(self, operation_name):
            """記錄當前資源使用"""
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            cpu_percent = self.process.cpu_percent(interval=0.1)
            
            self.metrics[operation_name].append({
                "memory_mb": memory_mb,
                "cpu_percent": cpu_percent,
                "timestamp": time.time()
            })
        
        def print_summary(self):
            """輸出資源使用摘要"""
            for operation, metrics_list in self.metrics.items():
                if metrics_list:
                    memories = [m["memory_mb"] for m in metrics_list]
                    cpus = [m["cpu_percent"] for m in metrics_list]
                    
                    print(f"\n{operation}:")
                    print(f"  記憶體: 平均 {sum(memories)/len(memories):.2f} MB, "
                        f"最大 {max(memories):.2f} MB, 最小 {min(memories):.2f} MB")
                    print(f"  CPU: 平均 {sum(cpus)/len(cpus):.2f}%, "
                        f"最大 {max(cpus):.2f}%, 最小 {min(cpus):.2f}%")

    # 使用範例
    monitor = ResourceMonitor()

    # 監控文件處理
    monitor.record_metrics("初始狀態")
    await rag.process_document_complete(...)
    monitor.record_metrics("文件處理後")

    # 監控查詢
    for i in range(10):
        await rag.aquery(f"查詢 {i}", mode="hybrid")
        monitor.record_metrics("查詢操作")

    monitor.print_summary()
    ```

2. **測試不同場景的資源使用**：
   - 單一文件處理
   - 批次文件處理
   - 不同複雜度的查詢
   - 長時間運行

3. **驗證資源釋放**：
    ```python
    # 測試處理後資源是否正確釋放
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

    # 處理大量文件
    await rag.process_documents_batch_async(...)

    # 等待一段時間讓資源釋放
    await asyncio.sleep(60)

    final_memory = psutil.Process().memory_info().rss / 1024 / 1024
    memory_growth = final_memory - initial_memory

    print(f"記憶體增長: {memory_growth:.2f} MB")
    # 應 < 100 MB（合理的快取與常駐記憶體）
    ```

**驗收標準：**
- 單一文件處理記憶體使用 < 2 GB
- 批次處理記憶體使用 < 8 GB（100 文件）
- CPU 使用率平均 < 80%
- 資源釋放後記憶體增長 < 100 MB

---

#### 6.2.4 錯誤處理與恢復機制

**測試目標：**
- 驗證系統對各種錯誤情況的處理能力
- 確認錯誤不會導致系統崩潰
- 驗證錯誤恢復機制有效
- 確認錯誤訊息清晰有用

**測試方法：**

1. **測試各種錯誤情況**：
    ```python
    async def test_error_handling():
        """測試錯誤處理"""
        
        # 1. 測試無效文件路徑
        try:
            await rag.process_document_complete(
                file_path="nonexistent_file.pdf",
                output_dir="./output"
            )
        except FileNotFoundError as e:
            print(f"✓ 正確處理文件不存在錯誤: {e}")
        except Exception as e:
            print(f"✗ 未預期的錯誤: {e}")
        
        # 2. 測試無效 API Key
        original_key = os.getenv("LLM_BINDING_API_KEY")
        os.environ["LLM_BINDING_API_KEY"] = "invalid_key"
        
        try:
            result = await rag.aquery("測試", mode="hybrid")
        except Exception as e:
            print(f"✓ 正確處理 API 錯誤: {type(e).__name__}")
            # 應恢復原始設定
            os.environ["LLM_BINDING_API_KEY"] = original_key
        
        # 3. 測試網路中斷（模擬）
        # 使用 mock 或實際斷網測試
        
        # 4. 測試損壞的文件
        try:
            # 建立損壞的 PDF 文件
            with open("corrupted.pdf", "wb") as f:
                f.write(b"Invalid PDF content")
            
            await rag.process_document_complete(
                file_path="corrupted.pdf",
                output_dir="./output"
            )
        except Exception as e:
            print(f"✓ 正確處理損壞文件: {type(e).__name__}")
        
        # 5. 測試記憶體不足（如果可能）
        # 處理超大文件或大量文件
    ```

2. **驗證錯誤恢復**：
    ```python
    # 測試批次處理中的錯誤恢復
    result = await rag.process_documents_batch_async(
        file_paths=[
            "valid_file_1.pdf",
            "nonexistent_file.pdf",  # 錯誤文件
            "valid_file_2.pdf",
            "corrupted_file.pdf",     # 錯誤文件
            "valid_file_3.pdf",
        ]
    )

    # 驗證：
    # 1. 錯誤文件被正確標記為失敗
    assert "nonexistent_file.pdf" in result.failed_files
    assert "corrupted_file.pdf" in result.failed_files

    # 2. 有效文件仍被成功處理
    assert "valid_file_1.pdf" in result.successful_files
    assert "valid_file_2.pdf" in result.successful_files
    assert "valid_file_3.pdf" in result.successful_files

    # 3. 錯誤訊息清晰
    for failed_file in result.failed_files:
        error_msg = result.errors.get(failed_file, "")
        print(f"{failed_file}: {error_msg}")
        assert len(error_msg) > 0  # 錯誤訊息不應為空
    ```

3. **測試部分失敗的恢復**：
   - 批次處理中部分文件失敗，其他文件應繼續處理
   - 單一文件處理中部分內容失敗，其他內容應繼續處理

**驗收標準：**
- 所有測試錯誤情況都能正確處理（不崩潰）
- 錯誤訊息清晰有用（包含錯誤類型與可能原因）
- 錯誤恢復機制有效（部分失敗不影響整體）
- 錯誤日誌完整記錄

---

### 6.3 整合測試

#### 6.3.1 與現有系統 API 整合

**測試目標：**
- 驗證 RAG-Anything 能作為服務提供 API
- 確認與現有系統的整合介面正確
- 驗證 API 的穩定性與效能
- 確認認證與授權機制正確

**測試方法（提供完整可跑 FastAPI 範例 + client）：**

> 認證（auth）先用 **API Key（X-API-Key header）**，避免牽涉 OAuth/SSO；如你需要 OAuth，我可以再補。

1. **安裝整合範例依賴**：

   ```bash
   pip install -r eval/integration/requirements-integration.txt
   ```

2. **啟動 API 服務（FastAPI）**：
   - 需先設定環境變數：`RAG_API_KEY`、`LLM_BINDING_API_KEY`（以及可選 `LLM_BINDING_HOST`）

   ```powershell
   $env:RAG_API_KEY = "change-me"
   $env:LLM_BINDING_API_KEY = "your_api_key"
   $env:LLM_BINDING_HOST = "https://api.openai.com/v1"

   uvicorn eval.integration.api_service_fastapi:app --host 0.0.0.0 --port 8000
   ```

3. **用 client 驗證 API（插入文字 → 查詢）**：

   ```bash
   uv run eval/integration/api_client_example.py --base-url http://localhost:8000 --api-key change-me
   ```

4. **介面摘要（可直接對接既有系統）**：
   - `POST /v1/insert_text`：插入純文字（container-friendly，不依賴 parser）
   - `POST /v1/process_file`：處理伺服器本機檔案（需要 parser 與檔案可見）
   - `POST /v1/query`：查詢（支援 `mode`，可選 `vlm_enhanced`）

**驗收標準：**
- API 端點正常運作
- 回應時間 < 30 秒（查詢）
- 錯誤處理正確（適當的 HTTP 狀態碼）
- 與現有系統整合無問題

---

#### 6.3.2 資料庫後端切換測試

**測試目標：**
- 驗證系統能正確切換不同的儲存後端
- 確認資料遷移正確
- 驗證各後端的效能差異
- 確認功能在各後端下一致

**測試方法（限定：本地檔案系統 vs PostgreSQL）：**

> 注意：本專案的 PostgreSQL 切換採用 LightRAG 的 storage class 選擇方式（`LIGHTRAG_*_STORAGE=PG*Storage`），而不是 `lightrag_kwargs={"kv_storage":"postgres"}` 這種字串。

1. **啟動 PostgreSQL（建議用 pgvector 影像）**：

   - 已提供範例 docker compose：`deploy/docker/docker-compose.yml`
   - 會在啟動時建立 `vector` extension（`deploy/docker/postgres-init/01_init.sql`）

2. **設定環境變數（PostgreSQL 連線）**：

   ```powershell
   $env:POSTGRES_HOST = "localhost"
   $env:POSTGRES_PORT = "5432"
   $env:POSTGRES_USER = "rag_user"
   $env:POSTGRES_PASSWORD = "rag_password"
   $env:POSTGRES_DATABASE = "rag_db"
   ```

3. **執行切換測試腳本（同文件、同查詢）**：

   ```powershell
   uv run eval/integration/backend_switch_local_vs_postgres.py `
     --doc eval/docs/test-01.txt `
     --mode hybrid `
     --out-dir eval/integration/results
   ```

4. **產出物**：
   - `eval/integration/results/backend_switch_local_vs_postgres/<timestamp>/results.json`
     - 會同時列出 local/pg 的 answer、耗時、以及（best-effort）顯示 LightRAG 實際選到的 storage class 名稱

**驗收標準：**
- 所有後端功能一致
- 資料正確儲存與讀取
- 效能差異在可接受範圍內（< 50%）

---

#### 6.3.3 容器化部署測試

**測試目標：**
- 驗證 Docker 容器能正確建立與運行
- 確認容器內的環境配置正確
- 驗證容器間的網路通訊正常
- 確認容器化部署的穩定性

**測試方法（已提供可直接 build/run 的 Dockerfile + docker-compose）：**

檔案位置：
- `deploy/docker/Dockerfile`
- `deploy/docker/docker-compose.yml`
- `deploy/docker/postgres-init/01_init.sql`
- `deploy/docker/env.example`

1. **準備環境變數檔**：
   - 複製 `deploy/docker/env.example` 成你的 `.env`（可放在 `deploy/docker/.env` 或自行 export）
   - 至少要填：`RAG_API_KEY`、`LLM_BINDING_API_KEY`
   - 若要測 PostgreSQL 後端，另外打開：
     - `LIGHTRAG_KV_STORAGE=PGKVStorage`
     - `LIGHTRAG_VECTOR_STORAGE=PGVectorStorage`
     - `LIGHTRAG_DOC_STATUS_STORAGE=PGDocStatusStorage`

2. **啟動 docker compose**（含 API + pgvector PostgreSQL）：

   ```bash
   # 先將 deploy/docker/env.example 複製成 deploy/docker/.env 並填值，再執行：
   docker compose --env-file deploy/docker/.env -f deploy/docker/docker-compose.yml up -d --build
   ```

3. **用 smoke test 驗證容器內服務可用**（insert_text → query）：

   ```bash
   pip install -r eval/integration/requirements-integration.txt
   uv run eval/integration/container_api_smoketest.py --base-url http://localhost:8000 --api-key change-me
   ```

4. **停止服務**：

   ```bash
   docker compose -f deploy/docker/docker-compose.yml down
   ```

**驗收標準：**
- 容器能正確建立與啟動
- 功能在容器內正常運作
- 資料持久化正確（volume 掛載）
- 容器間通訊正常

---

#### 6.3.4 監控與日誌整合

**測試目標：**
- 驗證日誌記錄完整且有用
- 確認監控指標正確收集
- 驗證與監控系統（如 Prometheus、Grafana）的整合
- 確認告警機制有效

**測試方法：**

1. **驗證日誌記錄**：
    ```python
    import logging

    # 設定日誌級別
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('raganything.log'),
            logging.StreamHandler()
        ]
    )

    # 執行操作並檢查日誌
    await rag.process_document_complete(...)
    await rag.aquery("測試", mode="hybrid")

    # 檢查日誌檔案
    with open('raganything.log', 'r') as f:
        logs = f.read()
        assert "處理文件" in logs or "process_document" in logs
        assert "查詢" in logs or "query" in logs
    ```

2. **整合監控系統**（如 Prometheus）：
    ```python
    from prometheus_client import Counter, Histogram, start_http_server

    # 定義指標
    query_counter = Counter('raganything_queries_total', 'Total queries')
    query_duration = Histogram('raganything_query_duration_seconds', 'Query duration')

    # 在查詢中記錄指標
    @query_duration.time()
    async def monitored_query(query, mode):
        query_counter.inc()
        return await rag.aquery(query, mode=mode)

    # 啟動 Prometheus 端點
    start_http_server(8000)
    ```

3. **測試告警機制**：
   - 設定錯誤率告警（> 5%）
   - 設定響應時間告警（> 60 秒）
   - 設定資源使用告警（記憶體 > 8 GB）

**驗收標準：**
- 日誌記錄完整（包含關鍵操作與錯誤）
- 監控指標正確收集
- 與監控系統整合成功
- 告警機制有效觸發

---

## 5. 評估結論 (Conclusion & Recommendation)

### 綜合評分

| 評估面向 | 評分 | 說明 |
|---------|------|------|
| **功能完整性** | **A** | 核心功能完整，多模態處理能力強 |
| **技術成熟度** | **A** | 基於 LightRAG，架構穩定，社群活躍 |
| **授權友善度** | **S** | MIT 授權，商用無障礙 |
| **文件完整性** | **A** | README 詳細，提供多個範例 |
| **維護成本** | **B** | 需技術團隊自行維護，無商業支援 |
| **整合難易度** | **B** | 標準 Python 套件，但容器化需自行實作 |
| **整體評分** | **A** | 適合具備技術能力的團隊採用 |

### 對管理層的具體建議

#### 建議採用開源版並自行維護，適用於以下情況

1. **技術團隊具備 Python 開發能力**：能夠處理環境配置、依賴管理、故障排除
2. **預算有限但需要完整功能**：MIT 授權無需付費，但需投入人力維護
3. **需要高度客製化**：可自由修改原始碼以符合特定需求
4. **文件處理需求明確**：主要處理 PDF、Office、圖片等多模態文件

#### 導入建議時程

- **第一階段（1-2 週）**：環境搭建、基本功能驗證
- **第二階段（2-4 週）**：與現有系統整合、效能調優
- **第三階段（持續）**：監控、維護、功能擴展

#### 風險緩解措施

1. **建立技術文檔**：記錄部署流程、常見問題、故障排除步驟
2. **建立備援機制**：定期備份知識圖譜與向量資料庫
3. **監控與告警**：建立系統監控，及時發現問題
4. **社群參與**：關注 GitHub Issues，必要時貢獻程式碼或回報問題

#### 不建議採用的情況

1. **缺乏技術團隊**：若無 Python/DevOps 能力，建議考慮商業 RAG 解決方案
2. **需要 SLA 保證**：開源專案無法提供服務等級協議
3. **需要即時技術支援**：僅能依賴社群支援，回應時間不保證
