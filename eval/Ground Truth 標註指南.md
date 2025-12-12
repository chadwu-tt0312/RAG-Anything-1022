# 測試文件準備與 Ground Truth 標註指南

本指南說明如何準備測試文件並建立 ground truth 標註，用於驗證 RAG-Anything 的知識圖譜建立正確性。

---

## 1. 測試文件範本來源

### 1.1 學術論文（推薦）

**arXiv 論文庫**：
- **網址**：<https://arxiv.org/>
- **搜尋建議**：
  - 搜尋關鍵字：`"machine learning" "neural network"`（包含圖表與公式）
  - 搜尋關鍵字：`"knowledge graph" "entity extraction"`（相關領域）
  - 選擇包含多個圖表、表格、數學公式的論文
- **下載方式**：點選論文頁面的 PDF 下載連結
- **推薦論文類型**：
  - 技術報告（Technical Reports）
  - 系統論文（Systems Papers）- 通常包含架構圖與流程圖
  - 實驗論文（Experimental Papers）- 包含數據表格與圖表

**範例論文搜尋**：
```
site:arxiv.org "transformer" "architecture" "figure"
site:arxiv.org "BERT" "evaluation" "table"
site:arxiv.org "graph neural network" "dataset"
```

### 1.2 技術文件（推薦）

**開源專案文件**：
- **GitHub 技術文件**：搜尋大型開源專案的技術文件
  - Kubernetes 文件：<https://kubernetes.io/docs/>
  - TensorFlow 文件：<https://www.tensorflow.org/guide>
  - PyTorch 文件：<https://pytorch.org/docs/>
- **下載方式**：使用瀏覽器列印功能（PDF）或文件匯出功能

**API 文件**：
- **OpenAPI/Swagger 文件**：通常包含清晰的實體與關係定義
- **REST API 文件**：包含資源、端點、參數等明確實體

### 1.3 多模態文件範例

**包含以下內容的文件最適合測試**：
- ✅ **圖片**：流程圖、架構圖、數據圖表、示意圖
- ✅ **表格**：數據對比表、參數配置表、效能比較表
- ✅ **公式**：數學公式、化學方程式、演算法表示
- ✅ **明確實體**：技術名詞、系統組件、方法名稱
- ✅ **明確關係**：`belongs_to`、`implements`、`depends_on`、`uses` 等

### 1.4 推薦測試文件清單

| 文件類型 | 來源 | 推薦理由 |
|---------|------|---------|
| **Transformer 架構論文** | arXiv | 包含架構圖、數學公式、技術實體 |
| **Kubernetes 架構文件** | kubernetes.io | 包含系統組件、關係、流程圖 |
| **資料庫設計文件** | 技術部落格 | 包含 ER 圖、實體關係、表格 |
| **API 設計文件** | OpenAPI 規範 | 包含資源、端點、參數關係 |
| **機器學習評估報告** | 學術論文 | 包含實驗數據、表格、圖表 |

---

## 2. Ground Truth 標註方法

本 repo 的流程改為使用 **Label Studio** 進行「實體 + 關係」標註，並以匯出 JSON 做 ground truth。

### 2.1 測試文件放置規範（固定）

- **測試文件目錄**：`./eval/docs`
- **範例**：`./eval/docs/test-01.txt`

### 2.2 Label Studio 安裝與啟動

```bash
pip install label-studio
label-studio start
```

### 2.3 建立專案與標籤設定（Project config）

1. 在 Label Studio 建立新專案（Project）。
2. 進入 Project 的 **Settings → Labeling Interface**，貼上以下 XML（本 repo 也提供檔案：`eval/kg_eval/label_studio_project_config.xml`）。

```xml
<View>
  <!-- 建立標註區塊(region)之間的關係 -->
  <Relations>
    <Relation value="belongs_to" />
    <Relation value="implements" />
    <Relation value="depends_on" />
    <Relation value="uses" />
    <Relation value="proposes" />
    <Relation value="compares_with" />
    <Relation value="describes" />
    <Relation value="contains" />
  </Relations>

  <Labels name="label" toName="text">
    <!-- RAG-Anything 風格 entity types（含擴充 DATE；方案 B） -->
    <Label value="TECHNICAL_TERM" background="#FF6B6B"/>
    <Label value="SYSTEM_COMPONENT" background="#4D96FF"/>
    <Label value="PERSON" background="#6BCB77"/>
    <Label value="ORGANIZATION" background="#FFD93D"/>
    <Label value="DATE" background="#A66CFF"/>

    <!-- 多模態內容（選用） -->
    <Label value="IMAGE" background="#FF9F1C"/>
    <Label value="TABLE" background="#2EC4B6"/>
    <Label value="EQUATION" background="#8D99AE"/>
  </Labels>

  <!-- 顯示文本內容 -->
  <Text name="text" value="$text"/>
</View>
```

> **注意**：本 repo 的關係（relation_type）統一使用**小寫**（例如 `proposes`、`contains`）。

### 2.4 匯入測試文件（以 `test-01.txt` 為例）

1. 進入專案後，點 **Import**
2. 上傳 `./eval/docs/test-01.txt`
3. 開始標註：
   - **先框選 entity spans**（例如 `Transformer`、`Vaswani 等人`、`2017 年`）
   - 再用 Relation 工具把兩個 entity 連起來（例如 `Vaswani 等人` → `Transformer`，`proposes`）

### 2.5 匯出標註結果（Label Studio Export JSON）

完成標註後：專案 → **Export → JSON**，下載匯出檔。

本 repo 已有一份範例匯出檔（可直接用來跑完整流程）：
- `./eval/docs/prj1-text-02-at-2025-12-12.json`

### 2.6 轉換成 ground truth JSON（供比對）

使用本 repo 內的轉換腳本：`eval/kg_eval/label_studio_to_ground_truth.py`

**PowerShell（Windows）**：
```powershell
uv run ./eval/kg_eval/label_studio_to_ground_truth.py `
  --input ./eval/docs/prj1-text-02-at-2025-12-12.json `
  --output ./eval/kg_eval/text-02_ground_truth.json
```

**bash（Linux/macOS）**：
```bash
uv run eval/kg_eval/label_studio_to_ground_truth.py \
  --input eval/docs/prj1-text-02-at-2025-12-12.json \
  --output eval/kg_eval/text-02_ground_truth.json
```

---

## 3. 標註品質控制

### 3.1 標註一致性檢查

```python
def check_annotation_consistency(ground_truth_file):
    """檢查標註一致性"""
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        gt = json.load(f)
    
    issues = []
    
    # 檢查關係中的實體是否存在
    for relation_key, relation_data in gt["relations"].items():
        head = relation_data["head"]
        tail = relation_data["tail"]
        
        if head not in gt["entities"]:
            issues.append(f"關係 {relation_key} 的 head 實體 '{head}' 不存在")
        if tail not in gt["entities"]:
            issues.append(f"關係 {relation_key} 的 tail 實體 '{tail}' 不存在")
    
    # 檢查重複實體
    entity_names = list(gt["entities"].keys())
    if len(entity_names) != len(set(entity_names)):
        issues.append("存在重複的實體名稱")
    
    return issues

# 執行檢查
issues = check_annotation_consistency("./eval/kg_eval/_ground_truth/test-01_ground_truth.json")
if issues:
    print("發現標註問題：")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("標註一致性檢查通過")
```

### 3.2 標註完整性檢查

```python
def check_annotation_completeness(ground_truth_file, document_text):
    """檢查標註完整性"""
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        gt = json.load(f)
    
    # 檢查是否標註了所有重要實體
    # 這需要根據文件內容手動檢查
    
    print(f"已標註實體數: {len(gt['entities'])}")
    print(f"已標註關係數: {len(gt['relations'])}")
    
    # 建議：至少標註文件中的關鍵實體（出現頻率 > 3 次）
    # 建議：至少標註所有明確的技術名詞
```

### 3.3 標註檢查結果

#### 基本使用：只檢查一致性
`uv run eval/kg_eval/check_annotation.py --input eval/kg_eval/text-02_ground_truth.json`

#### 同時檢查一致性和完整性

`uv run eval/kg_eval/check_annotation.py --input eval/kg_eval/text-02_ground_truth.json --completeness`

```bash
檢查標註一致性：eval\kg_eval\text-02_ground_truth.json
------------------------------------------------------------
[OK] 標註一致性檢查通過

------------------------------------------------------------
檢查標註完整性
------------------------------------------------------------
已標註實體數：22
已標註關係數：3

實體類型分布：
  DATE: 1
  PERSON: 1
  SYSTEM_COMPONENT: 10
  TECHNICAL_TERM: 10

關係類型分布：
  contains: 1
  proposes: 2

檢查完成
```

---

## 4. 與 RAG-Anything 輸出比較

### 4.1 一鍵比對腳本（推薦）

本 repo 已提供「從文件建立 KG → 讀取 ground truth → 寬鬆比對 → 產出報告」的一鍵腳本：
- `eval/kg_eval/run_kg_correctness_eval.py`

比對策略（2B，寬鬆）：
- **實體（entity）**：先只比對 *entity name*（忽略 entity_type）
- **關係（relation）**：先只比對 *(head, tail)*（忽略 relation_type）

**PowerShell（Windows）**：
```powershell
uv run eval/kg_eval/run_kg_correctness_eval.py `
  --doc eval/docs/test-01.txt `
  --labelstudio-export eval/docs/prj1-text-02-at-2025-12-12.json `
  --working-dir eval/kg_eval/_rag_storage/test-01 `
  --out eval/kg_eval/_results/test-01
```

執行後會產生：
- `eval/kg_eval/_results/test-01/comparison_results.json`
  - `entity_metrics_loose`：實體 precision/recall/F1（寬鬆）
    - Entity metrics (loose): P=0.00%, R=0.00%, `F1=0.00%`
  - `relation_metrics_loose`：關係 precision/recall/F1（寬鬆）
    - Relation metrics (loose): P=0.00%, R=0.00%, `F1=0.00%`
  - `diff`：列出 gt_only / pred_only 方便人工檢查

### 4.2 如何判讀結果（建議流程）

1. **先看 count 是否合理**
   - `counts.gt_entities` vs `counts.pred_entities`
   - `counts.gt_relations` vs `counts.pred_relations`
2. **再看寬鬆指標是否達標**
   - 實體：`entity_metrics_loose.f1`
   - 關係：`relation_metrics_loose.f1`
3. **最後看差異清單做人工抽查**
   - `diff.gt_only_entities`：ground truth 有，但系統沒抽出（召回問題）
   - `diff.pred_only_entities`：系統抽出但 ground truth 沒有（精確率/過度抽取）
   - `diff.*_relations`：關係缺漏/多抽

> 本 repo 採用 2B 策略：先用寬鬆規則把流程跑通、找主要誤差來源；等命名規範穩定後再升級成嚴格比對（含 entity_type 與 relation_type）。

---

## 5. 快速開始範例

### 5.1 使用 repo 內既有檔案，跑通一輪（最小可行）

你不需要先做任何新增檔案，本 repo 已經包含：
- 測試文件：`eval/docs/test-01.txt`
- Label Studio 匯出範例：`eval/docs/prj1-text-02-at-2025-12-12.json`

直接執行（同 4.1）：
```powershell
uv run eval/kg_eval/run_kg_correctness_eval.py `
  --doc eval/docs/test-01.txt `
  --labelstudio-export eval/docs/prj1-text-02-at-2025-12-12.json `
  --working-dir eval/kg_eval/_rag_storage/test-01 `
  --out eval/kg_eval/_results/test-01
```

### 5.2 使用方式

1. `./eval/docs` 放入你要測的文件（txt/pdf/docx...）
2. 用 Label Studio 依 `eval/kg_eval/label_studio_project_config.xml` 的 labels/relation 規格標註並匯出 JSON
3. 執行 `eval/kg_eval/run_kg_correctness_eval.py` 產出 `comparison_results.json`

---

## 6. 標註最佳實踐

### 6.1 標註原則

1. **一致性**：相同類型的實體使用相同的標註方式
2. **完整性**：標註所有重要的實體與關係
3. **準確性**：確保標註正確，避免錯誤標註
4. **可驗證性**：標註應可被其他人驗證

### 6.2 標註優先順序

1. **高優先級**：
   - 技術名詞（方法名稱、演算法名稱）
   - 系統組件（模組、服務、API）
   - 明確的關係（belongs_to、implements、depends_on）

2. **中優先級**：
   - 人名、組織名
   - 圖片、表格的語義描述
   - 間接關係

3. **低優先級**：
   - 一般名詞
   - 模糊關係

### 6.3 標註時間估算

- **簡單文件**（< 10 頁）：1-2 小時
- **中等文件**（10-30 頁）：3-5 小時
- **複雜文件**（> 30 頁）：6-10 小時

建議從簡單文件開始，熟悉標註流程後再處理複雜文件。

---

## 7. 參考資源

- **Label Studio 文件**：<https://labelstud.io/>
- **arXiv 論文庫**：<https://arxiv.org/>
- **知識圖譜標註指南**：<https://www.w3.org/TR/annotation-model/>

---

## 8. 常見問題

### Q1: 如何處理多模態內容（圖片、表格）的標註？

**A**: 對於圖片與表格，標註其語義描述與相關實體：
- 圖片：標註圖片標題、描述、相關技術名詞
- 表格：標註表格標題、欄位名稱、數據實體
- 建立圖片/表格與文字實體的關係（如 `DESCRIBES`）

### Q2: 實體名稱不完全一致怎麼辦？

**A**: 使用模糊匹配（相似度 > 0.8）或建立別名對應表：
```python
entity_aliases = {
    "Transformer模型": ["Transformer", "Transformer Architecture", "Transformer Model"],
    "注意力機制": ["Attention Mechanism", "Self-Attention", "Attention"],
}
```

### Q3: 如何處理關係的多樣性？

**A**: 建立關係類型對應表：
```python
relation_mapping = {
    "包含": "contains",
    "屬於": "belongs_to",
    "實作": "implements",
    "實現": "implements",
    "依賴": "depends_on",
    "使用": "uses",
}
```

---

**最後更新**：2025年12月
