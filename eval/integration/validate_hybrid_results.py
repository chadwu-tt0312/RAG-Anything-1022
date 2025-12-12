"""驗收 Hybrid Retrieval 評估結果

根據 results.json 和日誌文件分析：
1. Hybrid 模式在複雜查詢上的相似度分數
2. Hybrid 模式優於單一模式的查詢比例
3. 各模式的回應時間是否在可接受範圍內
4. 檢索到的上下文相關性
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_results(json_path: Path) -> Dict:
    """載入評估結果 JSON"""
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def analyze_hybrid_complex_queries(results: Dict) -> Dict:
    """分析 Hybrid 模式在複雜查詢上的相似度分數"""
    complex_query_types = ["explain", "summary"]  # 複雜查詢類型

    complex_scores = []
    complex_cases = []

    for case in results["cases"]:
        if case["query_type"] in complex_query_types:
            hybrid_score = case["by_mode"]["hybrid"]["score"]
            complex_scores.append(hybrid_score)
            complex_cases.append(
                {
                    "id": case["id"],
                    "query": case["query"][:50] + "...",
                    "type": case["query_type"],
                    "score": hybrid_score,
                }
            )

    avg_complex_score = sum(complex_scores) / len(complex_scores) if complex_scores else 0.0

    return {
        "avg_score": avg_complex_score,
        "count": len(complex_scores),
        "scores": complex_scores,
        "cases": complex_cases,
    }


def analyze_hybrid_superiority(results: Dict) -> Dict:
    """分析 Hybrid 模式優於單一模式的查詢比例"""
    modes = ["naive", "local", "global"]

    total_queries = len(results["cases"])
    hybrid_better_count = {mode: 0 for mode in modes}
    hybrid_better_cases = {mode: [] for mode in modes}

    for case in results["cases"]:
        hybrid_score = case["by_mode"]["hybrid"]["score"]

        for mode in modes:
            mode_score = case["by_mode"][mode]["score"]
            if hybrid_score > mode_score:
                hybrid_better_count[mode] += 1
                hybrid_better_cases[mode].append(
                    {
                        "id": case["id"],
                        "query": case["query"][:50] + "...",
                        f"{mode}_score": mode_score,
                        "hybrid_score": hybrid_score,
                        "improvement": hybrid_score - mode_score,
                    }
                )

    ratios = {
        mode: (count / total_queries * 100) if total_queries > 0 else 0.0
        for mode, count in hybrid_better_count.items()
    }

    # 計算 Hybrid 優於所有單一模式的查詢數量
    hybrid_better_all = 0
    hybrid_better_all_cases = []
    for case in results["cases"]:
        hybrid_score = case["by_mode"]["hybrid"]["score"]
        all_better = all(hybrid_score > case["by_mode"][mode]["score"] for mode in modes)
        if all_better:
            hybrid_better_all += 1
            hybrid_better_all_cases.append(
                {
                    "id": case["id"],
                    "query": case["query"][:50] + "...",
                    "hybrid_score": hybrid_score,
                    "other_scores": {mode: case["by_mode"][mode]["score"] for mode in modes},
                }
            )

    return {
        "total_queries": total_queries,
        "better_than_naive": {
            "count": hybrid_better_count["naive"],
            "ratio": ratios["naive"],
            "cases": hybrid_better_cases["naive"],
        },
        "better_than_local": {
            "count": hybrid_better_count["local"],
            "ratio": ratios["local"],
            "cases": hybrid_better_cases["local"],
        },
        "better_than_global": {
            "count": hybrid_better_count["global"],
            "ratio": ratios["global"],
            "cases": hybrid_better_cases["global"],
        },
        "better_than_all": {
            "count": hybrid_better_all,
            "ratio": (hybrid_better_all / total_queries * 100) if total_queries > 0 else 0.0,
            "cases": hybrid_better_all_cases,
        },
    }


def analyze_response_times(results: Dict, acceptable_max_s: float = 30.0) -> Dict:
    """分析各模式的回應時間是否在可接受範圍內"""
    modes = ["naive", "local", "global", "hybrid"]

    analysis = {}
    for mode in modes:
        times = [case["by_mode"][mode]["elapsed_s"] for case in results["cases"]]
        avg_time = results["aggregate"][mode]["avg_elapsed_s"]
        max_time = max(times)
        min_time = min(times)

        acceptable_count = sum(1 for t in times if t <= acceptable_max_s)
        acceptable_ratio = (acceptable_count / len(times) * 100) if times else 0.0

        analysis[mode] = {
            "avg_s": avg_time,
            "min_s": min_time,
            "max_s": max_time,
            "acceptable_count": acceptable_count,
            "acceptable_ratio": acceptable_ratio,
            "all_acceptable": acceptable_count == len(times),
            "times": times,
        }

    return {"acceptable_max_s": acceptable_max_s, "modes": analysis}


def analyze_context_relevance(log_path: Path) -> Dict:
    """從日誌分析檢索到的上下文相關性"""
    if not log_path.exists():
        return {"error": "日誌文件不存在"}

    with log_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    # 分析每個查詢的上下文資訊
    context_info = []
    current_query = None
    current_mode = None

    for i, line in enumerate(lines):
        # 識別查詢開始
        if "Executing text query:" in line:
            query_text = line.split("Executing text query:")[1].strip()
            current_query = query_text[:50] + "..." if len(query_text) > 50 else query_text

        # 識別模式
        if "Query mode:" in line:
            current_mode = line.split("Query mode:")[1].strip()

        # 提取上下文資訊
        if "Final context:" in line and current_mode:
            parts = line.split("Final context:")[1].strip()
            # 解析 entities, relations, chunks
            entities = 0
            relations = 0
            chunks = 0

            if "entities" in parts:
                try:
                    entities = int(parts.split("entities")[0].strip().split()[-1])
                except:
                    pass
            if "relations" in parts:
                try:
                    relations = int(parts.split("relations")[0].strip().split()[-1])
                except:
                    pass
            if "chunks" in parts:
                try:
                    chunks = int(parts.split("chunks")[0].strip().split()[-1])
                except:
                    pass

            context_info.append(
                {
                    "query": current_query,
                    "mode": current_mode,
                    "entities": entities,
                    "relations": relations,
                    "chunks": chunks,
                    "total_context_items": entities + relations + chunks,
                }
            )

    # 按模式統計
    mode_stats = {}
    for item in context_info:
        mode = item["mode"]
        if mode not in mode_stats:
            mode_stats[mode] = {
                "count": 0,
                "total_entities": 0,
                "total_relations": 0,
                "total_chunks": 0,
                "total_items": 0,
            }

        mode_stats[mode]["count"] += 1
        mode_stats[mode]["total_entities"] += item["entities"]
        mode_stats[mode]["total_relations"] += item["relations"]
        mode_stats[mode]["total_chunks"] += item["chunks"]
        mode_stats[mode]["total_items"] += item["total_context_items"]

    # 計算平均值
    for mode in mode_stats:
        count = mode_stats[mode]["count"]
        if count > 0:
            mode_stats[mode]["avg_entities"] = mode_stats[mode]["total_entities"] / count
            mode_stats[mode]["avg_relations"] = mode_stats[mode]["total_relations"] / count
            mode_stats[mode]["avg_chunks"] = mode_stats[mode]["total_chunks"] / count
            mode_stats[mode]["avg_total_items"] = mode_stats[mode]["total_items"] / count

    return {"mode_stats": mode_stats, "detailed_contexts": context_info}


def generate_report(results_path: Path, log_path: Path, output_path: Path = None):
    """生成驗收報告"""
    print("=" * 80)
    print("Hybrid Retrieval 評估結果驗收報告")
    print("=" * 80)
    print()

    # 載入結果
    results = load_results(results_path)

    # 1. Hybrid 模式在複雜查詢上的相似度分數
    print("【1. Hybrid 模式在複雜查詢上的相似度分數】")
    print("-" * 80)
    complex_analysis = analyze_hybrid_complex_queries(results)
    print(f"複雜查詢數量: {complex_analysis['count']}")
    print(f"平均相似度分數: {complex_analysis['avg_score']:.4f}")
    print(
        f"分數範圍: {min(complex_analysis['scores']):.4f} ~ {max(complex_analysis['scores']):.4f}"
    )
    print("\n各複雜查詢詳情:")
    for case in complex_analysis["cases"]:
        print(f"  - {case['id']} ({case['type']}): {case['score']:.4f}")
        print(f"    查詢: {case['query']}")
    print()

    # 2. Hybrid 模式優於單一模式的查詢比例
    print("【2. Hybrid 模式優於單一模式的查詢比例】")
    print("-" * 80)
    superiority = analyze_hybrid_superiority(results)
    print(f"總查詢數: {superiority['total_queries']}")
    print()
    print(f"優於 Naive 模式:")
    print(f"  - 數量: {superiority['better_than_naive']['count']}/{superiority['total_queries']}")
    print(f"  - 比例: {superiority['better_than_naive']['ratio']:.1f}%")
    print()
    print(f"優於 Local 模式:")
    print(f"  - 數量: {superiority['better_than_local']['count']}/{superiority['total_queries']}")
    print(f"  - 比例: {superiority['better_than_local']['ratio']:.1f}%")
    print()
    print(f"優於 Global 模式:")
    print(f"  - 數量: {superiority['better_than_global']['count']}/{superiority['total_queries']}")
    print(f"  - 比例: {superiority['better_than_global']['ratio']:.1f}%")
    print()
    print(f"優於所有單一模式:")
    print(f"  - 數量: {superiority['better_than_all']['count']}/{superiority['total_queries']}")
    print(f"  - 比例: {superiority['better_than_all']['ratio']:.1f}%")
    print()

    # 3. 各模式的回應時間
    print("【3. 各模式的回應時間分析】")
    print("-" * 80)
    time_analysis = analyze_response_times(results, acceptable_max_s=30.0)
    print(f"可接受最大回應時間: {time_analysis['acceptable_max_s']:.1f} 秒")
    print()
    for mode, stats in time_analysis["modes"].items():
        print(f"{mode.upper()} 模式:")
        print(f"  - 平均時間: {stats['avg_s']:.2f} 秒")
        print(f"  - 最小時間: {stats['min_s']:.2f} 秒")
        print(f"  - 最大時間: {stats['max_s']:.2f} 秒")
        print(f"  - 可接受查詢數: {stats['acceptable_count']}/{len(stats['times'])}")
        print(f"  - 可接受比例: {stats['acceptable_ratio']:.1f}%")
        print(f"  - 全部可接受: {'是' if stats['all_acceptable'] else '否'}")
        print()

    # 4. 檢索到的上下文相關性
    print("【4. 檢索到的上下文相關性】")
    print("-" * 80)
    context_analysis = analyze_context_relevance(log_path)
    if "error" in context_analysis:
        print(f"錯誤: {context_analysis['error']}")
    else:
        print("各模式檢索到的上下文統計:")
        for mode, stats in context_analysis["mode_stats"].items():
            print(f"\n{mode.upper()} 模式:")
            print(f"  - 查詢次數: {stats['count']}")
            print(f"  - 平均實體數: {stats.get('avg_entities', 0):.1f}")
            print(f"  - 平均關係數: {stats.get('avg_relations', 0):.1f}")
            print(f"  - 平均 Chunks 數: {stats.get('avg_chunks', 0):.1f}")
            print(f"  - 平均總上下文項目數: {stats.get('avg_total_items', 0):.1f}")
    print()

    # 總結
    print("=" * 80)
    print("【驗收總結】")
    print("=" * 80)

    # 判斷標準
    checks = []

    # 檢查 1: Hybrid 在複雜查詢上的分數
    if complex_analysis["avg_score"] >= 0.70:
        checks.append(("[PASS]", "Hybrid 模式在複雜查詢上的平均分數 >= 0.70"))
    else:
        checks.append(
            (
                "[FAIL]",
                f"Hybrid 模式在複雜查詢上的平均分數 {complex_analysis['avg_score']:.4f} < 0.70",
            )
        )

    # 檢查 2: Hybrid 優於至少一種單一模式的比例
    max_ratio = max(
        superiority["better_than_naive"]["ratio"],
        superiority["better_than_local"]["ratio"],
        superiority["better_than_global"]["ratio"],
    )
    if max_ratio >= 30.0:
        checks.append(
            ("[PASS]", f"Hybrid 模式優於至少一種單一模式的比例 >= 30% (實際: {max_ratio:.1f}%)")
        )
    else:
        checks.append(("[FAIL]", f"Hybrid 模式優於至少一種單一模式的比例 {max_ratio:.1f}% < 30%"))

    # 檢查 3: 所有模式的回應時間都在可接受範圍內
    all_acceptable = all(stats["all_acceptable"] for stats in time_analysis["modes"].values())
    if all_acceptable:
        checks.append(("[PASS]", "所有模式的回應時間都在可接受範圍內 (<=30秒)"))
    else:
        checks.append(("[FAIL]", "部分模式的回應時間超過可接受範圍"))

    # 檢查 4: Hybrid 模式檢索到足夠的上下文
    if "mode_stats" in context_analysis and "hybrid" in context_analysis["mode_stats"]:
        hybrid_avg_items = context_analysis["mode_stats"]["hybrid"].get("avg_total_items", 0)
        if hybrid_avg_items >= 5:
            checks.append(
                (
                    "[PASS]",
                    f"Hybrid 模式平均檢索到足夠的上下文項目 (>=5, 實際: {hybrid_avg_items:.1f})",
                )
            )
        else:
            checks.append(
                ("[FAIL]", f"Hybrid 模式平均檢索到的上下文項目數 {hybrid_avg_items:.1f} < 5")
            )
    else:
        checks.append(("[UNKNOWN]", "無法從日誌中分析 Hybrid 模式的上下文檢索情況"))

    for status, message in checks:
        print(f"{status} {message}")

    print()
    print("=" * 80)

    # 如果指定了輸出路徑，保存報告
    if output_path:
        report_text = f"""
Hybrid Retrieval 評估結果驗收報告
生成時間: {Path(__file__).stat().st_mtime}

1. Hybrid 模式在複雜查詢上的相似度分數
   平均分數: {complex_analysis["avg_score"]:.4f}
   查詢數量: {complex_analysis["count"]}

2. Hybrid 模式優於單一模式的查詢比例
   優於 Naive: {superiority["better_than_naive"]["ratio"]:.1f}%
   優於 Local: {superiority["better_than_local"]["ratio"]:.1f}%
   優於 Global: {superiority["better_than_global"]["ratio"]:.1f}%
   優於所有: {superiority["better_than_all"]["ratio"]:.1f}%

3. 各模式的回應時間
   Naive: {time_analysis["modes"]["naive"]["avg_s"]:.2f}s (可接受: {time_analysis["modes"]["naive"]["acceptable_ratio"]:.1f}%)
   Local: {time_analysis["modes"]["local"]["avg_s"]:.2f}s (可接受: {time_analysis["modes"]["local"]["acceptable_ratio"]:.1f}%)
   Global: {time_analysis["modes"]["global"]["avg_s"]:.2f}s (可接受: {time_analysis["modes"]["global"]["acceptable_ratio"]:.1f}%)
   Hybrid: {time_analysis["modes"]["hybrid"]["avg_s"]:.2f}s (可接受: {time_analysis["modes"]["hybrid"]["acceptable_ratio"]:.1f}%)

4. 檢索到的上下文相關性
   (詳見上方統計)
"""
        with output_path.open("w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"報告已保存至: {output_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("用法: python validate_hybrid_results.py <results.json> <log.txt> [output.txt]")
        sys.exit(1)

    results_path = Path(sys.argv[1])
    log_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    generate_report(results_path, log_path, output_path)

# uv run eval/integration/validate_hybrid_results.py `
# eval/integration/results/hybrid_retrieval_accuracy/20251212-151649/results.json `
# eval/demo-251212-txt.log `
# eval/integration/results/hybrid_retrieval_accuracy/20251212-151649/validation_report.txt
