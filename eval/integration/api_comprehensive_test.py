"""RAG-Anything API 完整驗證測試

目標：
- 驗證 RAG-Anything 能作為服務提供 API
- 確認與現有系統的整合介面正確
- 驗證 API 的穩定性與效能
- 確認認證與授權機制正確

測試範圍：
1. 健康檢查測試
2. 認證與授權測試（正確/錯誤/缺失的 API key）
3. 端點整合測試（insert_text, process_file, query）
4. 並發測試（多個請求同時發送）
5. 效能測試（響應時間、吞吐量）
6. 錯誤處理測試（無效請求、缺失參數等）
7. 穩定性測試（長時間運行）

Usage:
  pip install -r eval/integration/requirements-integration.txt
  python eval/integration/api_comprehensive_test.py \
    --base-url http://localhost:8000 \
    --api-key YOUR_API_KEY \
    [--test-file eval/docs/test-01.txt] \
    [--concurrent-requests 10] \
    [--duration 300]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class TestResult:
    """單一測試結果"""

    name: str
    success: bool
    duration: float
    error: Optional[str] = None
    response_data: Optional[Dict] = None


@dataclass
class TestSuite:
    """測試套件結果"""

    name: str
    results: List[TestResult] = field(default_factory=list)
    total_duration: float = 0.0

    def add_result(self, result: TestResult):
        self.results.append(result)
        self.total_duration += result.duration

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failure_count(self) -> int:
        return len(self.results) - self.success_count

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.success_count / len(self.results) * 100

    @property
    def avg_duration(self) -> float:
        if not self.results:
            return 0.0
        durations = [r.duration for r in self.results]
        return statistics.mean(durations)

    @property
    def p95_duration(self) -> float:
        if not self.results:
            return 0.0
        durations = sorted([r.duration for r in self.results])
        index = int(len(durations) * 0.95)
        return durations[index] if index < len(durations) else durations[-1]


class APITestClient:
    """API 測試客戶端，支援重試和會話管理"""

    def __init__(self, base_url: str, api_key: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()

        # 設定重試策略
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _headers(self) -> Dict[str, str]:
        return {"X-API-Key": self.api_key, "Content-Type": "application/json"}

    def health_check(self) -> TestResult:
        """健康檢查測試"""
        start = time.time()
        try:
            r = self.session.get(f"{self.base_url}/health", timeout=10)
            duration = time.time() - start
            success = r.status_code == 200
            data = r.json() if success else None
            error = None if success else f"Status {r.status_code}: {r.text}"
            return TestResult("health_check", success, duration, error, data)
        except Exception as e:
            return TestResult("health_check", False, time.time() - start, str(e))

    def get_stats(self) -> TestResult:
        """取得統計資訊測試"""
        start = time.time()
        try:
            r = self.session.get(
                f"{self.base_url}/v1/stats", headers=self._headers(), timeout=10
            )
            duration = time.time() - start
            success = r.status_code == 200
            data = r.json() if success else None
            error = None if success else f"Status {r.status_code}: {r.text}"
            return TestResult("get_stats", success, duration, error, data)
        except Exception as e:
            return TestResult("get_stats", False, time.time() - start, str(e))

    def insert_text(
        self, doc_id: str, text: str, file_path: Optional[str] = None
    ) -> TestResult:
        """插入文字測試"""
        start = time.time()
        try:
            payload = {"doc_id": doc_id, "text": text}
            if file_path:
                payload["file_path"] = file_path

            r = self.session.post(
                f"{self.base_url}/v1/insert_text",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
            duration = time.time() - start
            success = r.status_code == 200
            data = r.json() if success else None
            error = None if success else f"Status {r.status_code}: {r.text}"
            return TestResult("insert_text", success, duration, error, data)
        except Exception as e:
            return TestResult("insert_text", False, time.time() - start, str(e))

    def query(self, query: str, mode: str = "hybrid", vlm_enhanced: bool = False) -> TestResult:
        """查詢測試"""
        start = time.time()
        try:
            r = self.session.post(
                f"{self.base_url}/v1/query",
                headers=self._headers(),
                json={"query": query, "mode": mode, "vlm_enhanced": vlm_enhanced},
                timeout=self.timeout,
            )
            duration = time.time() - start
            success = r.status_code == 200
            data = r.json() if success else None
            error = None if success else f"Status {r.status_code}: {r.text}"
            return TestResult("query", success, duration, error, data)
        except Exception as e:
            return TestResult("query", False, time.time() - start, str(e))

    def process_file(self, file_path: str, parse_method: str = "auto") -> TestResult:
        """處理檔案測試"""
        start = time.time()
        try:
            r = self.session.post(
                f"{self.base_url}/v1/process_file",
                headers=self._headers(),
                json={"file_path": file_path, "parse_method": parse_method},
                timeout=self.timeout * 2,  # 檔案處理可能需要更長時間
            )
            duration = time.time() - start
            success = r.status_code == 200
            data = r.json() if success else None
            error = None if success else f"Status {r.status_code}: {r.text}"
            return TestResult("process_file", success, duration, error, data)
        except Exception as e:
            return TestResult("process_file", False, time.time() - start, str(e))

    def test_auth(self, invalid_key: str = "invalid-key") -> List[TestResult]:
        """認證測試：測試正確、錯誤、缺失的 API key"""
        results = []

        # 測試 1: 正確的 API key
        start = time.time()
        try:
            r = self.session.get(f"{self.base_url}/health", headers=self._headers(), timeout=10)
            duration = time.time() - start
            # 健康檢查端點不需要認證，所以應該成功
            results.append(
                TestResult(
                    "auth_valid_key",
                    True,
                    duration,
                    None,
                    {"status_code": r.status_code},
                )
            )
        except Exception as e:
            results.append(TestResult("auth_valid_key", False, time.time() - start, str(e)))

        # 測試 2: 錯誤的 API key（需要認證的端點）
        start = time.time()
        try:
            r = self.session.post(
                f"{self.base_url}/v1/query",
                headers={"X-API-Key": invalid_key, "Content-Type": "application/json"},
                json={"query": "test"},
                timeout=10,
            )
            duration = time.time() - start
            # 應該返回 401
            success = r.status_code == 401
            results.append(
                TestResult(
                    "auth_invalid_key",
                    success,
                    duration,
                    None if success else f"Expected 401, got {r.status_code}",
                    {"status_code": r.status_code},
                )
            )
        except Exception as e:
            results.append(TestResult("auth_invalid_key", False, time.time() - start, str(e)))

        # 測試 3: 缺失的 API key
        start = time.time()
        try:
            r = self.session.post(
                f"{self.base_url}/v1/query",
                headers={"Content-Type": "application/json"},
                json={"query": "test"},
                timeout=10,
            )
            duration = time.time() - start
            # 應該返回 401 或 422
            success = r.status_code in [401, 422]
            results.append(
                TestResult(
                    "auth_missing_key",
                    success,
                    duration,
                    None if success else f"Expected 401/422, got {r.status_code}",
                    {"status_code": r.status_code},
                )
            )
        except Exception as e:
            results.append(TestResult("auth_missing_key", False, time.time() - start, str(e)))

        return results

    def test_error_handling(self) -> List[TestResult]:
        """錯誤處理測試：無效請求、缺失參數等"""
        results = []

        # 測試 1: 缺失必要參數
        start = time.time()
        try:
            r = self.session.post(
                f"{self.base_url}/v1/insert_text",
                headers=self._headers(),
                json={},  # 缺少 text 參數
                timeout=10,
            )
            duration = time.time() - start
            # 應該返回 422 (Validation Error)
            success = r.status_code == 422
            results.append(
                TestResult(
                    "error_missing_params",
                    success,
                    duration,
                    None if success else f"Expected 422, got {r.status_code}",
                    {"status_code": r.status_code},
                )
            )
        except Exception as e:
            results.append(TestResult("error_missing_params", False, time.time() - start, str(e)))

        # 測試 2: 無效的查詢模式
        start = time.time()
        try:
            r = self.session.post(
                f"{self.base_url}/v1/query",
                headers=self._headers(),
                json={"query": "test", "mode": "invalid_mode"},
                timeout=10,
            )
            duration = time.time() - start
            # 可能返回 400 或 500，取決於實作
            success = r.status_code in [400, 422, 500]
            results.append(
                TestResult(
                    "error_invalid_mode",
                    success,
                    duration,
                    None if success else f"Expected 4xx/5xx, got {r.status_code}",
                    {"status_code": r.status_code},
                )
            )
        except Exception as e:
            results.append(TestResult("error_invalid_mode", False, time.time() - start, str(e)))

        # 測試 3: 不存在的檔案路徑
        start = time.time()
        try:
            r = self.session.post(
                f"{self.base_url}/v1/process_file",
                headers=self._headers(),
                json={"file_path": "/nonexistent/file.txt"},
                timeout=10,
            )
            duration = time.time() - start
            # 應該返回 400
            success = r.status_code == 400
            results.append(
                TestResult(
                    "error_nonexistent_file",
                    success,
                    duration,
                    None if success else f"Expected 400, got {r.status_code}",
                    {"status_code": r.status_code},
                )
            )
        except Exception as e:
            results.append(
                TestResult("error_nonexistent_file", False, time.time() - start, str(e))
            )

        return results


def run_integration_tests(client: APITestClient, test_file: Optional[str] = None) -> TestSuite:
    """執行整合測試：完整的端到端流程"""
    suite = TestSuite("integration_tests")

    print("\n[整合測試] 開始端到端流程測試...")

    # 1. 健康檢查
    result = client.health_check()
    suite.add_result(result)
    print(f"  ✓ 健康檢查: {'成功' if result.success else '失敗'} ({result.duration:.2f}s)")

    # 1.5. 統計資訊（如果可用）
    result = client.get_stats()
    suite.add_result(result)
    if result.success and result.response_data:
        stats = result.response_data
        print(
            f"  ✓ 統計資訊: 總請求 {stats.get('total_requests', 0)}, "
            f"成功率 {stats.get('success_rate', 0):.1f}%, "
            f"平均響應 {stats.get('average_duration', 0):.3f}s"
        )
    else:
        print(f"  ✓ 統計資訊: {'成功' if result.success else '失敗'} ({result.duration:.2f}s)")

    # 2. 插入文字
    test_text = (
        "Transformer 是 Vaswani 等人在 2017 年提出的架構，核心是 self-attention，"
        "能有效建模長距離依賴並支援並行計算。相對於 RNN，Transformer 的主要優點包括："
        "1) 並行計算能力，2) 長距離依賴建模，3) 訓練效率更高。"
    )
    result = client.insert_text("integration-test-doc-1", test_text, "test.txt")
    suite.add_result(result)
    print(f"  ✓ 插入文字: {'成功' if result.success else '失敗'} ({result.duration:.2f}s)")

    # 3. 查詢測試（多種模式）
    queries = [
        ("Transformer 的核心創新是什麼？", "hybrid", False),
        ("Transformer 相對於 RNN 的主要優點是什麼？", "naive", False),
        ("Transformer 的優點有哪些？", "local", False),
    ]

    for query, mode, vlm in queries:
        result = client.query(query, mode=mode, vlm_enhanced=vlm)
        suite.add_result(result)
        status = "成功" if result.success else "失敗"
        if result.success and result.response_data:
            answer_len = len(result.response_data.get("result", ""))
            print(f"  ✓ 查詢 ({mode}): {status} ({result.duration:.2f}s, 答案長度: {answer_len})")
        else:
            print(f"  ✓ 查詢 ({mode}): {status} ({result.duration:.2f}s)")

    # 4. 處理檔案（如果提供了測試檔案）
    if test_file and Path(test_file).exists():
        result = client.process_file(test_file)
        suite.add_result(result)
        print(f"  ✓ 處理檔案: {'成功' if result.success else '失敗'} ({result.duration:.2f}s)")

    return suite


def run_concurrent_tests(client: APITestClient, num_requests: int = 10) -> TestSuite:
    """並發測試：同時發送多個請求"""
    suite = TestSuite("concurrent_tests")

    print(f"\n[並發測試] 發送 {num_requests} 個並發請求...")

    def make_request(i: int) -> TestResult:
        return client.query(f"測試查詢 {i}", mode="hybrid")

    start_time = time.time()
    with requests.Session() as session:
        # 使用 ThreadPoolExecutor 模擬並發請求
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_requests)]
            for future in as_completed(futures):
                result = future.result()
                suite.add_result(result)

    total_duration = time.time() - start_time
    suite.total_duration = total_duration

    print(f"  完成時間: {total_duration:.2f}s")
    print(f"  成功率: {suite.success_rate:.1f}%")
    print(f"  平均響應時間: {suite.avg_duration:.2f}s")
    print(f"  P95 響應時間: {suite.p95_duration:.2f}s")

    return suite


async def run_performance_tests(client: APITestClient, num_iterations: int = 20) -> TestSuite:
    """效能測試：測量響應時間和吞吐量"""
    suite = TestSuite("performance_tests")

    print(f"\n[效能測試] 執行 {num_iterations} 次迭代...")

    # 先插入測試資料
    test_text = "效能測試文件內容。這是一個用於測試 API 效能的範例文字。"
    insert_result = client.insert_text("perf-test-doc", test_text)
    if not insert_result.success:
        print(f"  警告: 無法插入測試資料: {insert_result.error}")
        return suite

    # 執行多次查詢以測量效能
    query = "效能測試查詢"
    durations = []

    for i in range(num_iterations):
        result = client.query(query, mode="hybrid")
        suite.add_result(result)
        if result.success:
            durations.append(result.duration)

    if durations:
        print(f"  平均響應時間: {statistics.mean(durations):.3f}s")
        print(f"  中位數響應時間: {statistics.median(durations):.3f}s")
        print(f"  最小響應時間: {min(durations):.3f}s")
        print(f"  最大響應時間: {max(durations):.3f}s")
        print(f"  標準差: {statistics.stdev(durations):.3f}s")
        if len(durations) > 1:
            print(f"  P95 響應時間: {sorted(durations)[int(len(durations) * 0.95)]:.3f}s")

    return suite


def run_stability_test(client: APITestClient, duration_seconds: int = 300) -> TestSuite:
    """穩定性測試：長時間運行，檢查記憶體洩漏和錯誤率"""
    suite = TestSuite("stability_tests")

    print(f"\n[穩定性測試] 運行 {duration_seconds} 秒...")

    # 先插入測試資料
    test_text = "穩定性測試文件內容。這是一個用於長時間運行測試的範例文字。"
    insert_result = client.insert_text("stability-test-doc", test_text)
    if not insert_result.success:
        print(f"  警告: 無法插入測試資料: {insert_result.error}")
        return suite

    start_time = time.time()
    request_count = 0
    error_count = 0
    durations = []

    queries = [
        "穩定性測試查詢 1",
        "穩定性測試查詢 2",
        "穩定性測試查詢 3",
    ]

    while time.time() - start_time < duration_seconds:
        query = queries[request_count % len(queries)]
        result = client.query(query, mode="hybrid")
        suite.add_result(result)

        request_count += 1
        if not result.success:
            error_count += 1
        else:
            durations.append(result.duration)

        # 每 10 個請求報告一次進度
        if request_count % 10 == 0:
            elapsed = time.time() - start_time
            error_rate = (error_count / request_count) * 100 if request_count > 0 else 0
            avg_duration = statistics.mean(durations) if durations else 0
            print(
                f"  進度: {request_count} 請求, "
                f"錯誤率: {error_rate:.1f}%, "
                f"平均響應: {avg_duration:.3f}s, "
                f"已運行: {elapsed:.0f}s"
            )

        # 避免過度負載
        time.sleep(0.5)

    total_duration = time.time() - start_time
    suite.total_duration = total_duration

    print(f"\n  總請求數: {request_count}")
    print(f"  錯誤數: {error_count}")
    print(f"  錯誤率: {(error_count/request_count*100) if request_count > 0 else 0:.2f}%")
    if durations:
        print(f"  平均響應時間: {statistics.mean(durations):.3f}s")
        print(f"  最大響應時間: {max(durations):.3f}s")

    return suite


def generate_report(all_suites: List[TestSuite], output_file: Optional[str] = None):
    """生成測試報告"""
    report_lines = [
        "=" * 80,
        "RAG-Anything API 驗證測試報告",
        "=" * 80,
        "",
    ]

    total_tests = sum(len(suite.results) for suite in all_suites)
    total_success = sum(suite.success_count for suite in all_suites)
    total_duration = sum(suite.total_duration for suite in all_suites)

    report_lines.extend(
        [
            f"總測試數: {total_tests}",
            f"成功: {total_success}",
            f"失敗: {total_tests - total_success}",
            f"總成功率: {(total_success/total_tests*100) if total_tests > 0 else 0:.1f}%",
            f"總執行時間: {total_duration:.2f}s",
            "",
            "-" * 80,
            "",
        ]
    )

    for suite in all_suites:
        report_lines.extend(
            [
                f"測試套件: {suite.name}",
                f"  測試數: {len(suite.results)}",
                f"  成功: {suite.success_count}",
                f"  失敗: {suite.failure_count}",
                f"  成功率: {suite.success_rate:.1f}%",
                f"  總時間: {suite.total_duration:.2f}s",
            ]
        )

        if suite.avg_duration > 0:
            report_lines.append(f"  平均響應時間: {suite.avg_duration:.3f}s")
        if suite.p95_duration > 0:
            report_lines.append(f"  P95 響應時間: {suite.p95_duration:.3f}s")

        # 列出失敗的測試
        failures = [r for r in suite.results if not r.success]
        if failures:
            report_lines.append("  失敗的測試:")
            for failure in failures:
                report_lines.append(f"    - {failure.name}: {failure.error}")

        report_lines.append("")

    report_text = "\n".join(report_lines)

    # 輸出到控制台
    print("\n" + report_text)

    # 寫入檔案
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\n報告已儲存至: {output_file}")

    # 同時生成 JSON 格式
    if output_file:
        json_file = output_file.replace(".txt", ".json")
        json_data = {
            "summary": {
                "total_tests": total_tests,
                "total_success": total_success,
                "total_failure": total_tests - total_success,
                "success_rate": (total_success / total_tests * 100) if total_tests > 0 else 0,
                "total_duration": total_duration,
            },
            "suites": [
                {
                    "name": suite.name,
                    "test_count": len(suite.results),
                    "success_count": suite.success_count,
                    "failure_count": suite.failure_count,
                    "success_rate": suite.success_rate,
                    "total_duration": suite.total_duration,
                    "avg_duration": suite.avg_duration,
                    "p95_duration": suite.p95_duration,
                    "results": [
                        {
                            "name": r.name,
                            "success": r.success,
                            "duration": r.duration,
                            "error": r.error,
                        }
                        for r in suite.results
                    ],
                }
                for suite in all_suites
            ],
        }
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"JSON 報告已儲存至: {json_file}")


def main():
    parser = argparse.ArgumentParser(description="RAG-Anything API 完整驗證測試")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API 基礎 URL")
    parser.add_argument("--api-key", required=True, help="API 金鑰")
    parser.add_argument("--test-file", help="用於測試的檔案路徑")
    parser.add_argument("--concurrent-requests", type=int, default=10, help="並發請求數")
    parser.add_argument("--duration", type=int, default=300, help="穩定性測試持續時間（秒）")
    parser.add_argument("--output", help="測試報告輸出檔案路徑")
    parser.add_argument(
        "--skip-stability", action="store_true", help="跳過穩定性測試（長時間運行）"
    )
    args = parser.parse_args()

    client = APITestClient(args.base_url, args.api_key)

    all_suites = []

    # 1. 健康檢查測試
    print("\n[測試 1/6] 健康檢查測試")
    suite = TestSuite("health_check")
    result = client.health_check()
    suite.add_result(result)
    all_suites.append(suite)
    print(f"結果: {'✓ 通過' if result.success else '✗ 失敗'}")

    if not result.success:
        print(f"錯誤: {result.error}")
        print("無法繼續測試，請確認 API 服務正在運行。")
        return 1

    # 2. 認證與授權測試
    print("\n[測試 2/6] 認證與授權測試")
    suite = TestSuite("authentication")
    results = client.test_auth()
    for r in results:
        suite.add_result(r)
    all_suites.append(suite)
    print(f"結果: {suite.success_count}/{len(results)} 通過")

    # 3. 錯誤處理測試
    print("\n[測試 3/6] 錯誤處理測試")
    suite = TestSuite("error_handling")
    results = client.test_error_handling()
    for r in results:
        suite.add_result(r)
    all_suites.append(suite)
    print(f"結果: {suite.success_count}/{len(results)} 通過")

    # 4. 整合測試
    print("\n[測試 4/6] 整合測試")
    suite = run_integration_tests(client, args.test_file)
    all_suites.append(suite)
    print(f"結果: {suite.success_count}/{len(suite.results)} 通過")

    # 5. 並發測試
    print("\n[測試 5/6] 並發測試")
    suite = run_concurrent_tests(client, args.concurrent_requests)
    all_suites.append(suite)
    print(f"結果: {suite.success_count}/{len(suite.results)} 通過")

    # 6. 效能測試
    print("\n[測試 6/6] 效能測試")
    suite = asyncio.run(run_performance_tests(client))
    all_suites.append(suite)
    print(f"結果: {suite.success_count}/{len(suite.results)} 通過")

    # 7. 穩定性測試（可選）
    if not args.skip_stability:
        print("\n[測試 7/7] 穩定性測試")
        suite = run_stability_test(client, args.duration)
        all_suites.append(suite)
        print(f"結果: {suite.success_count}/{len(suite.results)} 通過")

    # 生成報告
    output_file = args.output or f"api_test_report_{int(time.time())}.txt"
    generate_report(all_suites, output_file)

    # 判斷整體結果
    total_tests = sum(len(suite.results) for suite in all_suites)
    total_success = sum(suite.success_count for suite in all_suites)
    success_rate = (total_success / total_tests * 100) if total_tests > 0 else 0

    print(f"\n{'='*80}")
    print(f"整體測試結果: {total_success}/{total_tests} 通過 ({success_rate:.1f}%)")
    print(f"{'='*80}")

    return 0 if success_rate >= 90 else 1


if __name__ == "__main__":
    raise SystemExit(main())

