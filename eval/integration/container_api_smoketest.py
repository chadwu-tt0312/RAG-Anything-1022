"""Smoke test for containerized API (docker-compose).

Usage:
  pip install -r eval/integration/requirements-integration.txt
  python eval/integration/container_api_smoketest.py \
    --base-url http://localhost:8000 \
    --api-key change-me

Expected:
- /health returns ok
- /v1/insert_text returns success
- /v1/query returns a non-empty answer
"""

from __future__ import annotations

import argparse

import requests


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--api-key", required=True)
    args = p.parse_args()

    r = requests.get(f"{args.base_url}/health", timeout=30)
    r.raise_for_status()
    body = r.json()
    assert body.get("status") == "ok", body

    headers = {"X-API-Key": args.api_key}

    r = requests.post(
        f"{args.base_url}/v1/insert_text",
        headers=headers,
        json={
            "doc_id": "container-smoketest",
            "file_path": "container-smoketest.txt",
            "text": "Transformer 的核心是 self-attention，可並行計算並建模長距離依賴。",
        },
        timeout=60,
    )
    r.raise_for_status()
    assert r.json().get("status") == "success", r.json()

    r = requests.post(
        f"{args.base_url}/v1/query",
        headers=headers,
        json={"query": "Transformer 的核心創新是什麼？", "mode": "hybrid"},
        timeout=120,
    )
    r.raise_for_status()
    ans = r.json().get("result", "")
    assert isinstance(ans, str) and ans.strip(), r.json()

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
