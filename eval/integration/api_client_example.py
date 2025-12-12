"""Client example for api_service_fastapi.

Usage:
  pip install -r eval/integration/requirements-integration.txt
  python eval/integration/api_client_example.py --base-url http://localhost:8000 --api-key YOUR_KEY

This script:
- inserts a small text document
- queries with hybrid mode
"""

from __future__ import annotations

import argparse
import json

import requests


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--api-key", required=True)
    args = p.parse_args()

    headers = {"X-API-Key": args.api_key}

    r = requests.get(f"{args.base_url}/health", timeout=30)
    r.raise_for_status()
    print("/health:", r.json())

    text = (
        "Transformer 是 Vaswani 等人在 2017 年提出的架構，核心是 self-attention，"
        "能有效建模長距離依賴並支援並行計算。"
    )

    r = requests.post(
        f"{args.base_url}/v1/insert_text",
        headers=headers,
        json={"doc_id": "demo-doc-1", "file_path": "demo.txt", "text": text},
        timeout=60,
    )
    r.raise_for_status()
    print("/v1/insert_text:", r.json())

    r = requests.post(
        f"{args.base_url}/v1/query",
        headers=headers,
        json={"query": "Transformer 的核心創新是什麼？", "mode": "hybrid", "vlm_enhanced": False},
        timeout=120,
    )
    r.raise_for_status()
    print("/v1/query:")
    print(json.dumps(r.json(), ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
