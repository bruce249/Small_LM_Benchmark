"""Example: run evaluations via the REST API (requires the server to be running).

Usage
-----
    # Terminal 1 – start the API server:
    uvicorn arena.api.main:app --reload --port 8000

    # Terminal 2 – run this script:
    python -m examples.run_via_api
"""

from __future__ import annotations

import json
import sys
import time

import requests

API_BASE = "http://localhost:8000"


def main() -> None:
    # Step 1: List available models
    print("📋 Available models:")
    resp = requests.get(f"{API_BASE}/models")
    resp.raise_for_status()
    for m in resp.json():
        print(f"   • {m['model_id']} ({m['display_name']})")

    # Step 2: Launch an async experiment
    payload = {
        "task_type": "summarization",
        "model_ids": [
            "Qwen/Qwen2.5-7B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
        ],
        "max_samples": 5,
        "split": "test",
    }

    print("\n🚀 Launching experiment …")
    resp = requests.post(f"{API_BASE}/experiments", json=payload)
    resp.raise_for_status()
    data = resp.json()
    exp_id = data["experiment_id"]
    print(f"   Experiment ID: {exp_id}")

    # Step 3: Poll for results
    print("\n⏳ Polling for results …")
    for attempt in range(60):
        time.sleep(5)
        resp = requests.get(f"{API_BASE}/experiments/{exp_id}")
        if resp.status_code == 200:
            report = resp.json()
            if report["status"] == "completed":
                print("\n✅ Experiment completed!")
                print(json.dumps(report, indent=2))
                return
            elif report["status"] == "failed":
                print(f"\n❌ Experiment failed: {report.get('error')}")
                return
        print(f"   … still running (attempt {attempt + 1})")

    print("\n⚠️ Timed out waiting for experiment to complete")


if __name__ == "__main__":
    main()
