"""冒烟测试：向运行中的 MiniLLM-Serve 发一个 chat 请求，打印响应与端到端延迟。

用法（先启动服务）:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --model Qwen/Qwen2.5-0.5B-Instruct --url http://127.0.0.1:8000
"""

import argparse
import json
import time
import urllib.request


def main() -> None:
    ap = argparse.ArgumentParser(description="MiniLLM-Serve 冒烟测试")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--prompt", default="用一句话介绍你自己")
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    body = json.dumps(
        {
            "model": args.model,
            "messages": [{"role": "user", "content": args.prompt}],
            "max_tokens": args.max_tokens,
            "temperature": 0.7,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        f"{args.url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        latency_ms = (time.perf_counter() - t0) * 1000
        data = json.loads(resp.read().decode("utf-8"))

    print(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"\n端到端延迟: {latency_ms:.1f} ms")
    print(f"finish_reason: {data['choices'][0]['finish_reason']}")
    print(f"usage: {data['usage']}")


if __name__ == "__main__":
    main()
