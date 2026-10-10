"""Read-only readiness check for a running MiniLLM-Serve container."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request


def get_json(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check container health without running inference")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--wait-seconds", type=float, default=180)
    parser.add_argument("--request-timeout", type=float, default=5)
    parser.add_argument("--expect-device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.wait_seconds <= 0 or args.request_timeout <= 0:
        parser.error("timeouts must be positive")

    deadline = time.monotonic() + args.wait_seconds
    last_error: Exception | None = None
    health = None
    while time.monotonic() < deadline:
        try:
            health = get_json(f"{args.base_url.rstrip('/')}/health", args.request_timeout)
            if health.get("status") == "ok" and health.get("loaded") is True:
                break
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(1)
    else:
        raise SystemExit(f"service did not become healthy: {last_error or health}")

    device = str(health.get("device", ""))
    if not device.startswith(args.expect_device):
        raise SystemExit(f"expected {args.expect_device}, service reports {device!r}")
    scheduler = get_json(f"{args.base_url.rstrip('/')}/scheduler/status", args.request_timeout)
    if scheduler.get("shutting_down"):
        raise SystemExit("scheduler is shutting down")
    print(json.dumps({"health": health, "scheduler": scheduler}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
