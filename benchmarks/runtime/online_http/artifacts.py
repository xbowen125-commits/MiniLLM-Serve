"""Exclusive append-only experiment artifacts and source/environment provenance."""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[3]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")


class Journal:
    def __init__(self, path):
        self.output = Path(path).open("x", encoding="utf-8", newline="\n")
        self.lock = threading.Lock()

    def append(self, row):
        with self.lock:
            self.output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            self.output.flush()

    def close(self):
        self.output.close()


def git_snapshot():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, timeout=10)
    sha = git("rev-parse", "HEAD").decode().strip()
    dirty = bool(git("status", "--porcelain"))
    names = git("ls-files", "--cached", "--others", "--exclude-standard", "-z").decode("utf-8").split("\0")
    manifest = {}
    for name in sorted(set(names) - {""}):
        path = ROOT / name
        # Git-ignored local environments/results/secrets are never walked.
        if path.is_file():
            manifest[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    tree = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"sha": sha, "dirty": dirty, "source_tree_sha256": tree, "files": manifest}


def environment_snapshot():
    versions = {}
    for name in ("torch", "transformers", "accelerate", "uvicorn", "fastapi"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    clocks = {name: vars(time.get_clock_info(name)) for name in ("perf_counter", "monotonic")}
    return {"timestamp": utc_now(), "pid": os.getpid(), "python": platform.python_version(),
            "platform": platform.platform(), "versions": versions, "clocks": clocks}


def gpu_environment_snapshot(config):
    """Read identity for a gate; never load a model or run a throughput trial."""
    import torch
    data = config.data
    return {**environment_snapshot(), "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        **{key: data[key] for key in ("device", "dtype", "model_revision", "tokenizer_revision")}}


def hardware_snapshot():
    fields = "name,driver_version,memory.total,memory.used,temperature.gpu,power.draw,clocks.sm,utilization.gpu"
    try:
        result = subprocess.run(["nvidia-smi", f"--query-gpu={fields}",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        return {"timestamp": utc_now(), "fields": fields.split(","),
                "rows": result.stdout.strip().splitlines() if result.returncode == 0 else None,
                "returncode": result.returncode}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"timestamp": utc_now(), "rows": None, "unavailable": type(exc).__name__}
