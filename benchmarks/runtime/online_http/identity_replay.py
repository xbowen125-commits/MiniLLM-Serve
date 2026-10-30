"""One frozen six-request correctness replay, never formal performance sampling.

Uses the existing bounded HTTP replayer without moving arrival timestamps or
changing its 25ms gate. Explicit execution and GPU consent are required.
"""
import argparse
from pathlib import Path

from .artifacts import Journal, git_snapshot, write_new
from .config import load_config
from .identity_check import audit, sha
from .runner import ServerProcess, prepare_trace, replay, drain
from .workload import build_trace, trace_sha256


def execute(config_path, output, *, allow_gpu=False, mode="continuous"):
    config = load_config(config_path)
    if config.data["evidence_kind"] == "real_gpu":
        if not allow_gpu:
            raise ValueError("--allow-gpu required for bounded correctness replay")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no fake fallback")
    output = Path(output)
    output.mkdir(exist_ok=False)
    source = git_snapshot()
    write_new(output / "source.json", source)
    write_new(output / "config.json", config.as_dict())
    trace, token_count = prepare_trace(config)
    write_new(output / "workload.json", {"pristine_sha256": trace_sha256(build_trace(config)),
        "prepared_sha256": trace_sha256(trace), "requests": trace})
    directory = output / mode
    directory.mkdir(exist_ok=False)
    journal = Journal(directory / "requests.jsonl")
    server = None
    result = {"purpose": "bounded_frozen_correctness_replay", "formal_performance_samples": 0,
        "config_sha256": config.sha256, "trace_sha256": trace_sha256(trace), "mode": mode,
        "git": {k: v for k, v in source.items() if k != "files"}, "performance_statistics": None}
    try:
        server = ServerProcess(config, mode, directory, Path(config_path).resolve(), source,
                               host_module="benchmarks.runtime.online_http.identity_server")
        server.wait_ready()
        result["server"] = server.metadata
        result["before"] = drain(server.base_url, config, mode=mode)
        if not result["before"]["idle_after_test"]:
            raise RuntimeError("service not idle before correctness replay")
        rows, _ = replay(server.base_url, config, trace, token_count, journal,
            {"config_sha256": config.sha256, "trace_sha256": trace_sha256(trace),
             "server_pid": server.metadata["pid"], "performance_sample": False},
            phase="identity_correctness", identity_trace=True)
        result["completed_records"] = len(rows)
        result["after"] = drain(server.base_url, config, mode=mode)
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        journal.close()
        if server is not None:
            result["shutdown"] = server.close()
        write_new(directory / "replay.json", result)
        write_new(directory / "identity-inputs.json", {"sha256": {
            p.name: sha(p) for p in directory.iterdir() if p.is_file() and p.name != "server.log"}})
    if (directory / "trace-index.json").exists():
        checked = audit(directory, output / "recomputed", config, source)
        result["identity_status"] = checked["status"]
    else:
        result["identity_status"] = "UNKNOWN"
    write_new(output / "result.json", result)
    write_new(output / "artifacts.json", {"sha256": {
        str(p.relative_to(output)).replace("\\", "/"): sha(p)
        for p in output.rglob("*") if p.is_file()}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("Plan only: one continuous six-request correctness replay; performance samples=0")
        return
    result = execute(args.config, args.output, allow_gpu=args.allow_gpu)
    print({k: result.get(k) for k in ("identity_status", "error", "formal_performance_samples")})


if __name__ == "__main__":
    main()
