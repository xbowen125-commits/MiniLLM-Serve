"""Versioned shape obligations and independently replayable, fail-closed proof.

Exact matching includes every initialized KV layer. Empty prefill cache tensors
are an observed empty list, never a guessed decode layout. A successful forward
is evidence of execution, not a synchronized CUDA timing sample.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .artifacts import write_new
from .config import canonical_json
from .workload import trace_sha256

VERSION = 1
FIELDS = ("mode", "phase", "batch_size", "input_shape", "attention_mask_shape",
          "padding_width", "kv_length_before", "kv_tensor_shapes_before")
IDENTITY = ("config_sha256", "model_revision", "dtype", "device", "git")


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def shape_key(row):
    return canonical_json({key: row.get(key) for key in FIELDS})


def observable(row):
    """Missing cache layers/dimensions are unknown, not wildcard matches."""
    def dims(value, rank):
        return isinstance(value, list) and len(value) == rank and all(type(v) is int and v > 0 for v in value)
    cache = row.get("kv_length_before")
    tensors = row.get("kv_tensor_shapes_before")
    return (all(row.get(k) is not None for k in FIELDS) and
        dims(row["input_shape"], 2) and dims(row["attention_mask_shape"], 2) and
        row["input_shape"][0] == row["attention_mask_shape"][0] == row["batch_size"] and
        row["input_shape"][1] == row["padding_width"] and type(cache) is int and cache >= 0 and
        isinstance(tensors, list) and (cache == 0 or bool(tensors)) and all(
            isinstance(layer, dict) and dims(layer.get("keys"), 4) and dims(layer.get("values"), 4)
            for layer in tensors))


def build_manifest(config, trace, git, geometry):
    """Conservative singleton envelope; never assume a particular arrival merge.

    Decode t=1..max_tokens-1 uses a cache of prompt+t-1 before the
    forward. EOS/stop can shorten this envelope; the declaration does not infer
    a future stopping point from historical text or silently omit its suffix.
    Continuous re-padding/merge histories are an explicit unresolved obligation.
    """
    data = config.data
    identity = {"config_sha256": config.sha256, "git": git,
                **{k: data[k] for k in ("model_revision", "dtype", "device")}}
    shapes = {}
    unknown = []
    layers, heads, width = (geometry.get(k) for k in ("layers", "kv_heads", "head_dim"))
    valid_geometry = all(type(v) is int and v > 0 for v in (layers, heads, width))
    for item in trace:
        prompt = item.get("observed_prompt_tokens")
        if type(prompt) is not int or prompt <= 1:
            unknown.append({"id": item["request_id"], "required": True,
                            "reason": "prompt length unavailable or phase inference ambiguous"})
            continue
        for mode in ("legacy", "continuous"):
            for t in range(item["max_tokens"]):
                cache = 0 if t == 0 else prompt + t - 1
                shape = {"mode": mode, "phase": "prefill" if t == 0 else "decode",
                    "batch_size": 1, "input_shape": [1, prompt if t == 0 else 1],
                    "attention_mask_shape": [1, prompt + t],
                    "padding_width": prompt if t == 0 else 1, "kv_length_before": cache,
                    "kv_tensor_shapes_before": [] if t == 0 else (
                        [{"keys": [1, heads, cache, width], "values": [1, heads, cache, width]}
                         for _ in range(layers)] if valid_geometry else None)}
                key = shape_key(shape)
                if key not in shapes:
                    shapes[key] = {**shape, "required": True, "shape_id": digest(shape),
                        "source": "configuration_upper_bound", "rule": "singleton-cache-v1",
                        "evidence": []}
                shapes[key]["evidence"].append({"request_id": item["request_id"], "step": t})
    if not valid_geometry:
        unknown.append({"id": "model_geometry", "required": True,
                        "reason": "pinned full-attention cache geometry unavailable"})
    if data["admission_limit"] > 1:
        unknown.append({"id": "continuous_interleavings", "required": True,
            "reason": "arrival/forward timing, left-padding and regrouping KV widths are not exhaustively enumerated",
            "batch_sizes": list(range(1, data["admission_limit"] + 1)),
            "phases": ["prefill", "decode"], "source": "configuration_constraint"})
    return {"schema_version": VERSION, "kind": "expected_execution_shapes",
        **identity, "geometry": geometry, "trace": trace, "trace_sha256": trace_sha256(trace),
        "matching_rule": "exact-fields-and-all-initialized-kv-layers-v1",
        "extra_shape_policy": "FAIL_UNDECLARED_SHAPE", "shapes": list(shapes.values()),
        "unknown_obligations": unknown,
        "scope": "conservative required envelope; not an assertion every max-token suffix executes"}


def model_geometry(config):
    """Read pinned local config only; no model weights or network access."""
    if config.data["evidence_kind"] != "real_gpu":
        return {}
    from .server import pinned_assets
    path, _, _ = pinned_assets(config.data)
    value = json.loads((path / "config.json").read_text(encoding="utf-8"))
    return {"layers": value.get("num_hidden_layers"), "kv_heads": value.get("num_key_value_heads"),
        "head_dim": value.get("head_dim") or value["hidden_size"] // value["num_attention_heads"],
        "source": "pinned-model-config.json", "config_file_sha256": hashlib.sha256((path / "config.json").read_bytes()).hexdigest()}


def verify_bundle(root, *, expected_identity=None):
    """Recompute from immutable relative raw files; never trust saved counts."""
    root = Path(root).resolve()
    checks = []
    def check(code, status, evidence):
        checks.append({"code": code, "status": status, "evidence": evidence})
    try:
        index = json.loads((root / "shape-proof-index.json").read_text(encoding="utf-8"))
        for name, expected in index["sha256"].items():
            path = (root / name).resolve()
            if (not isinstance(name, str) or "\\" in name or Path(name).is_absolute()
                    or ".." in Path(name).parts or not path.is_relative_to(root) or not path.is_file()):
                raise ValueError("missing/unsafe evidence: " + str(name))
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("SHA256 mismatch: " + name)
        manifest = json.loads((root / "expected-shapes.json").read_text(encoding="utf-8"))
        if "expected-shapes.json" not in index["sha256"] or manifest.get("schema_version") != VERSION:
            raise ValueError("missing/current-version manifest required")
        identity = {k: manifest.get(k) for k in IDENTITY}
        if expected_identity is not None and identity != expected_identity:
            raise ValueError("current identity mismatch")
        if ("config.json" not in index["sha256"] or any(v is None for v in identity.values())
                or not manifest.get("shapes")):
            raise ValueError("incomplete manifest")
        # Derive declarations again, rather than trusting a truncated list.
        from .config import ProtocolConfig
        config = ProtocolConfig.from_dict(json.loads((root / "config.json").read_text(encoding="utf-8")))
        rebuilt = build_manifest(config, manifest["trace"], manifest["git"], manifest["geometry"])
        if rebuilt != manifest or index.get("manifest_sha256") != digest(manifest):
            raise ValueError("manifest derivation/hash mismatch")
        geometry = manifest["geometry"]
        if geometry.get("source") == "pinned-model-config.json":
            if "model-config.json" not in index["sha256"]:
                raise ValueError("missing raw model geometry evidence")
            model_raw = (root / "model-config.json").read_bytes()
            model_config = json.loads(model_raw)
            if (hashlib.sha256(model_raw).hexdigest() != geometry["config_file_sha256"] or
                    model_config.get("num_hidden_layers") != geometry["layers"] or
                    model_config.get("num_key_value_heads") != geometry["kv_heads"] or
                    (model_config.get("head_dim") or model_config["hidden_size"] // model_config["num_attention_heads"]) != geometry["head_dim"]):
                raise ValueError("raw model geometry mismatch")
        check("SHAPE_MANIFEST_INTEGRITY", "PASS", "expected-shapes.json")
        required = {shape_key(s) for s in manifest["shapes"] if s["required"]}
        observed, problems, unavailable = set(), [], []
        observation_ids = set()
        if not index.get("processes") or len({p["mode"] for p in index["processes"]}) != len(index["processes"]):
            raise ValueError("missing/duplicate process identities")
        for process in index["processes"]:
            name = process["evidence"]
            ready_name = process["ready"]
            active_name, finished_name = process["warmup_start"], process["warmup_end"]
            if any(n not in index["sha256"] for n in (name, ready_name, active_name, finished_name)):
                raise ValueError("unhashed raw observations")
            ready = json.loads((root / ready_name).read_text(encoding="utf-8"))
            if (process["run_id"] != root.name or ready.get("pid") != process["pid"] or
                    ready.get("mode") != process["mode"] or
                    any(ready.get(k) != v for k, v in identity.items())):
                raise ValueError("server-ready identity conflict")
            start = json.loads((root / active_name).read_text(encoding="utf-8"))
            end = json.loads((root / finished_name).read_text(encoding="utf-8"))
            for marker in (start, end):
                if any(marker.get(k) != ready.get(k) for k in ("pid", "mode", "config_sha256", "model_revision")):
                    raise ValueError("warmup marker identity conflict")
            if end.get("status") != "PASS" or end["started_at_s"] != start["monotonic_s"]:
                raise ValueError("warmup completion unconfirmed")
            for line in (root / name).read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row.get("kind") != "forward_completed" or row.get("execution_stage") != "warmup" or row.get("success") is not True:
                    continue
                if (row.get("schema_version") != VERSION or not row.get("observation_id") or
                        row["observation_id"] in observation_ids or row.get("run_id") != process["run_id"] or
                        row.get("pid") != process["pid"] or row.get("mode") != process["mode"] or
                        row.get("expected_manifest_sha256") != digest(manifest) or
                        any(row.get(k) != v for k, v in identity.items()) or
                        row.get("collector") != "forward-boundary-v1"):
                    problems.append(row.get("observation_id", "missing observation identity"))
                    continue
                observation_ids.add(row["observation_id"])
                stamp = row.get("monotonic_s")
                if (not isinstance(stamp, (float, int)) or not math.isfinite(stamp) or
                        not start["monotonic_s"] <= stamp <= end["finished_at_s"]):
                    problems.append(row["observation_id"] + ":outside-warmup-window")
                    continue
                if not observable(row):
                    unavailable.append(row["observation_id"])
                    continue
                observed.add(shape_key(row))
        missing, extras = sorted(required - observed), sorted(observed - required)
        check("SHAPE_OBSERVATION_IDENTITY", "FAIL" if problems else "PASS", problems)
        check("SHAPE_REQUIRED_DIMENSIONS", "UNKNOWN" if unavailable else "PASS", unavailable)
        check("SHAPE_REQUIRED_COVERAGE", "FAIL" if missing else "PASS", [json.loads(k) for k in missing])
        check("SHAPE_UNDECLARED_OBSERVATIONS", "FAIL" if extras else "PASS", [json.loads(k) for k in extras])
        unknown = manifest.get("unknown_obligations", [])
        check("SHAPE_EXPECTATION_COMPLETE", "UNKNOWN" if unknown else "PASS", unknown)
        counts = {"expected": len(required), "observed": len(observed), "covered": len(required & observed),
                  "missing": len(missing), "extra": len(extras), "unknown": len(unknown) + len(unavailable)}
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
        check("SHAPE_ARTIFACT_INTEGRITY", "FAIL", str(exc))
        counts = None
    status = "FAIL" if any(c["status"] == "FAIL" for c in checks) else (
        "UNKNOWN" if any(c["status"] == "UNKNOWN" for c in checks) else "PASS")
    return {"schema_version": VERSION, "status": status, "counts": counts, "checks": checks,
        "reason_codes": [c["code"] for c in checks if c["status"] != "PASS"]}


def save_proof(root, manifest, modes):
    """Index existing raw journals; all writes are exclusive, never overwrite."""
    root = Path(root)
    files = ["expected-shapes.json", "config.json"]
    if manifest["geometry"].get("source") == "pinned-model-config.json":
        files.append("model-config.json")
    processes = []
    for mode in modes:
        name = mode["mode"] + "/observer.jsonl"
        ready = mode["mode"] + "/server-ready.json"
        active, finished = (mode["mode"] + "/" + n for n in ("warmup-active", "warmup-finished"))
        files.extend([name, ready, active, finished])
        processes.append({"mode": mode["mode"], "pid": mode["metadata"]["pid"],
            "run_id": root.name, "evidence": name, "ready": ready,
            "warmup_start": active, "warmup_end": finished})
    write_new(root / "shape-proof-index.json", {"manifest_sha256": digest(manifest), "processes": processes,
        # Missing files are deliberately absent: the strict verifier reports
        # them, without replacing unrelated successful cleanup checks.
        "sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                   for name in files if (root / name).is_file()}})
    result = verify_bundle(root, expected_identity={k: manifest[k] for k in IDENTITY})
    write_new(root / "shape-coverage.json", result)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Read-only shape proof replay; no HTTP/model execution")
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = verify_bundle(args.directory)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
