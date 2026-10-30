"""Read-only v1 evidence classification; exclusive v2 research artifacts.

No HTTP/model execution and no changes to admission decisions. CLI returns one
until the complete execution space and process-local coverage can be proved.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .artifacts import ROOT, git_snapshot, write_new
from .config import ProtocolConfig
from .gate import verify_preflight_artifacts
from .shapes import digest, shape_key, verify_bundle
from .taxonomy import VERSION, SOURCE_FILES, build_taxonomy, check_shape, cohort_witnesses, enumerate_model

EVIDENCE_SCHEMA = {"schema_version": VERSION, "kind": "extra_shape_classification",
    "shape": "exact v1 FIELDS, unchanged integer tensor dimensions",
    "category": ["KNOWN_RULE_OMISSION_CONDITIONAL", "DYNAMIC_RULE_CONSISTENT_UNPROVEN", "UNEXPLAINED_OR_INVALID"],
    "raw_references": {"path": "relative str", "line": "1-based int", "observation_id": "str",
        "call_id": "str", "pid": "int", "run_id": "str", "monotonic_s": "float", "collector": "str"},
    "actual_request_membership": "nullable fields; UNKNOWN unless directly observed",
    "conditional_witness": "derived from prompt/length constraints without reading observations; not actual membership",
    "full_execution_legality": "UNKNOWN until request/slot path is proven"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classify_extras(root, taxonomy, replay):
    """Retain all raw observation references; never guess missing membership."""
    extras = next(c["evidence"] for c in replay["checks"] if c["code"] == "SHAPE_UNDECLARED_OBSERVATIONS")
    by_key = {shape_key(row): row for row in extras}
    witnesses = {shape_key(w["shape"]): w for w in cohort_witnesses(taxonomy)}
    references = {key: [] for key in by_key}
    for mode in ("legacy", "continuous"):
        path = root / mode / "observer.jsonl"
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            row = json.loads(line)
            if row.get("kind") != "forward_completed" or row.get("success") is not True or row.get("execution_stage") != "warmup":
                continue
            key = shape_key(row)
            if key not in references:
                continue
            references[key].append({"path": path.relative_to(root).as_posix(), "line": line_number,
                "observation_id": row["observation_id"], "call_id": row["call_id"],
                "pid": row["pid"], "run_id": row["run_id"], "monotonic_s": row["monotonic_s"],
                "execution_stage": row["execution_stage"], "collector": row["collector"]})
    records = []
    for index, key in enumerate(sorted(by_key), 1):
        shape = by_key[key]
        rule = check_shape(taxonomy, shape)
        witness = witnesses.get(key)
        category = ("KNOWN_RULE_OMISSION_CONDITIONAL" if rule["status"] == "PASS" and witness else
                    "DYNAMIC_RULE_CONSISTENT_UNPROVEN" if rule["status"] == "PASS" else "UNEXPLAINED_OR_INVALID")
        records.append({"schema_version": VERSION, "audit_id": f"extra-{index:02}",
            "shape_sha256": digest(shape), "shape": shape, "category": category,
            "local_rule_check": rule, "conditional_witness": witness,
            "raw_references": references[key], "observation_count": len(references[key]),
            "duplicate_observations": max(0, len(references[key]) - 1),
            "actual_request_membership": {"status": "UNKNOWN", "request_ids": None,
                "per_request_prompt_lengths": None, "per_request_cache_lengths": None,
                "iteration_id": None, "slot_ids": None,
                "reason": "v1 forward collector has no worker-input membership, per-row lengths, iteration or slot mapping"},
            "basis": ("v1 declares singleton shapes only; exact v2 P2/D2 equations plus source-derived conditional cohort witness"
                      if witness and rule["status"] == "PASS" else rule["reason"]),
            "normalization_change": False, "full_execution_legality": "UNKNOWN"})
    return records


def markdown(result, records):
    lines = ["# Continuous shape taxonomy and closure audit", "",
        "## Scope", "", "Existing GPU correctness artifacts only; no new GPU or performance run.",
        "Local tensor consistency is not proof of actual request membership or full schedule reachability.", "",
        "## Version and identity", "", f"Taxonomy version: {VERSION}; canonical SHA-256: `{result['taxonomy_sha256']}`.",
        f"Rule SHA-256: `{result['rule_sha256']}`.", "", "## Per-shape audit", "",
        "Every ID below links to all raw line/observation/call IDs in extra-shape-audit-v2.json.", "",
        "| ID | Phase | B | Input | Mask | KV before | Raw observations | Conclusion |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | --- |"]
    for record in records:
        s = record["shape"]
        lines.append(f"| {record['audit_id']} | {s['phase']} | {s['batch_size']} | {s['input_shape']} | {s['attention_mask_shape']} | {s['kv_length_before']} | {record['observation_count']} | {record['category']} |")
    lines += ["", "## Finite bounds and enumeration", "",
        "Pmax + sum(G_i-1) bounds stored cache width in a drained, fixed-size epoch: joining takes max; a ready forward adds one; first-token sampling and release add zero.",
        "This proves a finite over-approximation, not that every width is reachable under frozen arrivals.",
        f"Exploration: {result['enumeration_summary']}", "",
        "## Version difference", "", "V1 is retained without edits. V2 retains every v1 required shape and separately adds conditional rule derivations, never an admission whitelist.",
        "No field normalization was relaxed. Duplicate records retain all references.", "",
        "## Admission and limitations", "", f"Closure: **{result['closure']}**. Admission: **FAIL**.",
        "Historical v1 shape and admission gates remain unchanged. Current source identity differs from the old preflight after adding this audit; old GPU evidence is not a current-source admission certificate.",
        "Missing actual forward membership, per-row cache lengths and iteration IDs prevent historical path verification. Ordered-arrival abstraction has no wall-clock/admission equivalence proof; within-phase partial failures are not exhaustively modeled.",
        "A conditional cohort witness does not assert that those requests actually appeared together. No speedup, throughput or latency conclusion is drawn."]
    return "\n".join(lines) + "\n"


def run_audit(root, output, *, baseline=None, admission=None, max_states=2000):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError("new audit output must not alter the historical input directory")
    # Read and authenticate old records before producing any new artifact.
    integrity = verify_preflight_artifacts(root / "preflight.json")
    if integrity["status"] != "PASS":
        raise ValueError("historical artifact integrity failed: " + str(integrity))
    replay = verify_bundle(root)
    if replay != read(root / "shape-coverage.json"):
        raise ValueError("historical coverage differs from independent replay")
    manifest = read(root / "expected-shapes.json")
    source = git_snapshot()
    hashes = {name: file_hash(ROOT / name) for name in SOURCE_FILES}
    config = ProtocolConfig.from_dict(read(root / "config.json"))
    taxonomy = build_taxonomy(manifest, hashes, capacity=config.data["admission_limit"])
    records = classify_extras(root, taxonomy, replay)
    enumeration = enumerate_model(taxonomy, max_states=max_states)
    baseline_status = "UNKNOWN"
    if baseline is not None:
        original = read(baseline)
        computed = hashlib.sha256(json.dumps(original["files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        baseline_status = "PASS" if (computed == manifest["git"]["source_tree_sha256"] == original["source_tree_sha256"] and
            all(original["files"].get(name) == value for name, value in hashes.items())) else "FAIL"
    witnesses = cohort_witnesses(taxonomy)
    old_keys = {shape_key(s) for s in manifest["shapes"]}
    conditional_keys = {shape_key(w["shape"]) for w in witnesses}
    new_manifest = {"schema_version": VERSION, "kind": "conditional_research_manifest_not_admission",
        "taxonomy_sha256": digest(taxonomy), "historical_manifest_sha256": digest(manifest),
        "retained_v1_required_shapes": manifest["shapes"], "conditional_cohort_derivations": witnesses,
        "unknown_obligations": ["frozen-clock and admission reachability", "within-phase failures",
                                "historical actual membership and per-row cache/slot association"],
        "closure": "PARTIALLY_CLOSED", "formal_warmup_coverage": False}
    diff = {"schema_version": VERSION, "old_version": 1, "new_version": VERSION,
        "old_manifest_sha256": digest(manifest), "new_manifest_sha256": digest(new_manifest),
        "retained_required": len(old_keys), "removed_required": 0, "normalization_changes": 0,
        "conditional_distinct_shapes": len(conditional_keys),
        "new_conditional_shapes": len(conditional_keys - old_keys), "new_admission_allowed_shapes": 0}
    checks = [{"code": "HISTORICAL_ARTIFACT_INTEGRITY", "status": integrity["status"], "evidence": "historical artifacts.json"},
        {"code": "HISTORICAL_SOURCE_BINDING", "status": baseline_status, "evidence": "pre-edit source baseline plus six rule source hashes"},
        {"code": "LOCAL_TENSOR_RULES", "status": "PASS" if all(r["local_rule_check"]["status"] == "PASS" for r in records) else "FAIL", "evidence": "extra-shape-audit-v2.json"},
        {"code": "RAW_REFERENCE_COMPLETENESS", "status": "PASS" if records and all(r["raw_references"] for r in records) else "FAIL", "evidence": "extra-shape-audit-v2.json"},
        {"code": "FINITE_WIDTH_BOUND", "status": "PASS", "evidence": "taxonomy-v2.json constraints/proof"},
        {"code": "ABSTRACT_ENUMERATION", "status": enumeration["enumeration_status"], "evidence": "enumeration-v2.json"},
        {"code": "FROZEN_SCHEDULE_EQUIVALENCE", "status": "UNKNOWN", "evidence": enumeration["reason"]},
        {"code": "ACTUAL_REQUEST_PATH", "status": "UNKNOWN", "evidence": "v1 collector fields lack membership"}]
    result = {"schema_version": VERSION, "kind": "historical_shape_research_not_performance",
        "historical_run_id": root.name, "audit_run_id": output.name,
        "git": {k: v for k, v in source.items() if k != "files"},
        "historical_git": manifest["git"], "config_sha256": config.sha256,
        "taxonomy_sha256": digest(taxonomy), "rule_sha256": taxonomy["rule_sha256"],
        "evidence_schema_version": VERSION, "evidence_schema_sha256": digest(EVIDENCE_SCHEMA),
        "counts": {"extra_shapes": len(records), "observations": sum(r["observation_count"] for r in records),
            "duplicate_observations": sum(r["duplicate_observations"] for r in records),
            "categories": dict(Counter(r["category"] for r in records))},
        "checks": checks, "historical_shape_gate": replay,
        "historical_admission": read(admission) if admission is not None else None,
        "enumeration_summary": {k: v for k, v in enumeration.items() if k != "shapes"},
        "closure": "UNKNOWN" if any(c["status"] == "FAIL" for c in checks) else "PARTIALLY_CLOSED",
        "admitted": False, "formal_requests_launched": 0,
        "historical_root": root.relative_to(ROOT).as_posix() if root.is_relative_to(ROOT) else root.name}
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("taxonomy-v2.json", taxonomy), ("evidence-schema-v2.json", EVIDENCE_SCHEMA), ("manifest-v2.json", new_manifest),
                        ("manifest-diff-v2.json", diff), ("extra-shape-audit-v2.json", records),
                        ("enumeration-v2.json", enumeration), ("audit-v2.json", result)):
        write_new(output / name, value)
    with (output / "report.md").open("x", encoding="utf-8", newline="\n") as report:
        report.write(markdown(result, records))
    files = ["config.json", "expected-shapes.json", "model-config.json", "preflight.json",
             "shape-coverage.json", "shape-proof-index.json", "artifacts.json",
             "legacy/observer.jsonl", "continuous/observer.jsonl"]
    write_new(output / "artifacts.json", {"schema_version": VERSION,
        "historical_files_sha256": {name: file_hash(root / name) for name in files},
        "baseline_sha256": file_hash(baseline) if baseline else None,
        "historical_admission_sha256": file_hash(admission) if admission else None,
        "sha256": {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--max-states", type=int, default=2000)
    args = parser.parse_args()
    result = run_audit(args.input, args.output, baseline=args.baseline, admission=args.admission, max_states=args.max_states)
    print(json.dumps({k: result[k] for k in ("counts", "closure", "admitted", "taxonomy_sha256", "rule_sha256")}))
    raise SystemExit(0 if result["closure"] == "SHAPE_SPACE_CLOSED" and result["admitted"] else 1)


if __name__ == "__main__":
    main()
