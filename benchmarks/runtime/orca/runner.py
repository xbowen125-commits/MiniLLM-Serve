"""IterationScheduler benchmark using unmodified continuous runtime and worker."""

from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys
import torch

from minillm.continuous import BatchExecutor, IterationScheduler
from minillm.engine import GenerationConfig, RequestState
from minillm.runtime import RuntimeRequestContext
from ..continuous_batch.runner import FakeWorker
from ..continuous_batch.metrics import request_metrics
from ..continuous_batch.scenario import Scenario
from .metrics import summarize
from .scenario import workload_metadata
from ..common.arrival import normalize_requests, pop_due


def environment_metadata():
    root = Path(__file__).resolve().parents[3]
    def git(*args):
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else None
    try:
        commit = git("rev-parse", "HEAD")
        dirty = git("status", "--porcelain")
    except (OSError, subprocess.TimeoutExpired):
        commit, dirty = None, None
    return {"python": sys.version.split()[0], "pytorch": version("torch"),
        "transformers": version("transformers"), "cuda_build": torch.version.cuda,
        "gpu": None, "gpu_note": "not queried; no GPU used",
        "commit_sha": commit, "working_tree_dirty": None if dirty is None else bool(dirty)}


def run_benchmark(scenario: Scenario) -> dict:
    worker = FakeWorker(scenario)
    scheduler = IterationScheduler(max_batch_size=scenario.max_active,
                                   max_waiting=len(scenario.requests))
    executor = BatchExecutor(worker, scheduler=scheduler)
    workload = normalize_requests(scenario.requests)
    pending = list(workload)
    contexts, trace = {}, []
    budget = sum(r.max_tokens for r in pending) + len(pending)
    try:
        while pending or executor.batch.active_requests or executor.batch.waiting_requests:
            if not executor.batch.active_requests and not executor.batch.waiting_requests and pending:
                worker.now = max(worker.now, pending[0].arrival_time)
            admitted = []
            for request in pop_due(pending, worker.now):
                context = RuntimeRequestContext(RequestState(request_id=request.request_id,
                    prompt="simulation", created_time=request.arrival_time,
                    generation_config=GenerationConfig(max_tokens=request.max_tokens)))
                contexts[request.request_id] = context
                executor.add(context)
                admitted.append(request.request_id)
            start = worker.now
            step = executor.step()
            trace.append({**asdict(scheduler.last_state), "arrival_request_ids": admitted,
                "start_s": start, "end_s": worker.now,
                "completed_this_iteration": list(step.retired),
                "active_after": list(executor.batch.active_requests),
                "waiting_after": list(executor.batch.waiting_requests)})
            if len(trace) > budget:
                raise RuntimeError("iteration progress budget exceeded")
        rows = []
        metadata = {r["request_id"]: r for r in workload_metadata(scenario)}
        for spec in scenario.requests:
            observation = worker.observations[spec.request_id]
            rows.append({**metadata[spec.request_id], **request_metrics(spec.arrival_time,
                observation["started"], observation["first_token"], observation["finished"],
                len(contexts[spec.request_id].request.generated_tokens))})
        metrics = summarize(rows, worker.trace, trace, scenario.max_active,
                            worker.now - min(r.arrival_time for r in scenario.requests))
        return {"schema_version": 3, "scope": "orca-inspired-virtual-iteration-simulation",
            "environment": environment_metadata(),
            "normalized_workload": [asdict(r) for r in workload],
            "timestamp": datetime.now(timezone.utc).isoformat(), "scenario": asdict(scenario),
            "workload": workload_metadata(scenario),
            "metrics": metrics, "requests": rows, "iteration_trace": trace,
            "phase_trace": worker.trace,
            "idle_after_test": not executor.batch.active_requests and not executor.batch.waiting_requests}
    finally:
        executor.close()
