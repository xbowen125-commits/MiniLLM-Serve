"""Bounded warmup recipe inside each actual serving process session."""
from concurrent.futures import ThreadPoolExecutor
import time
from .artifacts import utc_now, write_new
from .client import observe_request


def run_warmup(config, trace, token_count, server, directory, journal, drain):
    identity = {"pid": server.metadata["pid"], "mode": server.metadata["mode"],
        "config_sha256": config.sha256, "model_revision": config.data["model_revision"]}
    start = time.perf_counter()
    write_new(directory / "warmup-active", {**identity, "timestamp": utc_now(), "monotonic_s": start})
    rows = []
    groups = [[item] for item in trace]
    if config.data.get("warmup_strategy") == "sequential_and_pairs":
        groups += [trace[i:i + 2] for i in range(0, len(trace), 2)]
    for group_index, group in enumerate(groups):
        origin = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(group)) as pool:
            results = list(pool.map(lambda item: observe_request(server.base_url, config,
                {**item, "arrival_s": 0}, origin, token_count), group))
        for row in results:
            row.update(probe="warmup", phase="warmup", performance_sample=False, group=group_index, **identity)
            journal.append(row)
        rows.extend(results)
        settled = drain(server.base_url, config, mode=identity["mode"])
        if not settled["idle_after_test"] or any(not row["success"] for row in results):
            raise RuntimeError("process-local warmup failed or cleanup unconfirmed")
    result = {**identity, "started_at_s": start, "finished_at_s": time.perf_counter(),
              "timestamp": utc_now(), "requests": len(rows), "status": "PASS", "recipe": config.data.get("warmup_strategy")}
    write_new(directory / "warmup-finished", result)
    return result
