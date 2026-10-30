"""Correctness-only subprocess instrumentation around the existing HTTP host.

Observers copy terminal token IDs and forward input dimensions; they do not
change sampling, scheduling, cache management or any production source file.
These instrumented requests must never be counted as performance samples.
"""

from contextlib import asynccontextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

from .artifacts import Journal, utc_now


def prompt_hash(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def install_observers(directory):
    """Install process-local read-only observers before lifespan startup."""
    import minillm.server as production
    from minillm.engine.request import RequestState
    from minillm.engine.generation import GenerationRunner

    journal = Journal(directory / "observer.jsonl")
    original_transition = RequestState._transition
    original_stream = GenerationRunner.stream
    original_create = production.create_app
    manifest_path = directory.parent / "expected-shapes.json"
    shape_identity = {}
    if manifest_path.exists():
        from .shapes import digest, IDENTITY, VERSION
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Identity is checked against independently recomputed server provenance
        # by the verifier; it is never accepted solely from this declaration.
        from .artifacts import git_snapshot
        source = {k: v for k, v in git_snapshot().items() if k != "files"}
        shape_identity = {k: manifest[k] for k in IDENTITY}
        shape_identity["git"] = source
        shape_identity.update(schema_version=VERSION, expected_manifest_sha256=digest(manifest),
            run_id=directory.parent.name, collector="forward-boundary-v1")
        mode = sys.argv[sys.argv.index("--mode") + 1]
        shape_identity["mode"] = mode

    def emit(kind, **fields):
        journal.append({**shape_identity, "observation_id": uuid.uuid4().hex,
                        "kind": kind, "timestamp": utc_now(),
                        "monotonic_s": time.perf_counter(), "pid": os.getpid(),
                        "execution_stage": "warmup" if (directory / "warmup-active").exists() and not
                        (directory / "warmup-finished").exists() else
                        "fault" if (directory / "fault-active").exists() else "correctness", **fields})

    def terminal(self, target, **kwargs):
        result = original_transition(self, target, **kwargs)
        if self.is_terminal and self.prompt:
            emit("terminal", request_id=self.request_id, prompt_sha256=prompt_hash(self.prompt),
                 generated_token_ids=list(self.generated_tokens), status=self.status.value,
                 generation_config=asdict(self.generation_config))
        return result

    def stream(self, **kwargs):
        try:
            yield from original_stream(self, **kwargs)
        finally:
            state = kwargs["request"]
            emit("legacy_runner_released", request_id=state.request_id,
                 kv_reference_cleared=state.kv_cache_ref is None)

    def create(*args, **kwargs):
        app = original_create(*args, **kwargs)
        original_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app):
            async with original_lifespan(app):
                engine = app.state.engine
                original_forward = getattr(engine.model, "forward", None)
                if original_forward is not None:
                    def forward(*args, **kwargs):
                        ids = kwargs.get("input_ids", args[0] if args else None)
                        mask = kwargs.get("attention_mask")
                        cache = kwargs.get("past_key_values")
                        cache_shapes = [] if cache is None else None
                        if cache is not None:
                            cache_shapes = [{"keys": list(layer.keys.shape), "values": list(layer.values.shape)}
                                for layer in getattr(cache, "layers", []) if getattr(layer, "keys", None) is not None]
                        shape = list(ids.shape) if ids is not None else None
                        fields = {"call_id": uuid.uuid4().hex, "input_shape": shape,
                            "batch_size": shape[0] if shape else None,
                            "padding_width": shape[-1] if shape else None,
                            "attention_mask_shape": list(mask.shape) if mask is not None else None,
                            "kv_tensor_shapes_before": cache_shapes,
                            "kv_length_before": cache.get_seq_length() if cache is not None else 0,
                            "phase": "prefill" if shape and shape[-1] > 1 else "decode",
                            "phase_inference": "input_width; one-token prompt is ambiguous",
                            "completion_boundary": "Python forward return; no added CUDA synchronization"}
                        parameter = next(engine.model.parameters(), None)
                        fields.update(dtype=str(parameter.dtype).removeprefix("torch.") if parameter is not None else None,
                                      device=parameter.device.type if parameter is not None else None)
                        emit("forward_attempt", **fields)
                        try:
                            flag = directory / "inject-next-forward"
                            if flag.exists():
                                flag.unlink()
                                emit("injected_forward_error", call_id=fields["call_id"])
                                raise RuntimeError("intentional preflight forward failure")
                            output = original_forward(*args, **kwargs)
                        except BaseException as exc:
                            emit("forward_completed", **fields, success=False,
                                 error={"type": type(exc).__name__, "message": str(exc)})
                            raise
                        emit("forward_completed", **fields, success=True, error=None)
                        return output
                    engine.model.forward = forward
                emit("observers_ready", actual_model_forward=original_forward is not None)
                yield
            emit("shutdown_complete", pending_finalizers=len(app.state.cleanup_tasks),
                 stream_consumers=len(app.state.stream_sessions),
                 scheduler=await app.state.request_scheduler.snapshot(),
                 continuous=app.state.continuous_owner.status() if app.state.continuous_owner else None,
                 owner_thread_alive=bool(app.state.continuous_owner and
                                         app.state.continuous_owner.thread.is_alive()))

        app.router.lifespan_context = lifespan
        return app

    RequestState._transition = terminal
    GenerationRunner.stream = stream
    production.create_app = create
    # The journal remains open until process exit; each event is flushed by
    # Journal.append, so the parent can inspect evidence before shutdown.
    return journal


def main():
    from . import server
    try:
        directory = Path(sys.argv[sys.argv.index("--trial-dir") + 1])
    except (ValueError, IndexError) as exc:
        raise SystemExit("--trial-dir is required") from exc
    trace = None
    if os.environ.get("MINILLM_BENCH_RUNTIME_TRACE") == "1":
        from .runtime_trace import RuntimeTrace
        from .artifacts import git_snapshot
        manifest = json.loads((directory.parent / "expected-shapes.json").read_text(encoding="utf-8"))
        source = {k: v for k, v in git_snapshot().items() if k != "files"}
        identity = {k: manifest[k] for k in ("config_sha256", "model_revision", "dtype", "device", "geometry")}
        identity.update(git=source, run_id=directory.parent.name)
        trace = RuntimeTrace(directory, identity).install()
    journal = install_observers(directory)
    try:
        server.main()
    finally:
        journal.close()
        if trace is not None:
            trace.close()
            from .trace_check import seal
            seal(directory)


if __name__ == "__main__":
    main()
