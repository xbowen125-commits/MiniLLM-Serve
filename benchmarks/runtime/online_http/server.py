"""Benchmark-only process host; no production source/HTTP generation edits.

The real backend replaces only model *loading* with an offline pinned snapshot
loader shared by both execution modes. The fake backend validates HTTP tooling,
not model performance. No private metrics endpoint is added to the application.
"""

import argparse
from contextlib import asynccontextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
from types import SimpleNamespace

from .artifacts import environment_snapshot, git_snapshot, write_new
from .config import load_config


class FakeTokenizer:
    chat_template = "benchmark-fake-template-v1"
    pad_token_id = 0
    eos_token_id = 999999

    def apply_chat_template(self, messages, **kwargs):
        return " ".join(f"{m['role']}: {m['content']}" for m in messages) + " assistant:"

    def encode(self, text, add_special_tokens=True):
        if "[FAIL]" in text:
            raise ValueError("intentional CPU fake request failure")
        return list(range(1, len(text.split()) + 1))

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(f"w{token} " for token in tokens)


def fake_backend(app, settings):
    from minillm.contracts import ExecutionPhase, FinishReason
    from minillm.continuous import ContinuousBatchStepResult
    from minillm.engine import GenerationChunk, GenerationResult
    from minillm.memory.slot_kv_pool import SlotKVPool
    from minillm.runtime.http_owner import ContinuousHTTPOwner
    engine = app.state.engine

    def load():
        engine.model = SimpleNamespace(config=SimpleNamespace())
        engine.tokenizer = FakeTokenizer()
        engine.device = "cpu"
        engine.dtype = "float32"
    engine.load = load

    def stream(messages, *, cancel_event, config):
        prompt = engine.build_prompt(messages)
        prompt_count = len(engine.tokenizer.encode(prompt))
        for index in range(config.max_tokens):
            time.sleep(.002)
            if cancel_event.is_set():
                return
            yield GenerationChunk(f"w{index + 1} ", None, prompt_count, index + 1)
        yield GenerationChunk("", "length", prompt_count, config.max_tokens)

    def chat(messages, *, cancel_event, config):
        chunks = list(stream(messages, cancel_event=cancel_event, config=config))
        if not chunks or chunks[-1].finish_reason is None:
            raise RuntimeError("fake request cancelled")
        return GenerationResult("".join(c.text for c in chunks), "length",
                                chunks[-1].prompt_tokens, chunks[-1].completion_tokens)
    engine.stream_chat = stream
    engine.chat = chat

    class FakeBatchWorker:
        max_batch_size = 2
        def __init__(self):
            self.pool = SlotKVPool(settings.continuous_batch_size)
            self.contexts = {}
        def prefill_batch(self, contexts):
            time.sleep(.002)
            results = []
            for context in contexts:
                cache = object()
                self.pool.allocate(context.request_id, cache)
                context.request.kv_cache_ref = cache
                self.contexts[context.request_id] = context
                results.append(ContinuousBatchStepResult(context.request_id, ExecutionPhase.PREFILL,
                    prompt_tokens=len(engine.tokenizer.encode(context.request.prompt))))
            return results
        def decode_batch(self, contexts):
            time.sleep(.002)
            results = []
            for context in contexts:
                token = len(context.request.generated_tokens) + 1
                finished = token >= context.request.generation_config.max_tokens
                results.append(ContinuousBatchStepResult(context.request_id, ExecutionPhase.DECODE,
                    token_id=token, finished=finished,
                    finish_reason=FinishReason.LENGTH if finished else None))
                if finished:
                    self.release_request(context.request_id)
            return results
        def release_request(self, rid):
            context = self.contexts.pop(rid, None)
            if context is not None:
                context.request.kv_cache_ref = None
            self.pool.release(rid)
        def close(self):
            for rid in tuple(self.contexts):
                self.release_request(rid)
    if settings.execution_mode == "continuous":
        # create_app owns the instantiated owner; replace its optional backend
        # injection boundary before lifespan startup, not the HTTP handler.
        app.state.continuous_owner.worker_factory = lambda _engine: FakeBatchWorker()


def pinned_assets(data):
    from huggingface_hub import snapshot_download
    model_path = Path(snapshot_download(data["model"], revision=data["model_revision"], local_files_only=True))
    tokenizer_path = Path(snapshot_download(data["model"], revision=data["tokenizer_revision"], local_files_only=True))
    if model_path.name != data["model_revision"] or tokenizer_path.name != data["tokenizer_revision"]:
        raise RuntimeError("cached revision does not match frozen protocol")
    files = {}
    for path in tokenizer_path.iterdir():
        if path.is_file() and any(name in path.name for name in ("tokenizer", "vocab", "merges", "chat_template")):
            files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return model_path, tokenizer_path, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("legacy", "continuous"), required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--trial-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    data = config.data
    source = {key: value for key, value in git_snapshot().items() if key != "files"}
    write_new(args.trial_dir / "server-starting.json", {**environment_snapshot(),
        "config_sha256": config.sha256, "git": source, "mode": args.mode, "port": args.port})
    from minillm.config import Settings
    from minillm.server import create_app
    from minillm.continuous_server import ContinuousServer
    import uvicorn
    settings = Settings(model_name=data["model"], device=data["device"], dtype=data["dtype"],
        execution_mode=args.mode, max_model_len=data["max_model_len"],
        max_concurrent_generations=data["admission_limit"], request_queue_capacity=data["queue_capacity"],
        queue_timeout_seconds=data["queue_timeout_s"], inference_timeout_seconds=data["inference_timeout_s"],
        continuous_batch_size=data["admission_limit"], continuous_output_capacity=data["output_capacity"],
        continuous_send_timeout_seconds=data["send_timeout_s"], use_kv_cache=True)
    app = create_app(settings)
    from .usage import install
    install(app)
    metadata = {**environment_snapshot(), "mode": args.mode, "config_sha256": config.sha256, "git": source,
        "model": data["model"], "model_revision": data["model_revision"],
        "tokenizer_revision": data["tokenizer_revision"], "dtype": data["dtype"], "device": data["device"],
        "settings": {key: value for key, value in vars(settings).items()}}
    metadata["benchmark_sse_usage"] = "terminal_request_state_v1"
    if data["evidence_kind"] == "cpu_fake":
        fake_backend(app, settings)
        metadata.update(cuda_runtime=None, gpu=None, tokenizer_files_sha256={"fake": "word-tokenizer-v1"})
    else:
        import torch
        if data["device"] == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no CPU fallback")
        model_path, tokenizer_path, files = pinned_assets(data)
        def load():
            from transformers import AutoModelForCausalLM, AutoTokenizer
            engine = app.state.engine
            engine.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
            engine.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                dtype=getattr(torch, data["dtype"])).to(data["device"])
            engine.model.eval()
            engine.device, engine.dtype = data["device"], getattr(torch, data["dtype"])
            if engine.tokenizer.pad_token_id is None:
                engine.tokenizer.pad_token_id = engine.tokenizer.eos_token_id
        app.state.engine.load = load
        metadata.update(cuda_runtime=torch.version.cuda,
            gpu=torch.cuda.get_device_name() if data["device"] == "cuda" else None,
            tokenizer_files_sha256=files)
    original = app.router.lifespan_context
    @asynccontextmanager
    async def lifespan(app):
        async with original(app):
            engine = app.state.engine
            metadata["tokenizer_class"] = type(engine.tokenizer).__name__
            metadata["chat_template_sha256"] = hashlib.sha256(
                str(engine.tokenizer.chat_template).encode("utf-8")).hexdigest()
            if data["evidence_kind"] == "cpu_fake":
                metadata.update(actual_model_dtype="fake", actual_model_device="cpu")
            else:
                parameter = next(engine.model.parameters())
                metadata.update(actual_model_dtype=str(parameter.dtype), actual_model_device=str(parameter.device))
                if parameter.dtype != getattr(torch, data["dtype"]) or parameter.device.type != data["device"]:
                    raise RuntimeError("loaded model precision/device differs from frozen protocol")
            write_new(args.trial_dir / "server-ready.json", metadata)
            try:
                yield
            finally:
                # Observe genuine drain before original context closes worker.
                # Client runner drains before asking this server to stop.
                pass
        write_new(args.trial_dir / "server-stopped.json", {"pid": os.getpid(), "config_sha256": config.sha256, "git": source,
            "owner_thread_alive": bool(app.state.continuous_owner and app.state.continuous_owner.thread.is_alive()),
            "pending_finalizers": len(app.state.cleanup_tasks), "stream_consumers": len(app.state.stream_sessions)})
    app.router.lifespan_context = lifespan
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", args.port))
    listener.listen(128)
    metadata["port"] = listener.getsockname()[1]
    server = ContinuousServer(uvicorn.Config(app, host="127.0.0.1", port=args.port,
        workers=1, log_level="warning", access_log=False))
    def control():
        line = sys.stdin.readline()
        if not line or line.strip() == "STOP":
            server.should_exit = True
    threading.Thread(target=control, daemon=True, name="benchmark-stop-control").start()
    try:
        server.run(sockets=[listener])
    finally:
        listener.close()


if __name__ == "__main__":
    main()
