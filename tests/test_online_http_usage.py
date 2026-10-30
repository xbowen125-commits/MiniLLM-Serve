"""Terminal counters, never retokenized visible text or artificial content."""
import asyncio
import json
import threading
from types import SimpleNamespace
import pytest
from minillm.engine.request import RequestState
from benchmarks.runtime.online_http.usage import usage_payload


@pytest.mark.parametrize("ending", ["finished", "failed", "cancelled"])
def test_terminal_counter_includes_eos_stop_and_hidden_tokens(ending):
    state = RequestState()
    state.mark_running()
    for token in [1, 9, 7]: state.append_token(token)
    getattr(state, "mark_" + ending)()
    result = usage_payload(state, response_id="chatcmpl-a", lease_id=1, prompt_tokens=2)
    assert result["usage"] == {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
    assert result["status"] == ending and "choices" not in result
    assert result["includes_eos"] and result["includes_stop_and_hidden_tokens"]


def test_nonterminal_or_unknown_counter_is_na_not_zero():
    for state in (None, RequestState()):
        assert usage_payload(state, response_id="a", lease_id=1)["usage"]["completion_tokens"] is None


def test_named_counter_requires_identity_and_opt_in_at_gate():
    from benchmarks.runtime.online_http.gate import authoritative_count
    state = RequestState(); state.mark_running(); state.append_token(9); state.mark_finished()
    payload = usage_payload(state, response_id="chatcmpl-x", lease_id=2, prompt_tokens=3)
    row = {"response_id": "chatcmpl-x", "lease_id": "2", "authoritative_usage": payload,
        "generated_output_tokens": 1, "generated_count_source": "terminal_request_state_sse"}
    assert authoritative_count(row, {"stream": True, "authoritative_sse_usage": True})
    assert not authoritative_count(row, {"stream": True})
    row["lease_id"] = "3"
    assert not authoritative_count(row, {"stream": True, "authoritative_sse_usage": True})


def test_only_actual_warmup_forwards_count_and_full_tensor_proof_is_not_inferred():
    from benchmarks.runtime.online_http.config import load_config
    from benchmarks.runtime.online_http.preflight import required_shapes, warmup_coverage
    config = load_config("benchmarks/runtime/online_http/configs/qwen_admission_v2.json")
    events = [{"kind": "forward_completed", "success": True, "execution_stage": "warmup",
        "mode": shape["mode"], "phase": shape["phase"], "input_shape": [shape["batch_size"], 37],
        "attention_mask_shape": [shape["batch_size"], 37], "kv_length_before": 0,
        "kv_tensor_shapes_before": []} for shape in required_shapes(config)]
    report = warmup_coverage(config, events)
    assert not report["missing_probe_shapes"]
    assert report["tensor_shape_coverage"]["status"] == "UNKNOWN"
    assert report["formal_warmup_coverage"] is False and report["status"] != "PASS"
    for row in events: row["execution_stage"] = "fault"
    assert warmup_coverage(config, events)["observed_shapes"] == []


def test_versioned_recipe_runs_warmup_in_each_real_fake_server_process(tmp_path):
    from benchmarks.runtime.online_http.config import ProtocolConfig, load_config
    from benchmarks.runtime.online_http.preflight import run_preflight
    data = load_config("benchmarks/runtime/online_http/configs/qwen_admission_v2.json").data
    data.update(evidence_kind="cpu_fake", device="cpu")
    data["requests"] = data["requests"][:2]
    result = run_preflight(ProtocolConfig.from_dict(data), tmp_path / "process-warmup", scope="single")
    assert len(result["modes"]) == 2 and result["status"] != "PASS"
    for mode in result["modes"]:
        warm = mode["process_warmup"]
        assert warm["pid"] == mode["metadata"]["pid"]
        assert warm["requests"] == 4 and warm["finished_at_s"] >= warm["started_at_s"]
        assert mode["shutdown"]["stopped"]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("ending", ["finished", "failed", "cancelled"])
@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_extension_is_separate_named_event_and_default_frames_unchanged(monkeypatch, enabled, ending, finish_reason):
    import minillm.server as server
    from minillm.engine.generation import GenerationRunner
    import benchmarks.runtime.online_http.usage as usage
    originals = {name: getattr(server, name) for name in ("_stream_chat_events", "_continuous_chat_events")}
    async def events(session):
        yield server._sse_data({"id": "chatcmpl-test", "object": "chat.completion.chunk", "choices": []})
        yield server._sse_data({"id": "chatcmpl-test", "choices": [{"finish_reason": finish_reason}]}
            if ending == "finished" else {"error": {"code": ending}})
        yield server._sse_data("[DONE]")
    def stream(self, **kwargs):
        yield from ()
    for name in originals: monkeypatch.setattr(server, name, events)
    monkeypatch.setattr(server._StreamSession, "__init__", server._StreamSession.__init__)
    monkeypatch.setattr(GenerationRunner, "stream", stream)
    app = SimpleNamespace(add_middleware=lambda *_: None)
    usage.install(app)
    marker = usage._enabled.set(enabled)
    try:
        lease = SimpleNamespace(request_id=3, cancel_event=threading.Event())
        engine = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda _: [1, 2]))
        session = server._StreamSession(None, engine, lease)
        state = RequestState(prompt="test")
        state.mark_running(); state.append_token(9); getattr(state, "mark_" + ending)()
        list(GenerationRunner.stream(object(), request=state, cancel_event=lease.cancel_event))
        async def run(): return [item async for item in server._stream_chat_events(session)]
        frames = asyncio.run(run())
    finally:
        usage._enabled.reset(marker)
    assert frames[-1] == b"data: [DONE]\n\n"
    assert len(frames) == (4 if enabled else 3)
    if enabled:
        assert frames[-2].startswith(b"event: minillm.usage\n")
        value = json.loads(frames[-2].split(b"data: ", 1)[1])
        assert value["usage"]["completion_tokens"] == 1
        assert value["id"] == "chatcmpl-test" and value["request_id"] == "3"
