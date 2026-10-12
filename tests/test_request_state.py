"""RequestState lifecycle tests."""

import pytest

from minillm.engine import GenerationConfig, RequestState, RequestStatus


def test_request_state_defaults():
    before = __import__("time").time()
    request = RequestState(prompt="hello", generation_config=GenerationConfig())
    after = __import__("time").time()

    assert request.request_id.startswith("req_")
    assert request.prompt == "hello"
    assert request.status is RequestStatus.WAITING
    assert before <= request.created_time <= after
    assert request.started_time is None
    assert request.finished_time is None
    assert request.generated_tokens == []
    assert request.kv_cache_ref is None


def test_request_state_records_running_and_finished_transitions():
    request = RequestState(
        request_id="request-1",
        prompt="hello",
        generation_config=GenerationConfig(),
        created_time=10.0,
    )

    request.mark_running(now=11.0)
    request.mark_finished(now=12.0)

    assert request.status is RequestStatus.FINISHED
    assert request.started_time == 11.0
    assert request.finished_time == 12.0
    assert request.is_terminal is True


@pytest.mark.parametrize(
    ("method_name", "expected"),
    [
        ("mark_failed", RequestStatus.FAILED),
        ("mark_cancelled", RequestStatus.CANCELLED),
    ],
)
def test_request_state_records_terminal_time(method_name, expected):
    request = RequestState(created_time=20.0)

    getattr(request, method_name)(now=21.0)

    assert request.status is expected
    assert request.started_time is None
    assert request.finished_time == 21.0


def test_request_state_appends_tokens_while_running():
    request = RequestState()
    request.mark_running(now=1.0)

    request.append_token(7)
    request.append_token(9)

    assert request.generated_tokens == [7, 9]


def test_request_state_rejects_invalid_transition_and_early_token():
    request = RequestState()

    with pytest.raises(RuntimeError, match="only be appended"):
        request.append_token(1)
    with pytest.raises(RuntimeError, match="invalid request transition"):
        request.mark_finished()
