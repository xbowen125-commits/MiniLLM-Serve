"""Deadline-bounded HTTP/1.1 observations; content events are never token IDs."""

import hashlib
import http.client
import json
import socket
import threading
import time
from urllib.parse import urlsplit

from .metrics import calculate_request_metrics


def get_json(base_url, path, timeout=5):
    address = urlsplit(base_url)
    connection = http.client.HTTPConnection(address.hostname, address.port, timeout=timeout)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"status probe HTTP {response.status}")
        return json.loads(response.read())
    finally:
        connection.close()


def request_record(config, item):
    """Common nullable raw schema, also used for failures before transport."""
    data = config.data
    return {"request_id": item["request_id"], "scheduled_at_s": item["arrival_s"],
        "sent_at_s": None, "first_content_at_s": None, "last_content_at_s": None,
        "terminal_at_s": None, "http_status": None, "response_id": None, "lease_id": None,
        "success": False, "outcome": "error", "error_code": None, "error_message": None,
        "visible_output_tokens": None, "generated_output_tokens": None,
        "generated_count_source": "unavailable_sse" if data["stream"] else "server_usage",
        "expected_visible_output_tokens": item.get("expected_visible_output_tokens"),
        "expected_prompt_tokens": item.get("expected_prompt_tokens"),
        "prompt_tokens": item.get("observed_prompt_tokens"), "max_tokens": item["max_tokens"],
        "prompt_count_source": "pinned_tokenizer_chat_template_preflight",
        "finish_reason": None, "saw_done": False, "content_events": 0,
        "protocol_errors": [], "runtime_metrics": None, "send_lateness_ms": None,
        "queue_wait_ms": None, "effective_timeout_s": None, "output_sha256": None,
        "reached_requested_max_tokens": None,
        "generator_scheduled_at_s": None, "worker_started_at_s": None,
        "body_started_at_s": None, "body_ready_at_s": None, "http_call_at_s": None,
        "generator_lateness_ms": None, "pool_wait_ms": None,
        "body_build_ms": None, "client_preparation_ms": None, "http_setup_ms": None}


def observe_request(base_url, config, item, origin_s, token_count, *, now=time.perf_counter,
                    run_deadline=None, generator_scheduled_at_s=None, worker_started_at_s=None,
                    identity_record_id=None):
    """Measure send-call start -> first/last visible content -> DONE or failure.

    Body serialization precedes timing. Each request uses a fresh connection in
    both modes; connect, network and JSON parsing are client-observed overhead.
    TCP ingress times and internal model token production remain unobservable.
    """
    body_started = now() - origin_s
    data = config.data
    body = json.dumps({"model": data["model"], "messages": item["messages"],
        "max_tokens": item["max_tokens"], "stream": data["stream"],
        **{key: data[key] for key in ("temperature", "top_p", "top_k", "seed", "stop")}},
        ensure_ascii=False).encode("utf-8")
    body_ready = now() - origin_s
    row = request_record(config, item)
    if identity_record_id is not None:
        row.update(planned_request_id=item["request_id"], client_record_id=identity_record_id,
                   request_body_sha256=hashlib.sha256(body).hexdigest())
    row.update(generator_scheduled_at_s=generator_scheduled_at_s, worker_started_at_s=worker_started_at_s,
               body_started_at_s=body_started, body_ready_at_s=body_ready)
    address = urlsplit(base_url)
    response = transport = None
    expired = threading.Event()
    text = ""
    began = now()
    deadline = began + data["request_timeout_s"]
    if run_deadline is not None:
        deadline = min(deadline, run_deadline)
    budget = max(0, deadline - began)
    connection = http.client.HTTPConnection(address.hostname, address.port, timeout=max(budget, .001))
    row["effective_timeout_s"] = budget
    row["sent_at_s"] = began - origin_s
    row["send_lateness_ms"] = (row["sent_at_s"] - item["arrival_s"]) * 1000
    row["body_build_ms"] = (body_ready - body_started) * 1000
    if generator_scheduled_at_s is not None:
        row["generator_lateness_ms"] = (generator_scheduled_at_s - item["arrival_s"]) * 1000
    if worker_started_at_s is not None:
        row["client_preparation_ms"] = (row["sent_at_s"] - worker_started_at_s) * 1000
        if generator_scheduled_at_s is not None:
            row["pool_wait_ms"] = (worker_started_at_s - generator_scheduled_at_s) * 1000

    def remaining():
        value = deadline - now()
        if value <= 0:
            raise TimeoutError("absolute HTTP request deadline exceeded")
        sock = connection.sock or transport
        if sock is not None:
            sock.settimeout(value)

    def abort():
        expired.set()
        sock = connection.sock or transport
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
    watchdog = threading.Timer(budget, abort)
    watchdog.daemon = True
    watchdog.start()

    try:
        remaining()
        row["http_call_at_s"] = now() - origin_s
        row["http_setup_ms"] = (row["http_call_at_s"] - row["sent_at_s"]) * 1000
        headers = {"Content-Type": "application/json", "Connection": "close"}
        if identity_record_id is not None:
            headers.update({"X-MiniLLM-Planned-ID": item["request_id"],
                            "X-MiniLLM-Client-ID": identity_record_id})
        if data.get("authoritative_sse_usage"):
            headers["X-MiniLLM-Benchmark-Usage"] = "1"
        connection.request("POST", "/v1/chat/completions", body, headers)
        transport = connection.sock
        remaining()
        response = connection.getresponse()
        row["http_status"] = response.status
        row["lease_id"] = response.getheader("X-MiniLLM-Request-ID")
        value = response.getheader("X-MiniLLM-Queue-Wait-Ms")
        row["queue_wait_ms"] = float(value) if value is not None else None
        payload = None
        if response.status != 200 or not data["stream"]:
            remaining()
            raw = response.read(8 * 1024 * 1024 + 1)
            row["terminal_at_s"] = now() - origin_s
            if len(raw) > 8 * 1024 * 1024:
                raise ValueError("response exceeds bounded size")
            payload = json.loads(raw)
            if response.status == 200 and "error" not in payload:
                text = payload["choices"][0]["message"]["content"]
                row["response_id"] = payload["id"]
                row["finish_reason"] = payload["choices"][0]["finish_reason"]
                usage = payload.get("usage", {})
                row["generated_output_tokens"] = usage.get("completion_tokens")
                row["prompt_tokens"] = usage.get("prompt_tokens")
                row["prompt_count_source"] = "server_usage"
        else:
            if not (response.getheader("Content-Type") or "").startswith("text/event-stream"):
                raise ValueError("expected SSE Content-Type")
            role_seen = False
            finish_seen = False
            event_type = None
            usage_seen = False
            for _ in range(item["max_tokens"] * 4 + 100):
                remaining()
                line = response.readline(1024 * 1024 + 1)
                arrived = now() - origin_s
                if not line:
                    break
                if len(line) > 1024 * 1024:
                    raise ValueError("SSE line exceeds bounded size")
                if line.startswith(b"event: "):
                    event_type = line[7:].strip().decode()
                    continue
                if not line.startswith(b"data: "):
                    if not line.strip(): event_type = None
                    continue
                event = line[6:].strip().decode("utf-8", errors="strict")
                if event == "[DONE]":
                    row["saw_done"] = True
                    row["terminal_at_s"] = arrived
                    break
                event = json.loads(event)
                if event_type == "minillm.usage":
                    if (usage_seen or event.get("id") != row["response_id"] or
                            event.get("request_id") != row["lease_id"] or event.get("object") != "minillm.benchmark.usage"):
                        row["protocol_errors"].append("invalid usage identity or duplicate usage")
                    else:
                        usage_seen = True
                        row["authoritative_usage"] = event
                        row["generated_output_tokens"] = event["usage"]["completion_tokens"]
                        row["generated_count_source"] = "terminal_request_state_sse"
                        from .gate import authoritative_count
                        if row["generated_output_tokens"] is not None and not authoritative_count(row, data):
                            row["protocol_errors"].append("unverified terminal usage metadata")
                        if event["usage"]["prompt_tokens"] is not None:
                            row["prompt_tokens"] = event["usage"]["prompt_tokens"]
                            row["prompt_count_source"] = "server_usage"
                    continue
                if "error" in event:
                    payload = event
                    continue
                if event.get("object") != "chat.completion.chunk":
                    row["protocol_errors"].append("invalid chunk object")
                current_id = event.get("id")
                if not row["response_id"]:
                    row["response_id"] = current_id
                if not current_id or current_id != row["response_id"]:
                    row["protocol_errors"].append("unstable response ID")
                choice = event["choices"][0]
                delta = choice.get("delta", {})
                if delta.get("role"):
                    if role_seen or row["content_events"] or delta["role"] != "assistant":
                        row["protocol_errors"].append("invalid role order")
                    role_seen = True
                content = delta.get("content")
                if content:
                    if not role_seen or finish_seen:
                        row["protocol_errors"].append("content outside role/finish boundaries")
                    text += content
                    if len(text) > 8 * 1024 * 1024:
                        raise ValueError("visible output exceeds bounded size")
                    row["content_events"] += 1
                    # Timestamp is receipt/parsing time, not time spent running
                    # this offline tokenizer. The tokenizer confirms visibility.
                    if row["first_content_at_s"] is not None or token_count(text) > 0:
                        if row["first_content_at_s"] is None:
                            row["first_content_at_s"] = arrived
                        row["last_content_at_s"] = arrived
                if choice.get("finish_reason") is not None:
                    if finish_seen:
                        row["protocol_errors"].append("duplicate finish")
                    finish_seen = True
                    row["finish_reason"] = choice["finish_reason"]
            if not row["saw_done"]:
                row["protocol_errors"].append("missing DONE")
            if data.get("authoritative_sse_usage") and not usage_seen:
                row["protocol_errors"].append("missing authoritative usage")
            if payload is None and (not role_seen or not finish_seen):
                row["protocol_errors"].append("missing role or finish")
        if payload and "error" in payload:
            row["error_code"] = payload["error"].get("code") or payload["error"].get("type") or "server_error"
            row["error_message"] = payload["error"].get("message")
        if row["error_code"] in {"queue_timeout", "inference_timeout"}:
            row["outcome"] = "timeout"
        elif row["error_code"] in {"request_cancelled", "cancelled"}:
            row["outcome"] = "cancelled"
        elif response.status == 200 and not row["error_code"] and not row["protocol_errors"]:
            if row["finish_reason"] not in {"stop", "length"}:
                row["protocol_errors"].append("invalid finish reason")
            else:
                row.update(success=True, outcome="completed")
    except (TimeoutError, socket.timeout) as exc:
        row.update(outcome="timeout", error_code="client_timeout", error_message=str(exc))
    except Exception as exc:
        row.update(error_code="client_timeout" if expired.is_set() else type(exc).__name__, error_message=str(exc))
        if expired.is_set():
            row["outcome"] = "timeout"
    finally:
        if row["terminal_at_s"] is None:
            row["terminal_at_s"] = now() - origin_s
        watchdog.cancel()
        if response is not None:
            response.close()
        connection.close()
    try:
        row["visible_output_tokens"] = token_count(text)
    except Exception as exc:
        row.update(success=False, outcome="error", error_code="client_tokenization_error", error_message=str(exc))
    row["reached_requested_max_tokens"] = (row["generated_output_tokens"] == item["max_tokens"]
        if row["generated_output_tokens"] is not None else None)
    row["output_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return calculate_request_metrics(row)
