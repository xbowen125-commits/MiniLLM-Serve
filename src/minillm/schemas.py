"""OpenAI 兼容的请求/响应数据结构（/v1/chat/completions）。

参考 https://platform.openai.com/docs/api-reference/chat
同时支持完整响应与 SSE 增量响应。
"""

from __future__ import annotations

import time
import uuid
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(min_length=1)
    messages: list[ChatMessage]
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=-1, ge=-1)  # -1 = 不限制
    stop: Optional[list[str]] = Field(default=None)
    stream: bool = Field(default=False)
    seed: Optional[int] = None

    @field_validator("stop", mode="before")
    @classmethod
    def _normalize_stop(cls, value):
        if isinstance(value, str):
            value = [value]
        if value is not None:
            if not isinstance(value, list):
                raise ValueError("stop 必须是字符串或字符串数组")
            if not value or len(value) > 4:
                raise ValueError("stop 必须包含 1 到 4 个字符串")
            if any(not isinstance(item, str) or not item for item in value):
                raise ValueError("stop 不能包含空字符串")
        return value

    @model_validator(mode="after")
    def _check_messages(self) -> "ChatCompletionRequest":
        if not self.messages:
            raise ValueError("messages 不能为空")
        return self


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatMessageOut(BaseModel):
    role: str = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessageOut
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


class ChatCompletionChunkDelta(BaseModel):
    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


class APIError(BaseModel):
    message: str
    type: str
    code: Optional[str] = None


class APIErrorResponse(BaseModel):
    error: APIError


def make_response_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def make_response(
    model: str,
    content: str,
    finish_reason: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=make_response_id(),
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessageOut(content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def make_chunk(
    response_id: str,
    created: int,
    model: str,
    *,
    role: Optional[Literal["assistant"]] = None,
    content: Optional[str] = None,
    finish_reason: Optional[Literal["stop", "length"]] = None,
) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id=response_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                delta=ChatCompletionChunkDelta(role=role, content=content),
                finish_reason=finish_reason,
            )
        ],
    )
