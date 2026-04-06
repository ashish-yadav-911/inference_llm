"""
serving/schemas.py

All Pydantic request/response models for the inference API.
Every type imported in server.py is defined here.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input prompt text")
    max_tokens: int = Field(512, ge=1, le=8192)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=1, le=10000)
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)
    stop: List[str] = Field(default_factory=list)
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    stream: bool = False

    @field_validator("prompt")
    @classmethod
    def prompt_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt cannot be blank or whitespace only")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "prompt": "Explain PagedAttention in vLLM.",
                "max_tokens": 256,
                "temperature": 0.7,
            }
        }
    }


class GenerateResponse(BaseModel):
    request_id: str
    text: str
    tokens_generated: int
    prompt_tokens: int
    latency_seconds: float
    tokens_per_second: float
    backend: str
    model: str


# ---------------------------------------------------------------------------
# Streaming (SSE)
# ---------------------------------------------------------------------------

class StreamChunk(BaseModel):
    """One chunk in a token-by-token SSE stream."""
    token: str
    index: int
    done: bool = False


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

class BatchRequest(BaseModel):
    prompts: List[str] = Field(..., min_length=1)
    max_tokens: int = Field(256, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)

    @field_validator("prompts")
    @classmethod
    def check_batch_size(cls, v: List[str]) -> List[str]:
        if len(v) > 64:
            raise ValueError("maximum 64 prompts per batch request")
        if any(not p.strip() for p in v):
            raise ValueError("prompts must not contain blank entries")
        return v


class BatchResponse(BaseModel):
    texts: List[str]
    count: int
    latency_seconds: float
    tokens_per_second: float


# ---------------------------------------------------------------------------
# OpenAI-compatible  (/v1/completions  and  /v1/chat/completions)
# ---------------------------------------------------------------------------

class OpenAICompletionRequest(BaseModel):
    """POST /v1/completions — drop-in for openai.Completion.create()"""
    model: str = "local"
    prompt: str
    max_tokens: int = Field(512, ge=1, le=8192)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    stop: Optional[List[str]] = None
    stream: bool = False
    n: int = Field(1, ge=1, le=1, description="Only n=1 is supported")


class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    logprobs: None = None
    finish_reason: Literal["stop", "length"] = "stop"


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAICompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:12]}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage


# OpenAI chat format
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """POST /v1/chat/completions — drop-in for openai.ChatCompletion.create()"""
    model: str = "local"
    messages: List[ChatMessage]
    max_tokens: int = Field(512, ge=1, le=8192)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    stream: bool = False

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, v: List[ChatMessage]) -> List[ChatMessage]:
        if not v:
            raise ValueError("messages cannot be empty")
        return v


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "length"] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatChoice]
    usage: CompletionUsage


# ---------------------------------------------------------------------------
# Health and metrics
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "starting"]
    backend: str
    model: str
    quantization: str
    device: str
    gpu_count: int
    gpu_allocated_gb: float
    gpu_reserved_gb: float
    gpu_free_gb: float
    uptime_seconds: float
    version: str = "2.0.0"


class GPUStats(BaseModel):
    index: int
    name: str
    allocated_gb: float
    reserved_gb: float
    total_gb: float
    free_gb: float
    utilization_pct: Optional[float] = None


class MetricsResponse(BaseModel):
    total_requests: int
    total_tokens_generated: int
    total_prompt_tokens: int
    avg_latency_seconds: float
    avg_tokens_per_second: float
    p50_latency_seconds: float
    p95_latency_seconds: float
    p99_latency_seconds: float
    requests_rejected_oom: int
    requests_rejected_quota: int
    uptime_seconds: float
    gpus: List[GPUStats]


# ---------------------------------------------------------------------------
# Admission control
# ---------------------------------------------------------------------------

class AdmissionVerdict(BaseModel):
    """Internal result from the memory-budget router."""
    allowed: bool
    backend: Literal["vllm", "hf_offload", "queue", "reject"]
    reason: str
    estimated_kv_gb: float
    free_gpu_gb: float
