"""Anthropic Messages API client that speaks the same dialect as ollama_client.

`anthropic_chat()` accepts Ollama-style arguments (messages incl. role "tool", Ollama tool definitions, json_mode) and
returns an Ollama-shaped response: {"message": {"role", "content", "tool_calls": [{"function": {"name", "arguments"}}]},
"done_reason", "prompt_eval_count", "eval_count"}. Callers therefore need no provider-specific code.

Claude Haiku 5.5 rules honoured here (Anthropic migration guide): no temperature/top_p/top_k, no assistant prefill (the request
always ends with a user turn), adaptive thinking with `output_config.effort`, thinking tokens count toward max_tokens
(headroom added), content blocks are selected by type, and thinking blocks are passed back unmodified in tool loops.
The API key is read from ANTHROPIC_API_KEY only; it is never logged, returned or placed in an error message.
"""
import copy
import json
import os

import httpx

from app.engines.ollama_client import OllamaError, OllamaTimeout

DEFAULT_MODEL = "claude-haiku-5-5"
DEFAULT_BASE_URL = "https://api.anthropic.com"
DEFAULT_TIMEOUT = 120.0
DEFAULT_EFFORT = "low"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096
THINKING_HEADROOM = 2048          # thinking tokens count toward max_tokens
MAX_TOKENS_CAP = 32000
JSON_INSTRUCTION = ("Respond with a single valid JSON object and nothing else: no markdown code fences, no text before or "
                    "after the JSON.")
MISSING_RESULT = "[no result was recorded for this tool call]"

_TRANSPORT = None                 # test seam: an httpx.MockTransport


class AnthropicError(OllamaError):
    """Any Anthropic failure. Subclasses OllamaError so every existing `except OllamaError` handler keeps working."""


class AnthropicConfigError(AnthropicError):
    pass


class AnthropicTimeout(OllamaTimeout, AnthropicError):
    pass


# ---------------------------------------------------------------- configuration (read at call time)
def base_url() -> str:
    return (os.getenv("ANTHROPIC_BASE_URL") or DEFAULT_BASE_URL).strip().rstrip("/")


def model() -> str:
    return (os.getenv("ANTHROPIC_MODEL") or DEFAULT_MODEL).strip()


def timeout() -> float:
    try:
        value = float(os.getenv("ANTHROPIC_TIMEOUT") or DEFAULT_TIMEOUT)
    except ValueError:
        return DEFAULT_TIMEOUT
    return value if value > 0 else DEFAULT_TIMEOUT


def effort() -> str:
    value = (os.getenv("ANTHROPIC_EFFORT") or DEFAULT_EFFORT).strip().lower()
    if value not in EFFORTS:
        raise AnthropicConfigError(f"ANTHROPIC_EFFORT must be one of {', '.join(EFFORTS)} (got {value!r}).")
    return value


def credentials_configured() -> bool:
    return bool((os.getenv("ANTHROPIC_API_KEY") or "").strip())


def _api_key() -> str:
    key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
    if not key:
        raise AnthropicConfigError("ANTHROPIC_API_KEY is not set on the server. Add it as a secret environment variable "
                                   "(it is never read from files or returned by the API).")
    return key


def _headers() -> dict:
    return {"x-api-key": _api_key(), "anthropic-version": ANTHROPIC_VERSION, "content-type": "application/json"}


# ---------------------------------------------------------------- request building (pure)
def _text_of(content) -> str:
    if content is None:
        return ""
    return content if isinstance(content, str) else json.dumps(content, default=str)


def convert_tools(tools) -> list[dict]:
    converted = []
    for tool in tools or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict) or not function.get("name"):
            continue
        schema = dict(function["parameters"]) if isinstance(function.get("parameters"), dict) else {}
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        converted.append({"name": function["name"], "description": function.get("description") or function["name"],
                          "input_schema": schema})
    return converted


def convert_messages(messages) -> tuple[str, list[dict]]:
    """Ollama-style messages -> (system string, Anthropic messages). Guarantees: first turn is user, last turn is user, roles
    alternate (adjacent same-role messages are merged), every tool_use has a tool_result in the next user message and
    tool_result blocks come first in that message."""
    system_parts: list[str] = []
    out: list[dict] = []
    pending: list[str] = []                     # tool_use ids still waiting for their result (FIFO)
    counter = 0

    def push(role: str, blocks: list[dict]):
        if not blocks:
            return
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": list(blocks)})

    def close_pending():
        if pending:
            push("user", [{"type": "tool_result", "tool_use_id": tool_id, "content": MISSING_RESULT} for tool_id in pending])
            pending.clear()

    for message in messages:
        role = message.get("role")
        text = _text_of(message.get("content"))
        if role == "system":
            if text.strip():
                system_parts.append(text)
        elif role == "assistant":
            close_pending()
            raw = message.get("anthropic_content")
            if isinstance(raw, list) and raw:                  # exact blocks from our own earlier response (thinking incl.)
                blocks = copy.deepcopy(raw)
                pending.extend(b["id"] for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id"))
            else:
                blocks = [{"type": "text", "text": text}] if text.strip() else []
                for call in message.get("tool_calls") or []:
                    function = call.get("function") or {}
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except ValueError:
                            arguments = {}
                    counter += 1
                    tool_id = f"toolu_sg{counter:04d}"
                    blocks.append({"type": "tool_use", "id": tool_id, "name": function.get("name") or "unknown",
                                   "input": arguments if isinstance(arguments, dict) else {}})
                    pending.append(tool_id)
            push("assistant", blocks)
        elif role == "tool":
            if pending:
                push("user", [{"type": "tool_result", "tool_use_id": pending.pop(0), "content": text or "(empty)"}])
            elif text.strip():
                push("user", [{"type": "text", "text": f"Tool result: {text}"}])
        else:                                                   # user (or an unknown role, treated as user)
            close_pending()
            if text.strip():
                push("user", [{"type": "text", "text": text}])
    close_pending()
    if not out:
        out.append({"role": "user", "content": [{"type": "text", "text": "Continue."}]})
    if out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": [{"type": "text", "text": "(Conversation resumed.)"}]})
    if out[-1]["role"] != "user":                               # prefill is rejected by Claude Haiku 5.5
        out.append({"role": "user", "content": [{"type": "text", "text": "Continue."}]})
    return "\n\n".join(system_parts), out


def effective_max_tokens(requested) -> int:
    try:
        requested = int(requested) if requested else 0
    except (TypeError, ValueError):
        requested = 0
    return min(requested + THINKING_HEADROOM, MAX_TOKENS_CAP) if requested > 0 else DEFAULT_MAX_TOKENS


def build_request(messages, *, tools=None, json_mode: bool = False, max_tokens=None) -> dict:
    """The JSON body for POST /v1/messages. Deliberately has NO temperature/top_p/top_k and never ends on an assistant turn."""
    system, converted = convert_messages(messages)
    if json_mode:
        system = f"{system}\n\n{JSON_INSTRUCTION}".strip()
    body = {"model": model(), "max_tokens": effective_max_tokens(max_tokens), "messages": converted,
            "output_config": {"effort": effort()}}
    if system:
        body["system"] = system
    anthropic_tools = convert_tools(tools)
    if anthropic_tools:
        body["tools"] = anthropic_tools
    return body


# ---------------------------------------------------------------- response handling (pure)
def extract_json_text(text: str) -> str:
    """Models sometimes wrap JSON in code fences or a sentence. Return the outermost {...} text; unchanged if none is found."""
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start, end = stripped.find("{"), stripped.rfind("}")
    return stripped[start:end + 1] if 0 <= start < end else text


def normalize_response(data, *, json_mode: bool = False) -> dict:
    if not isinstance(data, dict) or not isinstance(data.get("content"), list):
        raise AnthropicError("Anthropic returned an unexpected response (no content list).")
    stop_reason = data.get("stop_reason")
    if stop_reason == "refusal":
        raise AnthropicError("The model declined this request (safety refusal). Rephrase the question or ask something else.")
    texts, tool_calls = [], []
    for block in data["content"]:                                # select by type, never by position
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            texts.append(block.get("text") or "")
        elif block.get("type") == "tool_use":
            tool_input = block.get("input")
            tool_calls.append({"function": {"name": block.get("name"), "arguments": tool_input if isinstance(tool_input, dict) else {}}})
    text = "".join(texts)
    if json_mode and not tool_calls:
        text = extract_json_text(text)
    if not text.strip() and not tool_calls and stop_reason == "max_tokens":
        raise AnthropicError("The model used its whole output budget before answering (thinking tokens count toward "
                             "max_tokens). Lower ANTHROPIC_EFFORT or ask a narrower question.")
    message = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if any(isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking", "tool_use") for b in data["content"]):
        message["anthropic_content"] = copy.deepcopy(data["content"])   # replayed unmodified in tool loops
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    return {"model": data.get("model"), "message": message, "done": True,
            "done_reason": "length" if stop_reason == "max_tokens" else "stop",
            "prompt_eval_count": usage.get("input_tokens"), "eval_count": usage.get("output_tokens")}


def _error_detail(response: httpx.Response) -> str:
    try:
        detail = response.json()["error"]["message"]
    except Exception:
        detail = ""
    return " ".join(str(detail).split())[:300]


def raise_for_status(response: httpx.Response) -> None:
    code = response.status_code
    if code < 400:
        return
    detail = _error_detail(response)
    suffix = f": {detail}" if detail else ""
    if code == 401:
        raise AnthropicError("Anthropic API authentication failed (HTTP 401). Check ANTHROPIC_API_KEY.")
    if code == 403:
        raise AnthropicError(f"Anthropic API permission denied (HTTP 403){suffix}")
    if code == 404:
        raise AnthropicError(f"Anthropic model or endpoint not found (HTTP 404). Check ANTHROPIC_MODEL ({model()}){suffix}")
    if code == 429:
        retry = response.headers.get("retry-after")
        raise AnthropicError("Anthropic API rate limit reached (HTTP 429)" + (f"; retry after {retry}s" if retry else "") + ".")
    if code >= 500:
        raise AnthropicError(f"Anthropic API is temporarily unavailable or overloaded (HTTP {code}). Try again shortly.")
    raise AnthropicError(f"Anthropic rejected the request (HTTP {code}){suffix}")


# ---------------------------------------------------------------- network
def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout(), transport=_TRANSPORT)


async def anthropic_chat(messages, *, tools=None, json_mode: bool = False, temperature=None, max_tokens=None) -> dict:
    """Drop-in for ollama_chat. `temperature` is accepted for signature compatibility and deliberately NOT sent
    (Claude Haiku 5.5 rejects non-default sampling parameters)."""
    headers = _headers()                                         # raises AnthropicConfigError before any network use
    body = build_request(messages, tools=tools, json_mode=json_mode, max_tokens=max_tokens)
    try:
        async with _client() as client:
            response = await client.post(f"{base_url()}/v1/messages", headers=headers, json=body)
    except httpx.TimeoutException as exc:
        raise AnthropicTimeout(f"Anthropic request timed out after {timeout():g} seconds.") from exc
    except httpx.RequestError as exc:
        raise AnthropicError(f"Cannot connect to the Anthropic API ({type(exc).__name__}). Check the network.") from exc
    raise_for_status(response)
    try:
        data = response.json()
    except ValueError as exc:
        raise AnthropicError("Anthropic returned a response that is not valid JSON.") from exc
    return normalize_response(data, json_mode=json_mode)


async def check_model() -> tuple[str, str]:
    """Free credentials/model check (GET /v1/models/{model}). Returns (status, message); never raises, never includes the key."""
    try:
        headers = _headers()
    except AnthropicConfigError as exc:
        return "missing_credentials", str(exc)
    try:
        async with _client() as client:
            response = await client.get(f"{base_url()}/v1/models/{model()}", headers=headers)
    except httpx.TimeoutException:
        return "disconnected", "The Anthropic API did not answer in time."
    except httpx.RequestError:
        return "disconnected", "Cannot connect to the Anthropic API."
    if response.status_code < 300:
        return "ready", ""
    if response.status_code in (401, 403):
        return "invalid_credentials", "The Anthropic API rejected the configured key."
    if response.status_code == 404:
        return "model_unavailable", f"Model {model()} was not found for this account."
    return "unavailable", f"The Anthropic API answered HTTP {response.status_code}."
