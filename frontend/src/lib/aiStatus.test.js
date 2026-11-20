import test from "node:test";
import assert from "node:assert/strict";
import { describeAiStatus, getAiProviderLabel, TONE_DOT_CLASS } from "./aiStatus.js";

const anthropic = (extra) => ({ provider: "anthropic", connected: false, model: "claude-haiku-5-5", model_available: false, ...extra });

test("loading and empty payloads show a neutral checking banner", () => {
  assert.equal(describeAiStatus({ loading: true }).tone, "loading");
  assert.equal(describeAiStatus(null).tone, "loading");
});

test("Ollama wording is unchanged (ready, model missing, disconnected)", () => {
  const ready = describeAiStatus({ provider: "ollama", connected: true, model_available: true, model: "qwen2.5:3b" });
  assert.deepEqual([ready.tone, ready.headline], ["ready", "Local AI Ready - qwen2.5:3b"]);
  const missing = describeAiStatus({ provider: "ollama", connected: true, model_available: false, model: "qwen2.5:3b" });
  assert.deepEqual([missing.tone, missing.headline], ["warning", "Ollama running - Model missing: qwen2.5:3b"]);
  const down = describeAiStatus({ provider: "ollama", connected: false, model: "qwen2.5:3b" });
  assert.deepEqual([down.tone, down.headline, down.detail], ["error", "Ollama disconnected - Start Ollama", "Dataset upload remains available without local AI"]);
});

test("a payload without provider (older backend) is treated as Ollama", () => {
  assert.equal(describeAiStatus({ connected: true, model_available: true, model: "m" }).headline, "Local AI Ready - m");
  assert.equal(describeAiStatus({ connected: false, model: "m" }).headline, "Ollama disconnected - Start Ollama");
});

test("Anthropic states name the provider and the model", () => {
  assert.equal(describeAiStatus(anthropic({ connected: true, model_available: true, status: "ready" })).headline, "Anthropic API Ready - claude-haiku-5-5");
  assert.equal(describeAiStatus(anthropic({ status: "missing_credentials" })).headline, "Anthropic API key not configured on the server");
  assert.equal(describeAiStatus(anthropic({ status: "invalid_credentials" })).tone, "error");
  const unavailable = describeAiStatus(anthropic({ connected: true, status: "model_unavailable" }));
  assert.deepEqual([unavailable.tone, unavailable.headline], ["warning", "Anthropic API connected - Model unavailable: claude-haiku-5-5"]);
  assert.equal(describeAiStatus(anthropic({ status: "disconnected" })).headline, "Anthropic API unreachable - try again shortly");
  assert.equal(describeAiStatus(anthropic({ status: "unavailable" })).tone, "error");
});

test("a misconfigured provider shows the server's message", () => {
  const out = describeAiStatus({ provider: "nonsense", connected: false, status: "misconfigured", message: "AI_PROVIDER must be one of auto, ollama, anthropic" });
  assert.deepEqual([out.tone, out.headline, out.detail], ["error", "AI provider misconfigured", "AI_PROVIDER must be one of auto, ollama, anthropic"]);
});

test("every tone has a dot colour and no banner text can contain a key", () => {
  for (const tone of ["loading", "ready", "warning", "error"]) assert.ok(TONE_DOT_CLASS[tone]);
  const out = describeAiStatus(anthropic({ status: "invalid_credentials", credentials_configured: true }));
  assert.ok(!/sk-ant/i.test(JSON.stringify(out)));
});

test("provider label reflects Anthropic, Ollama, legacy and unknown status", () => {
  assert.equal(getAiProviderLabel({ provider: "anthropic" }), "Anthropic API");
  assert.equal(getAiProviderLabel({ provider: "ollama" }), "Local AI \u00b7 Ollama");
  assert.equal(getAiProviderLabel({}), "Local AI \u00b7 Ollama");
  assert.equal(getAiProviderLabel({ provider: "unknown" }), "AI Provider");
  assert.equal(getAiProviderLabel({ loading: true }), "AI Provider");
  assert.equal(getAiProviderLabel(null), "AI Provider");
});