// Turns the /api/ai/status payload into the upload-page banner. Works for Ollama (local) and Anthropic (Render);
// a payload without `provider` comes from an older backend and is treated as Ollama.
const READY_DETAIL = "AI Insight, Copilot and Report Builder are ready";

export function getAiProviderLabel(status) {
  if (!status || status.loading) return "AI Provider";
  if (status.provider === "anthropic") return "Anthropic API";
  if (status.provider === "ollama" || !status.provider) return "Local AI · Ollama";
  return "AI Provider";
}

export function describeAiStatus(status) {
  if (!status || status.loading) {
    return { tone: "loading", headline: "Checking AI provider...", detail: "Dataset upload remains available without AI" };
  }
  const model = status.model || "";
  const ready = Boolean(status.connected && status.model_available);
  if (status.status === "misconfigured") {
    return {
      tone: "error",
      headline: "AI provider misconfigured",
      detail: status.message || "Check the AI_PROVIDER setting. Dataset upload remains available without AI",
    };
  }
  if (status.provider === "anthropic") {
    if (ready) return { tone: "ready", headline: `Anthropic API Ready - ${model}`, detail: READY_DETAIL };
    const offline = "Dataset upload remains available without AI";
    switch (status.status) {
      case "missing_credentials":
        return { tone: "error", headline: "Anthropic API key not configured on the server", detail: offline };
      case "invalid_credentials":
        return { tone: "error", headline: "Anthropic API key was rejected - check ANTHROPIC_API_KEY", detail: offline };
      case "model_unavailable":
        return { tone: "warning", headline: `Anthropic API connected - Model unavailable: ${model}`, detail: offline };
      default:
        return { tone: "error", headline: "Anthropic API unreachable - try again shortly", detail: offline };
    }
  }
  const offline = "Dataset upload remains available without local AI";
  if (ready) return { tone: "ready", headline: `Local AI Ready - ${model}`, detail: READY_DETAIL };
  if (status.connected) return { tone: "warning", headline: `Ollama running - Model missing: ${model}`, detail: offline };
  return { tone: "error", headline: "Ollama disconnected - Start Ollama", detail: offline };
}

export const TONE_DOT_CLASS = {
  loading: "bg-ink-500",
  ready: "bg-emerald-400",
  warning: "bg-amber-400",
  error: "bg-red-400",
};
