// Translate the Ollama health payload into a concise upload-page banner.
const READY_DETAIL = "AI Insight, Copilot and Report Builder are ready";

export function getAiProviderLabel(status) {
  if (!status || status.loading) return "AI Provider";
  return status.provider === "ollama" || !status.provider ? "Local AI · Ollama" : "AI Provider";
}

export function describeAiStatus(status) {
  if (!status || status.loading) {
    return { tone: "loading", headline: "Checking local AI…", detail: "Dataset upload remains available without AI" };
  }
  const model = status.model || "";
  const offline = "Dataset upload remains available while local AI is offline";
  if (status.status === "misconfigured") {
    return { tone: "error", headline: "Ollama configuration needs attention", detail: status.message || offline };
  }
  if (status.connected && status.model_available) {
    return { tone: "ready", headline: `Local AI Ready - ${model}`, detail: READY_DETAIL };
  }
  if (status.connected) {
    return { tone: "warning", headline: `Ollama running - Model missing: ${model}`, detail: offline };
  }
  return { tone: "error", headline: "Ollama disconnected - Start Ollama", detail: offline };
}

export const TONE_DOT_CLASS = {
  loading: "bg-ink-500",
  ready: "bg-emerald-400",
  warning: "bg-amber-400",
  error: "bg-red-400",
};
