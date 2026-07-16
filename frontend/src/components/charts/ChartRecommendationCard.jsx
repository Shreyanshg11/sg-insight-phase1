import { useState } from "react";
import ConfidenceMeter from "../ConfidenceMeter";
import ChartRenderer from "./ChartRenderer";
import { explainChart, pinChart } from "../../api/client";
import ErrorState from "../ui/ErrorState";

export default function ChartRecommendationCard({ datasetId, rec }) {
  const [explanation, setExplanation] = useState(null);
  const [explaining, setExplaining] = useState(false);
  const [pinState, setPinState] = useState("idle"); // idle | pinning | pinned | error
  const [pinError, setPinError] = useState(null);

  const handleExplain = async () => {
    if (explanation) { setExplanation(null); return; } // toggle off
    setExplaining(true);
    try {
      const res = await explainChart({ dataset_id: datasetId, chart_type: rec.chart_type, columns: rec.columns });
      setExplanation(res.explanation);
    } catch (e) {
      setExplanation(`Couldn't generate an explanation: ${e.message}`);
    } finally {
      setExplaining(false);
    }
  };

  const handlePin = async () => {
    setPinState("pinning");
    setPinError(null);
    try {
      await pinChart({ dataset_id: datasetId, chart_type: rec.chart_type, columns: rec.columns, title: rec.title });
      setPinState("pinned");
    } catch (e) {
      setPinState("error");
      setPinError(e.message);
    }
  };

  return (
    <div className="rounded-lg border border-graphite-border bg-graphite-800 p-4">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="font-display text-sm text-ink-100">{rec.title}</p>
          <p className="mt-1 text-xs text-ink-300">{rec.reason}</p>
        </div>
        <ConfidenceMeter confidence={rec.confidence} />
      </div>

      <div className="mt-3">
        <ChartRenderer chartType={rec.chart_type} data={rec.data} trend={rec.trend} />
      </div>

      {rec.alternative_chart_types.length > 0 && (
        <p className="mt-2 font-mono text-[11px] text-ink-700">
          Alternatives considered: {rec.alternative_chart_types.join(" · ")}
        </p>
      )}

      {explanation && (
        <p className="mt-2 rounded border border-graphite-border bg-graphite-900 p-2 text-xs text-ink-300">
          {explanation}
        </p>
      )}

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <button
          onClick={handleExplain}
          disabled={explaining}
          className="rounded border border-graphite-border bg-graphite-900 px-3 py-1 font-mono text-xs text-ink-300 transition-colors hover:border-signal-amber hover:text-signal-amber disabled:opacity-50"
        >
          {explaining ? "Explaining…" : explanation ? "Hide explanation" : "Explain this chart"}
        </button>
        <button
          onClick={handlePin}
          disabled={pinState === "pinning" || pinState === "pinned"}
          className="rounded border border-signal-amber/50 bg-signal-amber/10 px-3 py-1 font-mono text-xs text-signal-amber transition-colors hover:bg-signal-amber/20 disabled:opacity-60"
        >
          {pinState === "pinning" ? "Pinning…" : pinState === "pinned" ? "Pinned ✓" : "Pin to Dashboard"}
        </button>
      </div>
      {pinError && <div className="mt-2"><ErrorState message={pinError} onRetry={handlePin} compact /></div>}
    </div>
  );
}
