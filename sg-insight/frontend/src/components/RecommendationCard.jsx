import { useState } from "react";
import ConfidenceMeter from "./ConfidenceMeter";

const STEP_LABELS = {
  duplicate_manager: "Duplicate Manager",
  missing_value_center: "Missing Value Center",
  outlier_center: "Outlier Center",
  category_merge_center: "Category Merge Center",
  text_cleaning: "Text Cleaning",
};

/**
 * Reusable across every cleaning module (missing values, duplicates, outliers,
 * category merge, text cleaning, ...). This is the one interaction pattern
 * the whole app is built around: the AI proposes, the user decides.
 *
 * Nothing in `onConfirm` runs until the user explicitly clicks Apply —
 * this component never calls the mutation endpoint on its own.
 */
export default function RecommendationCard({ recommendation, onConfirm, onDismiss }) {
  const [status, setStatus] = useState("pending"); // pending | applying | applied | dismissed | error
  const [error, setError] = useState(null);

  const handleApply = async () => {
    setStatus("applying");
    setError(null);
    try {
      await onConfirm(recommendation);
      setStatus("applied");
    } catch (e) {
      setError(e.message);
      setStatus("pending");
    }
  };

  const handleDismiss = () => {
    setStatus("dismissed");
    onDismiss?.(recommendation);
  };

  return (
    <div className="rounded-lg border border-graphite-border bg-graphite-800 p-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <span className="font-mono text-[11px] uppercase tracking-wider text-signal-amber">
            {STEP_LABELS[recommendation.step] || recommendation.step}
            {recommendation.column ? ` · ${recommendation.column}` : ""}
          </span>
          <p className="mt-1 font-display text-sm font-medium text-ink-100">
            {recommendation.issue}
          </p>
        </div>
        <ConfidenceMeter confidence={recommendation.confidence} />
      </div>

      <p className="mt-3 text-sm leading-relaxed text-ink-300">{recommendation.reason}</p>

      <div className="mt-3 rounded border border-graphite-border bg-graphite-900 p-3">
        <p className="mb-1 font-mono text-[11px] uppercase tracking-wider text-ink-500">Preview</p>
        <pre className="overflow-x-auto font-mono text-xs text-ink-300">
          {JSON.stringify(recommendation.preview, null, 2)}
        </pre>
      </div>

      <div className="mt-3 flex items-center justify-between">
        <span className="font-mono text-xs text-ink-500">
          Recommended action: <span className="text-ink-100">{recommendation.recommended_action}</span>
        </span>

        {status === "applied" ? (
          <span className="font-mono text-xs text-signal-green">Applied ✓</span>
        ) : status === "dismissed" ? (
          <span className="font-mono text-xs text-ink-500">Dismissed</span>
        ) : (
          <div className="flex gap-2">
            <button
              onClick={handleDismiss}
              disabled={status === "applying"}
              className="rounded border border-graphite-border px-3 py-1.5 text-xs text-ink-300 hover:bg-graphite-700 disabled:opacity-50"
            >
              Dismiss
            </button>
            <button
              onClick={handleApply}
              disabled={status === "applying"}
              className="rounded bg-signal-amber px-3 py-1.5 text-xs font-medium text-graphite-950 hover:bg-signal-amber/90 disabled:opacity-50"
            >
              {status === "applying" ? "Applying…" : "Apply"}
            </button>
          </div>
        )}
      </div>

      {error && <p className="mt-2 font-mono text-xs text-signal-red">{error}</p>}
    </div>
  );
}
