import { useState } from "react";
import ConfidenceMeter from "./ConfidenceMeter";
import LoadingState from "./ui/LoadingState";
import ErrorState from "./ui/ErrorState";

const STEP_LABELS = {
  duplicate_manager: "Duplicate Manager",
  missing_value_center: "Missing Value Center",
  outlier_center: "Outlier Center",
  category_merge_center: "Category Merge Center",
  text_cleaning: "Text Cleaning",
  ratio: "Ratio Feature",
  date_features: "Date Features",
  bucketing: "Bucketing",
  scaling_normalization: "Scaling / Normalization",
  encoding: "Encoding",
  log_transform: "Log Transform",
};

function BeforeAfterPreview({ preview }) {
  const hasSamples = preview.before_sample !== undefined || preview.after_sample !== undefined;
  if (!hasSamples) return null;
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
      <div>
        <p className="font-mono text-[10px] uppercase tracking-wider text-ink-500">Before</p>
        <p className="mt-0.5 font-mono text-xs text-ink-300">
          {preview.before_sample?.length ? preview.before_sample.join(", ") : "—"}
        </p>
      </div>
      <div>
        <p className="font-mono text-[10px] uppercase tracking-wider text-ink-500">After</p>
        <p className="mt-0.5 font-mono text-xs text-signal-green">
          {preview.after_sample?.length ? preview.after_sample.join(", ") : "(removed)"}
        </p>
      </div>
    </div>
  );
}

function CompareMethodsPanel({ column, onCompareMethods, onUseMethod }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [open, setOpen] = useState(false);

  const handleToggle = async () => {
    if (!open && !data) {
      setLoading(true);
      setError(null);
      try {
        const result = await onCompareMethods(column);
        setData(result);
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    }
    setOpen((v) => !v);
  };

  return (
    <div className="mt-3">
      <button
        onClick={handleToggle}
        className="font-mono text-[11px] uppercase tracking-wider text-ink-500 hover:text-ink-300"
      >
        {open ? "− Hide" : "+ Compare methods"}
      </button>
      {open && (
        <div className="mt-2 rounded border border-graphite-border bg-graphite-900 p-3">
          {loading && <LoadingState compact />}
          {error && <ErrorState message={error} compact />}
          {data && (
            <div className="space-y-2">
              {data.methods.map((m) => (
                <div key={m.method} className="flex items-center justify-between rounded border border-graphite-border bg-graphite-800 px-2.5 py-2">
                  <div>
                    <p className="font-mono text-xs text-ink-100">
                      {m.method}{m.fill_value != null ? ` → ${m.fill_value}` : ""}
                    </p>
                    <p className="mt-0.5 font-mono text-[11px] text-ink-500">
                      {Object.entries(m.after_stats).map(([k, v]) => `${k}: ${typeof v === "object" ? JSON.stringify(v) : v}`).join(" · ")}
                    </p>
                  </div>
                  <button
                    onClick={() => onUseMethod(m.method)}
                    className="whitespace-nowrap rounded border border-graphite-border px-2 py-1 font-mono text-[11px] text-ink-300 hover:bg-graphite-700"
                  >
                    Use this
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function FuzzyMergeControls({ recommendation, onConfirm, setStatus }) {
  const [renaming, setRenaming] = useState(false);
  const [customName, setCustomName] = useState(recommendation.preview.suggested_canonical);

  const doMerge = async (canonical) => {
    setStatus("applying");
    try {
      await onConfirm(recommendation, {
        action: "fuzzy_merge",
        params: { values: recommendation.preview.values, canonical },
      });
      setStatus("applied");
    } catch (e) {
      setStatus("pending");
    }
  };

  return (
    <div className="mt-3">
      <div className="flex flex-wrap items-center gap-1.5">
        {recommendation.preview.values.map((v) => (
          <span key={v} className="rounded border border-graphite-border bg-graphite-900 px-2 py-0.5 font-mono text-xs text-ink-300">
            {v}
          </span>
        ))}
      </div>
      <p className="mt-2 font-mono text-[11px] text-ink-500">
        Suggested canonical: <span className="text-signal-amber">{recommendation.preview.suggested_canonical}</span>
        {" · "}similarity {Math.round(recommendation.preview.similarity * 100)}%
      </p>

      {renaming ? (
        <div className="mt-2 flex items-center gap-2">
          <input
            value={customName}
            onChange={(e) => setCustomName(e.target.value)}
            className="rounded border border-graphite-border bg-graphite-900 px-2 py-1 font-mono text-xs text-ink-100 focus:border-signal-amber focus:outline-none"
          />
          <button
            onClick={() => doMerge(customName)}
            className="rounded bg-signal-amber px-3 py-1 font-mono text-xs font-medium text-graphite-950 hover:bg-signal-amber/90"
          >
            Confirm rename
          </button>
          <button onClick={() => setRenaming(false)} className="font-mono text-xs text-ink-500 hover:text-ink-300">
            Cancel
          </button>
        </div>
      ) : (
        <div className="mt-2 flex gap-2">
          <button
            onClick={() => doMerge(recommendation.preview.suggested_canonical)}
            className="rounded bg-signal-amber px-3 py-1.5 font-mono text-xs font-medium text-graphite-950 hover:bg-signal-amber/90"
          >
            Merge as "{recommendation.preview.suggested_canonical}"
          </button>
          <button
            onClick={() => setRenaming(true)}
            className="rounded border border-graphite-border px-3 py-1.5 font-mono text-xs text-ink-300 hover:bg-graphite-700"
          >
            Rename…
          </button>
        </div>
      )}
    </div>
  );
}

/**
 * Reusable across every cleaning module (missing values, duplicates, outliers,
 * category merge, text cleaning, ...). This is the one interaction pattern
 * the whole app is built around: the AI proposes, the user decides.
 *
 * Nothing in `onConfirm` runs until the user explicitly clicks Apply/Merge —
 * this component never calls the mutation endpoint on its own.
 */
export default function RecommendationCard({ recommendation, onConfirm, onDismiss, onCompareMethods }) {
  const [status, setStatus] = useState("pending"); // pending | applying | applied | dismissed | error
  const [error, setError] = useState(null);
  const [showExplain, setShowExplain] = useState(false);

  const isFuzzyMerge = recommendation.step === "category_merge_center" && recommendation.recommended_action === "fuzzy_merge";
  const isMissingValue = recommendation.step === "missing_value_center";

  const handleApply = async (overrides = {}) => {
    setStatus("applying");
    setError(null);
    try {
      await onConfirm(recommendation, overrides);
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

  const hasExplainability = recommendation.how_detected || recommendation.risks?.length > 0 || recommendation.alternative_actions?.length > 0;
  const hasOtherPreviewFields = Object.keys(recommendation.preview || {}).some(
    (k) => !["before_sample", "after_sample", "values", "suggested_canonical", "similarity"].includes(k)
  );

  return (
    <div className={`rounded-lg border p-4 ${recommendation.goal_relevant ? "border-signal-amber/50 bg-graphite-800" : "border-graphite-border bg-graphite-800"}`}>
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="font-mono text-[11px] uppercase tracking-wider text-signal-amber">
              {STEP_LABELS[recommendation.step] || recommendation.step}
              {recommendation.column ? ` · ${recommendation.column}` : ""}
            </span>
            {recommendation.goal_relevant && (
              <span className="rounded-full bg-signal-amber/15 px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider text-signal-amber">
                Goal relevant
              </span>
            )}
          </div>
          <p className="mt-1 font-display text-sm font-medium text-ink-100">
            {recommendation.issue}
          </p>
        </div>
        <ConfidenceMeter confidence={recommendation.confidence} />
      </div>

      <p className="mt-3 text-sm leading-relaxed text-ink-300">{recommendation.reason}</p>

      {recommendation.preview?.estimated_impact && (
        <p className="mt-2 rounded border border-graphite-border bg-graphite-900 px-3 py-2 text-xs text-ink-300">
          {recommendation.preview.estimated_impact}
        </p>
      )}

      {!isFuzzyMerge && <div className="mt-3"><BeforeAfterPreview preview={recommendation.preview || {}} /></div>}

      {isFuzzyMerge && (
        <FuzzyMergeControls recommendation={recommendation} onConfirm={onConfirm} setStatus={setStatus} />
      )}

      {isMissingValue && onCompareMethods && (
        <CompareMethodsPanel
          column={recommendation.column}
          onCompareMethods={onCompareMethods}
          onUseMethod={(method) => handleApply({ action: method })}
        />
      )}

      {hasOtherPreviewFields && (
        <div className="mt-3 rounded border border-graphite-border bg-graphite-900 p-3">
          <p className="mb-1 font-mono text-[11px] uppercase tracking-wider text-ink-500">Details</p>
          <pre className="overflow-x-auto font-mono text-xs text-ink-300">
            {JSON.stringify(recommendation.preview, null, 2)}
          </pre>
        </div>
      )}

      {hasExplainability && (
        <div className="mt-3">
          <button
            onClick={() => setShowExplain((v) => !v)}
            className="font-mono text-[11px] uppercase tracking-wider text-ink-500 hover:text-ink-300"
          >
            {showExplain ? "− Hide" : "+ Why & risks"}
          </button>
          {showExplain && (
            <div className="mt-2 space-y-2 rounded border border-graphite-border bg-graphite-900 p-3 text-xs">
              {recommendation.how_detected && (
                <div>
                  <p className="font-mono uppercase tracking-wider text-ink-500">How this was detected</p>
                  <p className="mt-0.5 text-ink-300">{recommendation.how_detected}</p>
                </div>
              )}
              {recommendation.risks?.length > 0 && (
                <div>
                  <p className="font-mono uppercase tracking-wider text-ink-500">Risks if applied</p>
                  <ul className="mt-0.5 list-inside list-disc text-ink-300">
                    {recommendation.risks.map((r, i) => <li key={i}>{r}</li>)}
                  </ul>
                </div>
              )}
              {recommendation.alternative_actions?.length > 0 && (
                <div>
                  <p className="font-mono uppercase tracking-wider text-ink-500">Alternative actions</p>
                  <p className="mt-0.5 text-ink-300">{recommendation.alternative_actions.join(", ")}</p>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
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
              {isFuzzyMerge ? "Ignore" : "Dismiss"}
            </button>
            {!isFuzzyMerge && (
              <button
                onClick={() => handleApply()}
                disabled={status === "applying"}
                className="rounded bg-signal-amber px-3 py-1.5 text-xs font-medium text-graphite-950 hover:bg-signal-amber/90 disabled:opacity-50"
              >
                {status === "applying" ? "Applying…" : "Apply"}
              </button>
            )}
          </div>
        )}
      </div>

      {error && <div className="mt-2"><ErrorState message={error} compact /></div>}
    </div>
  );
}
