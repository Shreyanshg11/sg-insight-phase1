import { useState, useEffect, useRef } from "react";

const STEPS = [
  { stage: "profile", label: "Profile", text: "Every upload starts here — column types, roles, and missing-value counts, inferred automatically." },
  { stage: "quality", label: "Quality", text: "A single data quality score, broken down by completeness, consistency, validity, and uniqueness." },
  { stage: "cleaning", label: "Cleaning", text: "Every cleaning suggestion comes with a reason and a confidence reading — nothing applies until you confirm it." },
  { stage: "feature_engineering", label: "Feature Engineering", text: "Ratios, date parts, bucketing, scaling, encoding — proposed the same confirm-first way as cleaning." },
  { stage: "eda", label: "EDA", text: "Distributions, correlations, missing patterns, and relationships — all filterable by the global filter bar." },
  { stage: "analysis", label: "Analysis", text: "Regression, clustering, hypothesis tests, forecasting — recommended based on what your data can actually support." },
  { stage: "visualization", label: "Visualization", text: "Recommended charts you can pin straight to a dashboard, or drill into by clicking a bar or box-plot group." },
  { stage: "dashboard", label: "Dashboard", text: "Every chart you've pinned, always live off the current dataset version." },
  { stage: "report", label: "Report", text: "The Smart Report Builder — AI-drafted outline, drag to reorder, live preview, then generate as Markdown, HTML, PDF, or a real Jupyter notebook." },
];

const STEP_MS = 5000;

export default function TourGuide({ onSelectStage, onClose }) {
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(true);
  const timerRef = useRef(null);

  useEffect(() => { onSelectStage(STEPS[index].stage); }, [index, onSelectStage]);

  useEffect(() => {
    if (!playing) return;
    timerRef.current = setTimeout(() => {
      setIndex((i) => (i + 1 < STEPS.length ? i + 1 : i));
      if (index + 1 >= STEPS.length) setPlaying(false);
    }, STEP_MS);
    return () => clearTimeout(timerRef.current);
  }, [playing, index]);

  const step = STEPS[index];

  return (
    <div className="fixed inset-0 z-[60] bg-graphite-950/40" onClick={(e) => e.stopPropagation()}>
      <div className="fixed bottom-6 left-1/2 w-[92%] max-w-lg -translate-x-1/2 rounded-lg border border-signal-amber/40 bg-graphite-900 p-4 shadow-xl">
        <div className="flex items-center justify-between">
          <p className="font-mono text-[11px] uppercase tracking-wider text-signal-amber">
            Tour — {step.label} ({index + 1}/{STEPS.length})
          </p>
          <button onClick={onClose} className="font-mono text-ink-500 hover:text-ink-100" aria-label="End tour">✕</button>
        </div>
        <p className="mt-2 text-sm text-ink-100">{step.text}</p>
        <div className="mt-3 flex items-center gap-2">
          <button
            onClick={() => setIndex((i) => Math.max(0, i - 1))}
            disabled={index === 0}
            className="rounded border border-graphite-border px-2 py-1 font-mono text-xs text-ink-300 hover:bg-graphite-700 disabled:opacity-40"
          >
            ← Prev
          </button>
          <button
            onClick={() => setPlaying((p) => !p)}
            className="rounded border border-signal-amber/50 bg-signal-amber/10 px-3 py-1 font-mono text-xs text-signal-amber hover:bg-signal-amber/20"
          >
            {playing ? "Pause" : "Play"}
          </button>
          <button
            onClick={() => setIndex((i) => Math.min(STEPS.length - 1, i + 1))}
            disabled={index === STEPS.length - 1}
            className="rounded border border-graphite-border px-2 py-1 font-mono text-xs text-ink-300 hover:bg-graphite-700 disabled:opacity-40"
          >
            Next →
          </button>
          <div className="ml-auto flex gap-1">
            {STEPS.map((_, i) => (
              <span key={i} className={`h-1.5 w-1.5 rounded-full ${i === index ? "bg-signal-amber" : "bg-graphite-700"}`} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
