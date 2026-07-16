import ConfidenceMeter from "../ConfidenceMeter";

export default function AnalysisRecommendationCard({ rec, onRun, isSelected }) {
  if (!rec.recommended) {
    return (
      <div className="rounded-lg border border-graphite-border bg-graphite-900 p-3 opacity-70">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="font-mono text-sm text-ink-500">{rec.title}</p>
          <span className="font-mono text-[10px] uppercase tracking-wider text-ink-700">Not recommended</span>
        </div>
        <p className="mt-1 font-mono text-xs text-ink-700">{rec.not_recommended_because}</p>
      </div>
    );
  }

  return (
    <div className={`rounded-lg border p-4 ${isSelected ? "border-signal-amber" : "border-graphite-border"} bg-graphite-800`}>
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="font-display text-sm text-ink-100">{rec.title}</p>
          <p className="mt-1 text-xs text-ink-300">{rec.reason}</p>
        </div>
        <ConfidenceMeter confidence={rec.confidence} />
      </div>

      <p className="mt-2 font-mono text-[11px] text-ink-700">{rec.required_characteristics}</p>

      {rec.risks.length > 0 && (
        <ul className="mt-2 list-inside list-disc space-y-0.5 font-mono text-[11px] text-signal-amber/80">
          {rec.risks.map((r, i) => <li key={i}>{r}</li>)}
        </ul>
      )}

      <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
        <span className="font-mono text-[11px] text-ink-700">{rec.how_detected}</span>
        <button
          onClick={() => onRun(rec)}
          className="rounded border border-signal-amber/50 bg-signal-amber/10 px-3 py-1 font-mono text-xs text-signal-amber transition-colors hover:bg-signal-amber/20"
        >
          {isSelected ? "Selected" : "Run this analysis"}
        </button>
      </div>
    </div>
  );
}
