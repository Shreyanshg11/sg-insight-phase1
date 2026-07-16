import ConfidenceMeter from "../ConfidenceMeter";

export default function AIInsightCard({ insight }) {
  return (
    <div className="rounded-lg border border-graphite-border bg-graphite-800 p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="font-mono text-[11px] uppercase tracking-wider text-ink-500">AI Insight ({insight.model})</p>
        <ConfidenceMeter confidence={insight.confidence} label="AI's stated confidence" />
      </div>

      {!insight.grounded && (
        <div className="mt-3 rounded border border-signal-red/50 bg-signal-red/10 p-3">
          <p className="font-mono text-xs font-semibold text-signal-red">
            ⚠ Unverified figures — the AI stated number(s) below that don't match the actual computed result:
          </p>
          <ul className="mt-1.5 list-inside list-disc space-y-0.5 font-mono text-[11px] text-signal-red/90">
            {insight.unverified_figures.map((f, i) => (
              <li key={i}><span className="font-semibold">{f.figure}</span> — {f.reason}</li>
            ))}
          </ul>
          <p className="mt-1.5 font-mono text-[11px] text-ink-500">
            Treat the text below with extra caution and cross-check it against the raw result.
          </p>
        </div>
      )}

      <div className="mt-3 space-y-3">
        <Field label="Observation" value={insight.observation} />
        <Field label="Evidence" value={insight.evidence} />
        <Field label="Business meaning" value={insight.business_meaning} />
        {insight.possible_explanations?.length > 0 && (
          <FieldList label="Possible explanations" items={insight.possible_explanations} />
        )}
        {insight.limitations?.length > 0 && (
          <FieldList label="Limitations" items={insight.limitations} tone="amber" />
        )}
      </div>
    </div>
  );
}

function Field({ label, value }) {
  if (!value) return null;
  return (
    <div>
      <p className="font-mono text-[11px] uppercase tracking-wider text-ink-500">{label}</p>
      <p className="mt-0.5 text-sm text-ink-100">{value}</p>
    </div>
  );
}

function FieldList({ label, items, tone = "default" }) {
  const color = tone === "amber" ? "text-signal-amber/80" : "text-ink-300";
  return (
    <div>
      <p className="font-mono text-[11px] uppercase tracking-wider text-ink-500">{label}</p>
      <ul className={`mt-0.5 list-inside list-disc space-y-0.5 text-sm ${color}`}>
        {items.map((it, i) => <li key={i}>{it}</li>)}
      </ul>
    </div>
  );
}
