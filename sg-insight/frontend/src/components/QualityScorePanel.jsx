const DIMENSIONS = [
  { key: "completeness", label: "Completeness" },
  { key: "consistency", label: "Consistency" },
  { key: "validity", label: "Validity" },
  { key: "uniqueness", label: "Uniqueness" },
];

export default function QualityScorePanel({ score }) {
  return (
    <div className="rounded-lg border border-graphite-border bg-graphite-800 p-5">
      <div className="flex items-baseline justify-between">
        <p className="font-mono text-xs uppercase tracking-wider text-ink-500">Data Quality</p>
        <p className="font-display text-3xl font-semibold text-signal-amber tabular-nums">
          {score.overall}<span className="text-base text-ink-500">/100</span>
        </p>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-4">
        {DIMENSIONS.map((d) => (
          <div key={d.key}>
            <div className="flex justify-between font-mono text-xs text-ink-500">
              <span>{d.label}</span>
              <span className="text-ink-300 tabular-nums">{score[d.key]}</span>
            </div>
            <div className="mt-1 h-1.5 w-full rounded-full bg-graphite-700">
              <div
                className="h-1.5 rounded-full bg-signal-green"
                style={{ width: `${score[d.key]}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
