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
      <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
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

      {score.breakdown?.length > 0 && (
        <div className="mt-5 border-t border-graphite-border pt-4">
          <p className="mb-2 font-mono text-[11px] uppercase tracking-wider text-ink-500">
            Why the score is {score.overall}
          </p>
          <div className="space-y-1.5">
            {score.breakdown.map((item) => (
              <div key={item.label} tabIndex={0} className="group relative flex items-center justify-between rounded focus-visible:outline-none">
                <span className="font-mono text-xs text-ink-300">{item.label}</span>
                <span className={`font-mono text-xs tabular-nums ${item.points_deducted > 0 ? "text-signal-red" : "text-signal-green"}`}>
                  {item.points_deducted > 0 ? `−${item.points_deducted}` : "0"}
                </span>
                <div className="pointer-events-none absolute left-0 top-full z-10 mt-1 hidden w-64 rounded border border-graphite-border bg-graphite-900 p-2 text-[11px] leading-snug text-ink-500 shadow-lg group-hover:block group-focus-within:block">
                  {item.reason}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
