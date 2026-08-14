/**
 * Minimal horizontal bar chart used across every EDA sub-view (distributions,
 * category analysis, missing-row buckets, time analysis). Hand-rolled with
 * plain divs rather than a charting library, consistent with the rest of
 * this app's dependency-free frontend.
 */
export default function BarChart({ bars, valueFormatter = (v) => v, onBarClick }) {
  const max = Math.max(1, ...bars.map((b) => b.count));
  return (
    <div className="space-y-1.5">
      {bars.map((bar, i) => (
        <div
          key={i}
          className={`flex items-center gap-2 ${onBarClick ? "cursor-pointer rounded hover:bg-graphite-700/50" : ""}`}
          onClick={onBarClick ? () => onBarClick(bar) : undefined}
          title={onBarClick ? `Click to filter to '${bar.label}'` : bar.label}
        >
          <span className="w-32 flex-shrink-0 truncate font-mono text-[11px] text-ink-500">
            {bar.label}
          </span>
          <div className="h-4 flex-1 overflow-hidden rounded bg-graphite-900">
            <div
              className="h-full rounded bg-signal-amber/70"
              style={{ width: `${Math.max(2, (bar.count / max) * 100)}%` }}
            />
          </div>
          <span className="w-14 flex-shrink-0 text-right font-mono text-[11px] text-ink-300 tabular-nums">
            {valueFormatter(bar.count)}
          </span>
        </div>
      ))}
    </div>
  );
}
