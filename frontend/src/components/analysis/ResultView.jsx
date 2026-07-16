/**
 * Generic renderer for an analysis `result` payload — deliberately schema-free
 * (regression coefficients, cluster centers, PCA loadings, etc. all have
 * different shapes) rather than one bespoke chart per analysis type.
 */
export default function ResultView({ value, depth = 0 }) {
  if (value === null || value === undefined) {
    return <span className="font-mono text-ink-700">—</span>;
  }
  if (typeof value !== "object") {
    return <span className="font-mono text-ink-100">{String(value)}</span>;
  }
  if (Array.isArray(value)) {
    if (value.length === 0) return <span className="font-mono text-ink-700">(none)</span>;
    if (value.every((v) => typeof v !== "object")) {
      return <span className="font-mono text-ink-100">{value.join(", ")}</span>;
    }
    return (
      <div className="space-y-1 pl-3">
        {value.map((v, i) => (
          <div key={i} className="rounded border border-graphite-border bg-graphite-900 p-2">
            <ResultView value={v} depth={depth + 1} />
          </div>
        ))}
      </div>
    );
  }
  return (
    <div className={depth > 0 ? "space-y-1" : "space-y-1.5"}>
      {Object.entries(value).map(([k, v]) => (
        <div key={k} className="flex items-start gap-2 font-mono text-xs">
          <span className="w-40 flex-shrink-0 text-ink-500">{k}</span>
          <div className="flex-1"><ResultView value={v} depth={depth + 1} /></div>
        </div>
      ))}
    </div>
  );
}
