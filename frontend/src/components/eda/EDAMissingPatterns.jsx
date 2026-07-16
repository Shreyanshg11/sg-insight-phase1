import BarChart from "./BarChart";

export default function EDAMissingPatterns({ data }) {
  if (data.columns_with_missing.length === 0) {
    return <p className="font-mono text-sm text-signal-green">No missing values anywhere in this dataset.</p>;
  }

  return (
    <div>
      <p className="mb-2 font-mono text-[11px] uppercase tracking-wider text-ink-500">Rows by missing-value count</p>
      <BarChart bars={data.missing_by_row_bucket} />

      <p className="mb-2 mt-5 font-mono text-[11px] uppercase tracking-wider text-ink-500">
        Columns with missing values ({data.columns_with_missing.length})
      </p>
      <div className="flex flex-wrap gap-1.5">
        {data.columns_with_missing.map((c) => (
          <span key={c} className="rounded border border-graphite-border bg-graphite-900 px-2 py-0.5 font-mono text-xs text-ink-300">
            {c}
          </span>
        ))}
      </div>

      {data.co_missing_pairs.length > 0 && (
        <>
          <p className="mb-2 mt-5 font-mono text-[11px] uppercase tracking-wider text-ink-500">Co-missing pairs</p>
          <div className="space-y-1">
            {data.co_missing_pairs.map((p, i) => (
              <div key={i} className="flex items-center justify-between rounded border border-graphite-border bg-graphite-900 px-3 py-1.5 font-mono text-xs">
                <span className="text-ink-300">{p.column_a} ↔ {p.column_b}</span>
                <span className="text-ink-500">{p.rows_missing_both} rows · ratio {p.co_missing_ratio}</span>
              </div>
            ))}
          </div>
        </>
      )}
      <p className="mt-3 font-mono text-[11px] text-ink-700">{data.note}</p>
    </div>
  );
}
