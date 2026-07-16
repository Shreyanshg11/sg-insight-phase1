export default function EDAOverview({ data }) {
  return (
    <div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Rows" value={data.rows} />
        <Stat label="Columns" value={data.columns} />
        <Stat label="Numeric" value={data.numeric_columns.length} />
        <Stat label="Categorical" value={data.categorical_columns.length} />
      </div>

      <p className="mb-2 mt-5 font-mono text-[11px] uppercase tracking-wider text-ink-500">Per-column summary</p>
      <div className="overflow-x-auto rounded-lg border border-graphite-border">
        <table className="w-full text-left text-xs">
          <thead className="bg-graphite-900 text-ink-500">
            <tr>
              <th className="px-3 py-2 font-mono font-normal">Column</th>
              <th className="px-3 py-2 font-mono font-normal">Kind</th>
              <th className="px-3 py-2 font-mono font-normal">Missing %</th>
              <th className="px-3 py-2 font-mono font-normal">Detail</th>
            </tr>
          </thead>
          <tbody>
            {data.summary.map((row) => (
              <tr key={row.column} className="border-t border-graphite-border">
                <td className="px-3 py-2 font-mono text-ink-100">{row.column}</td>
                <td className="px-3 py-2 font-mono text-ink-500">{row.kind}</td>
                <td className="px-3 py-2 font-mono text-ink-300 tabular-nums">{row.missing_pct}%</td>
                <td className="px-3 py-2 font-mono text-ink-500">
                  {row.kind === "numeric" && `mean ${row.mean} · median ${row.median} · std ${row.std}`}
                  {row.kind === "date" && `${row.min} → ${row.max}`}
                  {row.kind === "categorical" && `${row.unique_count} unique · top "${row.top_value}" (${row.top_value_pct}%)`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="rounded-lg border border-graphite-border bg-graphite-800 p-3">
      <p className="font-mono text-[11px] uppercase tracking-wider text-ink-500">{label}</p>
      <p className="mt-1 font-display text-xl text-ink-100">{value}</p>
    </div>
  );
}
