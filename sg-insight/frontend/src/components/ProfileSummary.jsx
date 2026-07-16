function Stat({ label, value }) {
  return (
    <div>
      <p className="font-mono text-[11px] uppercase tracking-wider text-ink-500">{label}</p>
      <p className="font-display text-lg text-ink-100">{value}</p>
    </div>
  );
}

function Tag({ children }) {
  return (
    <span className="rounded border border-graphite-border bg-graphite-900 px-2 py-0.5 font-mono text-[11px] text-ink-300">
      {children}
    </span>
  );
}

export default function ProfileSummary({ profile }) {
  return (
    <div className="rounded-lg border border-graphite-border bg-graphite-800 p-5">
      <p className="font-mono text-xs uppercase tracking-wider text-ink-500">Dataset Profile</p>
      <div className="mt-4 grid grid-cols-4 gap-4">
        <Stat label="Rows" value={profile.rows.toLocaleString()} />
        <Stat label="Columns" value={profile.columns} />
        <Stat label="Memory" value={`${profile.memory_kb} KB`} />
        <Stat label="Duplicates" value={`${profile.duplicate_rows} (${profile.duplicate_pct}%)`} />
      </div>

      <div className="mt-5 space-y-2">
        {profile.id_column_candidates.length > 0 && (
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-mono text-[11px] text-ink-500">ID columns</span>
            {profile.id_column_candidates.map((c) => <Tag key={c}>{c}</Tag>)}
          </div>
        )}
        {profile.target_candidates.length > 0 && (
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-mono text-[11px] text-ink-500">Target candidates</span>
            {profile.target_candidates.map((c) => <Tag key={c}>{c}</Tag>)}
          </div>
        )}
        {profile.date_columns.length > 0 && (
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-mono text-[11px] text-ink-500">Date columns</span>
            {profile.date_columns.map((c) => <Tag key={c}>{c}</Tag>)}
          </div>
        )}
        {profile.geographic_columns.length > 0 && (
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-mono text-[11px] text-ink-500">Geographic columns</span>
            {profile.geographic_columns.map((c) => <Tag key={c}>{c}</Tag>)}
          </div>
        )}
      </div>

      <div className="mt-5 overflow-x-auto">
        <table className="w-full border-collapse text-left text-sm">
          <thead>
            <tr className="border-b border-graphite-border font-mono text-[11px] uppercase tracking-wider text-ink-500">
              <th className="py-2 pr-4">Column</th>
              <th className="py-2 pr-4">Type</th>
              <th className="py-2 pr-4">Role</th>
              <th className="py-2 pr-4">Missing</th>
              <th className="py-2 pr-4">Unique</th>
              <th className="py-2">Sample</th>
            </tr>
          </thead>
          <tbody>
            {profile.columns_detail.map((col) => (
              <tr key={col.name} className="border-b border-graphite-border/60">
                <td className="py-2 pr-4 font-medium text-ink-100">{col.name}</td>
                <td className="py-2 pr-4 font-mono text-xs text-ink-500">{col.dtype}</td>
                <td className="py-2 pr-4"><Tag>{col.inferred_role}</Tag></td>
                <td className="py-2 pr-4 font-mono text-xs text-ink-300">
                  {col.missing_count} ({col.missing_pct}%)
                </td>
                <td className="py-2 pr-4 font-mono text-xs text-ink-300">{col.unique_count}</td>
                <td className="py-2 font-mono text-xs text-ink-500 truncate max-w-[200px]">
                  {col.sample_values.join(", ")}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
