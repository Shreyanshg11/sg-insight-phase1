function formatTimestamp(iso) {
  try {
    return new Date(iso).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  } catch {
    return iso;
  }
}

export default function RecentActionsPanel({ history }) {
  if (!history || history.length === 0) {
    return (
      <div className="rounded-lg border border-graphite-border bg-graphite-800 p-4">
        <p className="font-mono text-xs uppercase tracking-wider text-ink-500">Recent Actions</p>
        <p className="mt-2 text-xs text-ink-700">No actions taken yet this session.</p>
      </div>
    );
  }

  const recent = [...history].reverse().slice(0, 8);

  return (
    <div className="rounded-lg border border-graphite-border bg-graphite-800 p-4">
      <p className="font-mono text-xs uppercase tracking-wider text-ink-500">Recent Actions</p>
      <div className="mt-2 space-y-2">
        {recent.map((h, i) => (
          <div key={i} className="border-l-2 border-graphite-border pl-2.5">
            <p className="font-mono text-[10px] text-ink-700">v{h.version} · {formatTimestamp(h.timestamp)}</p>
            <p className="text-xs text-ink-100">{h.step} · {h.action}{h.column ? ` · ${h.column}` : ""}</p>
            {h.result_summary && <p className="text-[11px] text-ink-500">{h.result_summary}</p>}
          </div>
        ))}
      </div>
    </div>
  );
}
