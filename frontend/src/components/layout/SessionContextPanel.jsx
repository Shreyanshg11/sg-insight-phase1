export default function SessionContextPanel({ goal, goalRelevantColumns }) {
  if (!goal) {
    return (
      <div className="rounded-lg border border-graphite-border bg-graphite-800 p-4">
        <p className="font-mono text-xs uppercase tracking-wider text-ink-500">Session Context</p>
        <p className="mt-2 text-xs text-ink-700">No investigation goal set for this session.</p>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-signal-amber/30 bg-signal-amber/5 p-4">
      <p className="font-mono text-xs uppercase tracking-wider text-signal-amber">Session Context</p>
      <p className="mt-2 text-sm text-ink-100">{goal}</p>
      {goalRelevantColumns?.length > 0 && (
        <div className="mt-2">
          <p className="font-mono text-[11px] text-ink-500">{goalRelevantColumns.length} relevant column(s):</p>
          <div className="mt-1 flex flex-wrap gap-1">
            {goalRelevantColumns.map((c) => (
              <span key={c} className="rounded border border-graphite-border bg-graphite-900 px-1.5 py-0.5 font-mono text-[10px] text-ink-300">
                {c}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
