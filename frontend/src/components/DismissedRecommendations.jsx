/**
 * Compact list of dismissed cleaning recommendations with a Restore control.
 * Restoring only changes the recommendation's status on the server; it never
 * applies anything or touches the dataset.
 */
export default function DismissedRecommendations({ recommendations, onRestore, busyId }) {
  if (!recommendations?.length) return null;
  return (
    <div className="mt-4 rounded-lg border border-graphite-border bg-graphite-800 p-3" data-testid="dismissed-recommendations">
      <p className="font-mono text-xs uppercase tracking-wider text-ink-500">
        Dismissed ({recommendations.length}) — hidden from the list, nothing was applied
      </p>
      <ul className="mt-2 space-y-2">
        {recommendations.map((rec) => (
          <li key={rec.id} className="flex flex-wrap items-center justify-between gap-2 text-xs">
            <span className="min-w-0 flex-1 text-ink-300">
              <span className="font-mono text-ink-500">
                {rec.step.replace(/_/g, " ")}{rec.column ? ` · ${rec.column}` : ""}
              </span>
              <span className="ml-2">{rec.issue}</span>
              {rec.changed_since_dismissed && (
                <span className="ml-2 rounded border border-signal-amber px-1.5 py-0.5 font-mono text-[10px] text-signal-amber">
                  Changed since dismissed
                </span>
              )}
            </span>
            <button
              onClick={() => onRestore(rec)}
              disabled={busyId === rec.id}
              className="rounded border border-graphite-border px-3 py-1 text-ink-300 hover:bg-graphite-700 disabled:opacity-50"
            >
              {busyId === rec.id ? "Restoring…" : "Restore"}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
