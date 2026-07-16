/**
 * Consistent "nothing to show here" state — distinct from ErrorState (which
 * means the fetch failed) and LoadingState (which means it's in flight).
 * Used when a request succeeded but legitimately returned nothing: no
 * outstanding recommendations, no columns of the needed type, no pinned
 * charts yet, etc.
 */
export default function EmptyState({ title, message, action, compact = false }) {
  return (
    <div
      className={`rounded-lg border border-graphite-border bg-graphite-800 text-ink-500 ${compact ? "p-3" : "p-4"}`}
    >
      {title && <p className="mb-1 font-mono text-xs uppercase tracking-wider text-ink-700">{title}</p>}
      <p className="text-sm">{message}</p>
      {action && <div className="mt-3">{action}</div>}
    </div>
  );
}
