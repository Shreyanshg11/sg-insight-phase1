/**
 * Consistent "in flight" state for any workbench tab/section that fetches
 * data. Replaces the ad-hoc `<p>Loading…</p>` that used to be duplicated in
 * every workspace component.
 */
export default function LoadingState({ message = "Loading…", compact = false }) {
  return (
    <div
      role="status"
      aria-live="polite"
      className={`flex items-center gap-3 font-mono text-sm text-ink-500 ${compact ? "py-2" : "py-10"}`}
    >
      <span
        aria-hidden="true"
        className="h-4 w-4 flex-shrink-0 animate-spin rounded-full border-2 border-graphite-border border-t-signal-amber"
      />
      <span>{message}</span>
    </div>
  );
}
