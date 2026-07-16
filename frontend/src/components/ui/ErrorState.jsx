/**
 * Consistent error state for any workbench tab/section. `onRetry` is
 * optional — omit it for errors that aren't safely retryable (e.g. a
 * destructive action that already partially applied).
 */
export default function ErrorState({ message, onRetry, compact = false }) {
  return (
    <div
      role="alert"
      className={`flex flex-col items-start gap-3 rounded-lg border border-signal-red/40 bg-signal-red/5 font-mono text-sm text-signal-red ${
        compact ? "p-3" : "p-4"
      }`}
    >
      <p>{message || "Something went wrong."}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="rounded border border-signal-red/50 px-3 py-1.5 text-xs text-signal-red hover:bg-signal-red/10 focus-visible:outline-none"
        >
          Retry
        </button>
      )}
    </div>
  );
}
