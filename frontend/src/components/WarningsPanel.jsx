const SEVERITY_STYLES = {
  critical: { dot: "bg-signal-red", label: "text-signal-red" },
  warning: { dot: "bg-signal-amber", label: "text-signal-amber" },
  info: { dot: "bg-ink-500", label: "text-ink-500" },
};

export default function WarningsPanel({ warnings }) {
  if (!warnings || warnings.length === 0) return null;

  return (
    <div className="rounded-lg border border-graphite-border bg-graphite-800 p-4">
      <p className="font-mono text-xs uppercase tracking-wider text-ink-500">
        Trust warnings ({warnings.length})
      </p>
      <div className="mt-3 space-y-3">
        {warnings.map((w) => {
          const style = SEVERITY_STYLES[w.severity] || SEVERITY_STYLES.info;
          return (
            <div key={w.id} className="flex gap-3">
              <span className={`mt-1.5 h-1.5 w-1.5 flex-shrink-0 rounded-full ${style.dot}`} />
              <div>
                <p className={`text-sm font-medium ${style.label}`}>{w.title}</p>
                <p className="mt-0.5 text-xs leading-relaxed text-ink-500">{w.description}</p>
                <p className="mt-1 font-mono text-[11px] text-ink-700">{w.reason}</p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
