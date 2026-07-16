const STATUS_STYLES = {
  good: { dot: "bg-signal-green", border: "border-graphite-border", text: "text-signal-green" },
  warning: { dot: "bg-signal-amber", border: "border-signal-amber/40", text: "text-signal-amber" },
  critical: { dot: "bg-signal-red", border: "border-signal-red/50", text: "text-signal-red" },
};

function HealthCard({ category }) {
  const style = STATUS_STYLES[category.status] || STATUS_STYLES.good;
  return (
    <div className={`rounded-lg border ${style.border} bg-graphite-800 p-3.5`}>
      <div className="flex items-center gap-2">
        <span className={`h-1.5 w-1.5 rounded-full ${style.dot}`} />
        <p className="font-mono text-[11px] uppercase tracking-wider text-ink-500">{category.label}</p>
      </div>
      <p className={`mt-1.5 text-sm font-medium ${category.status === "good" ? "text-ink-100" : style.text}`}>
        {category.summary}
      </p>
      <p className="mt-1 text-[11px] leading-snug text-ink-700">{category.detail}</p>
      {category.affected_columns?.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {category.affected_columns.slice(0, 5).map((c) => (
            <span key={c} className="rounded border border-graphite-border bg-graphite-900 px-1.5 py-0.5 font-mono text-[10px] text-ink-300">
              {c}
            </span>
          ))}
          {category.affected_columns.length > 5 && (
            <span className="font-mono text-[10px] text-ink-700">+{category.affected_columns.length - 5} more</span>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * "This becomes the first screen users see after profiling" — meant to be
 * placed above everything else in the workbench, giving an at-a-glance read
 * on the dataset before diving into the detailed profile/quality/cleaning views.
 */
export default function HealthDashboard({ health }) {
  if (!health?.categories?.length) return null;

  const issueCount = health.categories.filter((c) => c.status !== "good").length;

  return (
    <div className="rounded-lg border border-graphite-border bg-graphite-900 p-4">
      <div className="flex items-baseline justify-between">
        <p className="font-mono text-xs uppercase tracking-wider text-ink-500">Dataset Health</p>
        <p className="font-mono text-xs text-ink-500">
          {issueCount === 0 ? "All clear" : `${issueCount} area(s) flagged`}
        </p>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
        {health.categories.map((cat) => (
          <HealthCard key={cat.key} category={cat} />
        ))}
      </div>
    </div>
  );
}
