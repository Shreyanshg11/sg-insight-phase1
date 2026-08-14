import { useFilters } from "../../context/FilterContext";

function opLabel(f) {
  switch (f.op) {
    case "eq": return `= ${f.value}`;
    case "in": return `in [${(f.value || []).join(", ")}]`;
    case "gte": return `≥ ${f.value}`;
    case "lte": return `≤ ${f.value}`;
    case "between": return `between ${f.value?.[0]} – ${f.value?.[1]}`;
    default: return String(f.value);
  }
}

export default function FilterBar() {
  const { filters, removeFilter, clearFilters, setDrillDownOpen } = useFilters();
  if (filters.length === 0) return null;

  return (
    <div className="flex flex-wrap items-center gap-2 border-b border-graphite-border bg-graphite-900 px-4 py-2 sm:px-6">
      <span className="font-mono text-[10px] uppercase tracking-wider text-ink-500">Filtered:</span>
      {filters.map((f) => (
        <span key={f.column} className="flex items-center gap-1.5 rounded-full border border-signal-amber/40 bg-signal-amber/10 px-2.5 py-1 font-mono text-[11px] text-signal-amber">
          {f.column} {opLabel(f)}
          <button onClick={() => removeFilter(f.column)} aria-label={`Remove filter on ${f.column}`} className="text-signal-amber/70 hover:text-signal-amber focus-visible:outline-none">
            ×
          </button>
        </span>
      ))}
      <button onClick={() => setDrillDownOpen(true)} className="rounded border border-graphite-border px-2 py-1 font-mono text-[11px] text-ink-300 hover:bg-graphite-700">
        View matching rows
      </button>
      <button onClick={clearFilters} className="font-mono text-[11px] text-ink-500 hover:text-ink-300">
        Clear all
      </button>
    </div>
  );
}
