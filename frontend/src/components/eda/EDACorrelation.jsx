import { Fragment } from "react";

function cellColor(value) {
  if (value === null || value === undefined) return "bg-graphite-900";
  const intensity = Math.min(1, Math.abs(value));
  if (value >= 0) {
    // signal-green tint, opacity scaled by strength
    return `rgba(79, 174, 140, ${0.15 + intensity * 0.65})`;
  }
  return `rgba(217, 105, 95, ${0.15 + intensity * 0.65})`;
}

export default function EDACorrelation({ data }) {
  if (!data.columns.length) {
    return <p className="font-mono text-sm text-ink-500">{data.note}</p>;
  }

  return (
    <div>
      <div className="overflow-x-auto">
        <div className="inline-grid" style={{ gridTemplateColumns: `120px repeat(${data.columns.length}, 64px)` }}>
          <div />
          {data.columns.map((c) => (
            <div key={c} className="flex items-end justify-center px-1 pb-1 font-mono text-[9px] text-ink-500" style={{ writingMode: "vertical-rl" }}>
              {c}
            </div>
          ))}
          {data.columns.map((rowCol, i) => (
            <Fragment key={rowCol}>
              <div className="flex items-center truncate pr-2 font-mono text-[11px] text-ink-500">
                {rowCol}
              </div>
              {data.columns.map((_, j) => {
                const val = data.matrix[i][j];
                return (
                  <div
                    key={`${i}-${j}`}
                    className="flex h-8 items-center justify-center border border-graphite-950 font-mono text-[10px] text-ink-100"
                    style={{ backgroundColor: cellColor(val) }}
                    title={`${rowCol} vs ${data.columns[j]}: ${val ?? "n/a"}`}
                  >
                    {val === null ? "—" : val.toFixed(2)}
                  </div>
                );
              })}
            </Fragment>
          ))}
        </div>
      </div>

      <p className="mb-1 mt-4 font-mono text-[11px] uppercase tracking-wider text-ink-500">Strongest pairs</p>
      <div className="space-y-1">
        {data.top_pairs.map((p, i) => (
          <div key={i} className="flex items-center justify-between rounded border border-graphite-border bg-graphite-900 px-3 py-1.5 font-mono text-xs">
            <span className="text-ink-300">{p.column_a} ↔ {p.column_b}</span>
            <span className={p.correlation >= 0 ? "text-signal-green" : "text-signal-red"}>
              {p.correlation.toFixed(3)} ({p.strength})
            </span>
          </div>
        ))}
      </div>
      <p className="mt-2 font-mono text-[11px] text-ink-700">{data.note}</p>
    </div>
  );
}
