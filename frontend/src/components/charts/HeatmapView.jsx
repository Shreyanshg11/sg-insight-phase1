import { Fragment } from "react";

function cellColor(value) {
  if (value === null || value === undefined) return "bg-graphite-900";
  const intensity = Math.min(1, Math.abs(value));
  return value >= 0 ? `rgba(79, 174, 140, ${0.15 + intensity * 0.65})` : `rgba(217, 105, 95, ${0.15 + intensity * 0.65})`;
}

export default function HeatmapView({ columns, matrix }) {
  if (!columns || columns.length === 0) {
    return <p className="font-mono text-xs text-ink-500">Not enough numeric columns to correlate.</p>;
  }
  return (
    <div className="overflow-x-auto">
      <div className="inline-grid" style={{ gridTemplateColumns: `110px repeat(${columns.length}, 56px)` }}>
        <div />
        {columns.map((c) => (
          <div key={c} className="flex items-end justify-center px-1 pb-1 font-mono text-[9px] text-ink-500" style={{ writingMode: "vertical-rl" }}>
            {c}
          </div>
        ))}
        {columns.map((rowCol, i) => (
          <Fragment key={rowCol}>
            <div className="flex items-center truncate pr-2 font-mono text-[11px] text-ink-500">{rowCol}</div>
            {columns.map((_, j) => {
              const val = matrix[i][j];
              return (
                <div key={`${i}-${j}`}
                     className="flex h-7 items-center justify-center border border-graphite-950 font-mono text-[9px] text-ink-100"
                     style={{ backgroundColor: cellColor(val) }} title={`${rowCol} vs ${columns[j]}: ${val ?? "n/a"}`}>
                  {val === null ? "—" : val.toFixed(2)}
                </div>
              );
            })}
          </Fragment>
        ))}
      </div>
    </div>
  );
}
