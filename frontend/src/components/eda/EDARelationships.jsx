import { useEffect, useState, useCallback } from "react";
import { getEDARelationships } from "../../api/client";
import { ColumnSelect } from "./EDADistribution";
import BarChart from "./BarChart";
import LoadingState from "../ui/LoadingState";
import ErrorState from "../ui/ErrorState";

function Scatter({ points }) {
  const width = 480, height = 260, pad = 24;
  const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const scaleX = (x) => pad + ((x - xMin) / (xMax - xMin || 1)) * (width - 2 * pad);
  const scaleY = (y) => height - pad - ((y - yMin) / (yMax - yMin || 1)) * (height - 2 * pad);

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full max-w-xl rounded border border-graphite-border bg-graphite-900">
      <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="#2A2F3A" />
      <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="#2A2F3A" />
      {points.map((p, i) => (
        <circle key={i} cx={scaleX(p.x)} cy={scaleY(p.y)} r={2.5} fill="#E8A33D" fillOpacity={0.6} />
      ))}
    </svg>
  );
}

export default function EDARelationships({ datasetId, allColumns, x, y, onSelectX, onSelectY, filters }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const load = useCallback(() => {
    if (!x || !y) return;
    setLoading(true);
    setError(null);
    getEDARelationships(datasetId, x, y, filters)
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [datasetId, x, y, filters]);

  useEffect(() => { load(); }, [load]);

  return (
    <div>
      <div className="flex flex-wrap gap-4">
        <ColumnSelect columns={allColumns} selected={x} onSelect={onSelectX} label="X" />
        <ColumnSelect columns={allColumns} selected={y} onSelect={onSelectY} label="Y" />
      </div>

      {loading && <div className="mt-3"><LoadingState compact /></div>}
      {error && <div className="mt-3"><ErrorState message={error} onRetry={load} compact /></div>}

      {data && !loading && (
        <div className="mt-3">
          {data.kind === "numeric_numeric" && (
            <>
              <Scatter points={data.points} />
              <p className="mt-2 font-mono text-xs text-ink-300">
                Correlation: <span className="text-ink-100">{data.stats.correlation ?? "n/a"}</span> (n={data.stats.n})
              </p>
            </>
          )}
          {data.kind === "numeric_categorical" && (
            <BarChart bars={data.points.map((p) => ({ label: p.category, count: p.mean }))} valueFormatter={(v) => v.toFixed(1)} />
          )}
          {data.kind === "categorical_categorical" && (
            <div className="overflow-x-auto rounded border border-graphite-border">
              <table className="text-xs">
                <tbody>
                  {[...new Set(data.points.map((p) => p.row))].map((r) => (
                    <tr key={r} className="border-t border-graphite-border">
                      <td className="px-2 py-1 font-mono text-ink-500">{r}</td>
                      {[...new Set(data.points.map((p) => p.col))].map((c) => {
                        const cell = data.points.find((p) => p.row === r && p.col === c);
                        return <td key={c} className="px-2 py-1 text-center font-mono text-ink-100">{cell ? cell.count : 0}</td>;
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {data.note && <p className="mt-2 font-mono text-[11px] text-ink-700">{data.note}</p>}
        </div>
      )}
    </div>
  );
}
