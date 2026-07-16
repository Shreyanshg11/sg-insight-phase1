import { useEffect, useState, useCallback } from "react";
import { getEDADistribution } from "../../api/client";
import BarChart from "./BarChart";
import LoadingState from "../ui/LoadingState";
import ErrorState from "../ui/ErrorState";
import EmptyState from "../ui/EmptyState";

export default function EDADistribution({ datasetId, columns, selected, onSelect }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const load = useCallback(() => {
    if (!selected) return;
    setLoading(true);
    setError(null);
    getEDADistribution(datasetId, selected)
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [datasetId, selected]);

  useEffect(() => { load(); }, [load]);

  if (columns.length === 0) {
    return <EmptyState message="No numeric or categorical columns detected in this dataset." />;
  }

  return (
    <div>
      <ColumnSelect columns={columns} selected={selected} onSelect={onSelect} />
      {loading && <div className="mt-3"><LoadingState compact /></div>}
      {error && <div className="mt-3"><ErrorState message={error} onRetry={load} compact /></div>}
      {data && !loading && (
        <div className="mt-3">
          {data.stats && Object.keys(data.stats).length > 0 && (
            <div className="mb-3 flex flex-wrap gap-3">
              {Object.entries(data.stats).map(([k, v]) => (
                <span key={k} className="rounded border border-graphite-border bg-graphite-900 px-2 py-1 font-mono text-[11px] text-ink-300">
                  {k}: <span className="text-ink-100">{v}</span>
                </span>
              ))}
            </div>
          )}
          {data.bins.length > 0
            ? <BarChart bars={data.bins} />
            : <EmptyState compact message="No data to plot." />}
          {data.note && <p className="mt-2 font-mono text-[11px] text-ink-700">{data.note}</p>}
        </div>
      )}
    </div>
  );
}

export function ColumnSelect({ columns, selected, onSelect, label = "Column" }) {
  return (
    <div className="flex items-center gap-2">
      <span className="font-mono text-[11px] uppercase tracking-wider text-ink-500">{label}</span>
      <select
        value={selected || ""}
        onChange={(e) => onSelect(e.target.value)}
        className="rounded border border-graphite-border bg-graphite-800 px-2 py-1 font-mono text-[11px] text-ink-300 focus:border-signal-amber focus:outline-none"
      >
        {columns.map((c) => <option key={c} value={c}>{c}</option>)}
      </select>
    </div>
  );
}
