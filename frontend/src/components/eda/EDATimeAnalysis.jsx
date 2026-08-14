import { useEffect, useState, useCallback } from "react";
import { getEDATimeAnalysis } from "../../api/client";
import { ColumnSelect } from "./EDADistribution";
import BarChart from "./BarChart";
import LoadingState from "../ui/LoadingState";
import ErrorState from "../ui/ErrorState";
import EmptyState from "../ui/EmptyState";

export default function EDATimeAnalysis({ datasetId, columns, selected, onSelect, filters }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const load = useCallback(() => {
    if (!selected) return;
    setLoading(true);
    setError(null);
    getEDATimeAnalysis(datasetId, selected, filters)
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [datasetId, selected, filters]);

  useEffect(() => { load(); }, [load]);

  if (columns.length === 0) {
    return <EmptyState message="No date-like columns detected in this dataset." />;
  }

  return (
    <div>
      <ColumnSelect columns={columns} selected={selected} onSelect={onSelect} />
      {loading && <div className="mt-3"><LoadingState compact /></div>}
      {error && <div className="mt-3"><ErrorState message={error} onRetry={load} compact /></div>}
      {data && !loading && (
        <div className="mt-3">
          <p className="mb-2 font-mono text-[11px] uppercase tracking-wider text-ink-500">
            {data.granularity} buckets
          </p>
          {data.buckets.length > 0
            ? <BarChart bars={data.buckets.map((b) => ({ label: b.period, count: b.count }))} />
            : <p className="font-mono text-xs text-ink-500">No parseable dates in this column.</p>}
          {data.note && <p className="mt-2 font-mono text-[11px] text-ink-700">{data.note}</p>}
        </div>
      )}
    </div>
  );
}
