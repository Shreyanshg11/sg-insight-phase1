import { useEffect, useState, useCallback } from "react";
import { getEDACategoryAnalysis } from "../../api/client";
import { ColumnSelect } from "./EDADistribution";
import BarChart from "./BarChart";
import LoadingState from "../ui/LoadingState";
import ErrorState from "../ui/ErrorState";

export default function EDACategoryAnalysis({ datasetId, columns, selected, onSelect }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const load = useCallback(() => {
    if (!selected) return;
    setLoading(true);
    setError(null);
    getEDACategoryAnalysis(datasetId, selected)
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [datasetId, selected]);

  useEffect(() => { load(); }, [load]);

  return (
    <div>
      <ColumnSelect columns={columns} selected={selected} onSelect={onSelect} />
      {loading && <div className="mt-3"><LoadingState compact /></div>}
      {error && <div className="mt-3"><ErrorState message={error} onRetry={load} compact /></div>}
      {data && !loading && (
        <div className="mt-3">
          <p className="mb-2 font-mono text-xs text-ink-300">
            {data.unique_count} distinct value(s) · top {data.categories.length} cover {data.top_n_coverage_pct}% of rows
          </p>
          <BarChart bars={data.categories.map((c) => ({ label: c.value, count: c.count }))}
                    valueFormatter={(v) => v} />
          {data.note && <p className="mt-2 font-mono text-[11px] text-ink-700">{data.note}</p>}
        </div>
      )}
    </div>
  );
}
