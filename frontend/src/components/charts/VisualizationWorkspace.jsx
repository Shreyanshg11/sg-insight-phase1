import { useEffect, useState, useCallback } from "react";
import { getChartRecommendations } from "../../api/client";
import ChartRecommendationCard from "../charts/ChartRecommendationCard";
import LoadingState from "../ui/LoadingState";
import ErrorState from "../ui/ErrorState";
import EmptyState from "../ui/EmptyState";
import { useFilters } from "../../context/FilterContext";

export default function VisualizationWorkspace({ datasetId }) {
  const [recommendations, setRecommendations] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const { filters, addFilter } = useFilters();

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getChartRecommendations(datasetId, filters);
      setRecommendations(res.recommendations);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [datasetId, filters]);

  useEffect(() => { refresh(); }, [refresh]);

  if (loading && !recommendations) {
    return <LoadingState message="Working out which charts fit this dataset…" />;
  }
  if (error) {
    return <ErrorState message={error} onRetry={refresh} />;
  }
  if (recommendations.length === 0) {
    return <EmptyState message="No chart recommendations for this dataset yet." />;
  }

  return (
    <div>
      <p className="mb-3 font-mono text-xs uppercase tracking-wider text-ink-500">
        Recommended charts ({recommendations.length})
        <span className="ml-2 text-ink-700">— pin any of these to the Dashboard tab</span>
        {filters.length > 0 && <span className="ml-2 text-signal-amber">— reflecting active filter</span>}
      </p>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {recommendations.map((rec) => (
          <ChartRecommendationCard key={rec.id} datasetId={datasetId} rec={rec} onDrillDown={addFilter} />
        ))}
      </div>
    </div>
  );
}
