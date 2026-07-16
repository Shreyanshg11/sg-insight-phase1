import { useEffect, useState, useCallback } from "react";
import { getChartRecommendations } from "../../api/client";
import ChartRecommendationCard from "../charts/ChartRecommendationCard";
import LoadingState from "../ui/LoadingState";
import ErrorState from "../ui/ErrorState";
import EmptyState from "../ui/EmptyState";

export default function VisualizationWorkspace({ datasetId }) {
  const [recommendations, setRecommendations] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getChartRecommendations(datasetId);
      setRecommendations(res.recommendations);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [datasetId]);

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
      </p>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {recommendations.map((rec) => (
          <ChartRecommendationCard key={rec.id} datasetId={datasetId} rec={rec} />
        ))}
      </div>
    </div>
  );
}
