import { useEffect, useState, useCallback } from "react";
import { getDashboard, unpinChart } from "../../api/client";
import ChartRenderer from "../charts/ChartRenderer";
import LoadingState from "../ui/LoadingState";
import ErrorState from "../ui/ErrorState";
import EmptyState from "../ui/EmptyState";

export default function DashboardWorkspace({ datasetId }) {
  const [dashboard, setDashboard] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getDashboard(datasetId);
      setDashboard(res);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [datasetId]);

  useEffect(() => { refresh(); }, [refresh]);

  const handleUnpin = async (chartId) => {
    await unpinChart(datasetId, chartId);
    await refresh();
  };

  if (loading && !dashboard) {
    return <LoadingState message="Loading dashboard…" />;
  }
  if (error) {
    return <ErrorState message={error} onRetry={refresh} />;
  }
  if (dashboard.charts.length === 0) {
    return (
      <EmptyState message="Nothing pinned yet — go to the Visualization tab and pin a chart to build your dashboard." />
    );
  }

  return (
    <div>
      <p className="mb-3 font-mono text-xs uppercase tracking-wider text-ink-500">
        Dashboard ({dashboard.charts.length} chart{dashboard.charts.length === 1 ? "" : "s"}) — dataset version {dashboard.current_version}
      </p>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {dashboard.charts.map((chart) => (
          <div key={chart.id} className="rounded-lg border border-graphite-border bg-graphite-800 p-4">
            <div className="flex items-start justify-between gap-2">
              <p className="font-display text-sm text-ink-100">{chart.title}</p>
              <button
                onClick={() => handleUnpin(chart.id)}
                className="font-mono text-[11px] text-ink-700 transition-colors hover:text-signal-red"
              >
                Unpin
              </button>
            </div>

            {chart.stale && !chart.error && (
              <p className="mt-1 font-mono text-[11px] text-signal-amber/80">
                ⟳ Dataset has changed since this was pinned — showing live, current-version data.
              </p>
            )}

            {chart.error ? (
              <div className="mt-2">
                <ErrorState message={`Can't render this chart anymore: ${chart.error}`} compact />
              </div>
            ) : (
              <div className="mt-3">
                <ChartRenderer chartType={chart.chart_type} data={chart.data} trend={chart.trend} />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
