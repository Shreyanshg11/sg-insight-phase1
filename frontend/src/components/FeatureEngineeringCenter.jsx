import { useEffect, useState, useCallback } from "react";
import { getFeatureRecommendations, applyFeatureAction } from "../api/client";
import RecommendationCard from "./RecommendationCard";
import LoadingState from "./ui/LoadingState";
import ErrorState from "./ui/ErrorState";
import EmptyState from "./ui/EmptyState";

const CATEGORY_LABELS = {
  ratio: "Ratio",
  date_features: "Date Features",
  bucketing: "Bucketing",
  scaling_normalization: "Scaling / Normalization",
  encoding: "Encoding",
  log_transform: "Log Transform",
};

const FILTERS = [
  { value: "all", label: "All categories" },
  ...Object.entries(CATEGORY_LABELS).map(([value, label]) => ({ value, label })),
];

/**
 * Adapts a FeatureRecommendation (category/columns/new_column_name) into the
 * shape RecommendationCard already knows how to render (step/column) — this
 * is the concrete instance of the README's promise that RecommendationCard
 * "is the one component every future cleaning/analysis module should reuse."
 * No changes to RecommendationCard's logic were needed, only its STEP_LABELS map.
 */
function toCardShape(rec) {
  return {
    ...rec,
    step: rec.category,
    column: rec.columns.join(" & "),
  };
}

export default function FeatureEngineeringCenter({ datasetId, onDatasetChanged }) {
  const [recommendations, setRecommendations] = useState(null);
  const [dismissed, setDismissed] = useState(new Set());
  const [filter, setFilter] = useState("all");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getFeatureRecommendations(datasetId);
      setRecommendations(res.recommendations);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [datasetId]);

  useEffect(() => { refresh(); }, [refresh]);

  const handleConfirm = async (rec, overrides = {}) => {
    await applyFeatureAction({
      dataset_id: datasetId,
      category: rec.category,
      action: overrides.action || rec.recommended_action,
      columns: rec.columns,
      new_column_name: rec.new_column_name,
      params: overrides.params || {},
      reason: rec.reason,
      confidence: rec.confidence,
    });
    await refresh();
    await onDatasetChanged?.();
  };

  if (loading && !recommendations) {
    return <LoadingState message="Scanning dataset for feature opportunities…" />;
  }
  if (error) {
    return <ErrorState message={error} onRetry={refresh} />;
  }

  const visible = (recommendations ?? [])
    .filter((r) => !dismissed.has(r.id))
    .filter((r) => filter === "all" || r.category === filter);

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="font-mono text-xs uppercase tracking-wider text-ink-500">
          Feature suggestions ({visible.length})
          <span className="ml-2 text-ink-700">— every new column is additive; nothing is overwritten</span>
        </p>
        <select
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="rounded border border-graphite-border bg-graphite-800 px-2 py-1 font-mono text-[11px] text-ink-300 focus:border-signal-amber focus:outline-none"
        >
          {FILTERS.map((f) => <option key={f.value} value={f.value}>{f.label}</option>)}
        </select>
      </div>

      <div className="mt-3 space-y-3">
        {visible.length === 0 && (
          <EmptyState message="No outstanding feature suggestions in this category — try a different filter, or the dataset may not have obvious candidates left." />
        )}
        {visible.map((rec) => (
          <RecommendationCard
            key={rec.id}
            recommendation={toCardShape(rec)}
            onConfirm={(_cardRec, overrides) => handleConfirm(rec, overrides)}
            onDismiss={(r) => setDismissed((prev) => new Set(prev).add(r.id))}
          />
        ))}
      </div>
    </div>
  );
}
