import { useEffect, useState, useCallback } from "react";
import { getEDAOverview, getEDACorrelation, getEDAMissingPatterns } from "../api/client";
import EDAOverview from "./eda/EDAOverview";
import EDADistribution from "./eda/EDADistribution";
import EDACorrelation from "./eda/EDACorrelation";
import EDAMissingPatterns from "./eda/EDAMissingPatterns";
import EDARelationships from "./eda/EDARelationships";
import EDACategoryAnalysis from "./eda/EDACategoryAnalysis";
import EDATimeAnalysis from "./eda/EDATimeAnalysis";
import TabNav from "./ui/TabNav";
import LoadingState from "./ui/LoadingState";
import ErrorState from "./ui/ErrorState";
import EmptyState from "./ui/EmptyState";
import { useFilters } from "../context/FilterContext";

const SECTIONS = [
  { key: "overview", label: "Overview" },
  { key: "distributions", label: "Distributions" },
  { key: "correlation", label: "Correlation" },
  { key: "missing_patterns", label: "Missing Patterns" },
  { key: "relationships", label: "Relationships" },
  { key: "category_analysis", label: "Category Analysis" },
  { key: "time_analysis", label: "Time Analysis" },
];

/**
 * Each section is independently fetchable (#13) — the sub-nav below switches
 * between them without ever running one monolithic "compute everything" call,
 * matching the shape PipelineNav already anticipated for the EDA stage.
 *
 * Filter-aware (session 12): every section recomputes against the active
 * global filter, if any — correlation/missing-patterns caches are cleared
 * on filter change so they refetch rather than showing a stale view.
 */
export default function EDAWorkspace({ datasetId }) {
  const [activeSection, setActiveSection] = useState("overview");
  const [overview, setOverview] = useState(null);
  const [correlation, setCorrelation] = useState(null);
  const [missingPatterns, setMissingPatterns] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const { filters } = useFilters();

  const [distColumn, setDistColumn] = useState(null);
  const [categoryColumn, setCategoryColumn] = useState(null);
  const [timeColumn, setTimeColumn] = useState(null);
  const [relX, setRelX] = useState(null);
  const [relY, setRelY] = useState(null);

  const loadOverview = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const ov = await getEDAOverview(datasetId, filters);
      setOverview(ov);
      const allNumericLike = [...ov.numeric_columns, ...ov.categorical_columns];
      setDistColumn((prev) => prev || allNumericLike[0] || null);
      setCategoryColumn((prev) => prev || ov.categorical_columns[0] || null);
      setTimeColumn((prev) => prev || ov.date_columns[0] || null);
      setRelX((prev) => prev || allNumericLike[0] || null);
      setRelY((prev) => prev || allNumericLike[1] || allNumericLike[0] || null);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [datasetId, filters]);

  useEffect(() => { loadOverview(); }, [loadOverview]);

  // Filter changed — cached correlation/missing-patterns no longer reflect
  // the active view, so drop them and let the effect below refetch.
  useEffect(() => { setCorrelation(null); setMissingPatterns(null); }, [filters]);

  useEffect(() => {
    if (activeSection === "correlation" && !correlation) {
      getEDACorrelation(datasetId, filters).then(setCorrelation).catch((e) => setError(e.message));
    }
    if (activeSection === "missing_patterns" && !missingPatterns) {
      getEDAMissingPatterns(datasetId, filters).then(setMissingPatterns).catch((e) => setError(e.message));
    }
  }, [activeSection, datasetId, filters, correlation, missingPatterns]);

  if (loading && !overview) {
    return <LoadingState message="Loading EDA overview…" />;
  }
  if (error && !overview) {
    return <ErrorState message={error} onRetry={loadOverview} />;
  }

  const allColumns = [...overview.numeric_columns, ...overview.categorical_columns];

  return (
    <div>
      <div className="mb-4 -mx-4 overflow-x-auto border-b border-graphite-border px-4 sm:mx-0 sm:px-0">
        <TabNav
          items={SECTIONS}
          activeKey={activeSection}
          onChange={setActiveSection}
          orientation="horizontal"
          ariaLabel="EDA sections"
          idPrefix="eda-section"
          className="flex w-max gap-1"
          itemClassName={(item, isActive) =>
            `whitespace-nowrap px-3 py-2 font-mono text-xs uppercase tracking-wider transition-colors focus-visible:outline-none ${
              isActive ? "border-b-2 border-signal-amber text-signal-amber" : "border-b-2 border-transparent text-ink-500 hover:text-ink-300"
            }`
          }
        />
      </div>

      <div role="tabpanel" id={`eda-section-panel-${activeSection}`} aria-labelledby={`eda-section-${activeSection}`}>
        {activeSection === "overview" && <EDAOverview data={overview} />}

        {activeSection === "distributions" && (
          <EDADistribution datasetId={datasetId} columns={allColumns} selected={distColumn} onSelect={setDistColumn} filters={filters} />
        )}

        {activeSection === "correlation" && (
          correlation ? <EDACorrelation data={correlation} /> : <LoadingState compact />
        )}

        {activeSection === "missing_patterns" && (
          missingPatterns ? <EDAMissingPatterns data={missingPatterns} /> : <LoadingState compact />
        )}

        {activeSection === "relationships" && (
          <EDARelationships
            datasetId={datasetId}
            allColumns={allColumns}
            x={relX} y={relY}
            onSelectX={setRelX} onSelectY={setRelY}
            filters={filters}
          />
        )}

        {activeSection === "category_analysis" && (
          overview.categorical_columns.length > 0
            ? <EDACategoryAnalysis datasetId={datasetId} columns={overview.categorical_columns} selected={categoryColumn} onSelect={setCategoryColumn} filters={filters} />
            : <EmptyState message="No categorical columns detected in this dataset." />
        )}

        {activeSection === "time_analysis" && (
          <EDATimeAnalysis datasetId={datasetId} columns={overview.date_columns} selected={timeColumn} onSelect={setTimeColumn} filters={filters} />
        )}
      </div>
    </div>
  );
}
