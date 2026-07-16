import { useEffect, useState, useCallback, useMemo } from "react";
import { getAnalysisRecommendations, runAnalysis, getAIInsight } from "../../api/client";
import AnalysisRecommendationCard from "./AnalysisRecommendationCard";
import ResultView from "./ResultView";
import AIInsightCard from "./AIInsightCard";
import LoadingState from "../ui/LoadingState";
import ErrorState from "../ui/ErrorState";
import EmptyState from "../ui/EmptyState";

const TYPE_LABELS = {
  trend_analysis: "Trend Analysis", regression: "Regression", clustering: "Clustering",
  forecasting: "Forecasting", hypothesis_test: "Hypothesis Test",
  pca: "PCA", anomaly_detection: "Anomaly Detection",
};

export default function AnalysisWorkspace({ datasetId, investigationGoal, onAnalysisCompleted }) {
  const [recommendations, setRecommendations] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [selectedRec, setSelectedRec] = useState(null);
  const [params, setParams] = useState({});
  const [runResult, setRunResult] = useState(null);
  const [runId, setRunId] = useState(null);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState(null);

  const [insight, setInsight] = useState(null);
  const [insightLoading, setInsightLoading] = useState(false);
  const [insightError, setInsightError] = useState(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getAnalysisRecommendations(datasetId);
      setRecommendations(res.recommendations);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [datasetId]);

  useEffect(() => { refresh(); }, [refresh]);

  // Any recommendation's `columns` field carries the real usable-numeric-column
  // list even when not_recommended — reuse it as the value-column options for
  // trend/forecasting rather than re-deriving it here.
  const numericColumnGuess = useMemo(() => {
    const source = (recommendations ?? []).find((r) => ["regression", "clustering", "anomaly_detection"].includes(r.analysis_type));
    return source?.columns ?? [];
  }, [recommendations]);

  const handleSelect = (rec) => {
    setSelectedRec(rec);
    setRunResult(null);
    setInsight(null);
    setInsightError(null);
    setRunError(null);
    const defaults = {};
    if (rec.analysis_type === "regression") defaults.target = rec.preview?.candidate_targets?.[0];
    if (rec.analysis_type === "hypothesis_test") defaults.outcome_column = rec.preview?.candidate_outcomes?.[0];
    setParams(defaults);
  };

  const handleRun = async () => {
    setRunning(true);
    setRunError(null);
    try {
      const res = await runAnalysis({
        dataset_id: datasetId, analysis_type: selectedRec.analysis_type,
        columns: selectedRec.columns, params,
      });
      setRunResult(res);
      const id = `${selectedRec.analysis_type}:${selectedRec.columns.join(",")}:${Date.now()}`;
      setRunId(id);
      onAnalysisCompleted?.({
        id, analysis_type: res.analysis_type, columns: res.columns,
        result: res.result, summary: res.summary, limitations: res.limitations, ai_insight: null,
      });
    } catch (e) {
      setRunError(e.message);
    } finally {
      setRunning(false);
    }
  };

  const handleGenerateInsight = async () => {
    setInsightLoading(true);
    setInsightError(null);
    try {
      const res = await getAIInsight({
        dataset_id: datasetId, analysis_type: selectedRec.analysis_type,
        columns: selectedRec.columns, result: runResult.result,
        deterministic_summary: runResult.summary, investigation_goal: investigationGoal,
      });
      setInsight(res);
      onAnalysisCompleted?.({
        id: runId, analysis_type: runResult.analysis_type, columns: runResult.columns,
        result: runResult.result, summary: runResult.summary, limitations: runResult.limitations,
        ai_insight: res,
      });
    } catch (e) {
      setInsightError(e.message);
    } finally {
      setInsightLoading(false);
    }
  };

  if (loading && !recommendations) {
    return <LoadingState message="Evaluating which analyses fit this dataset…" />;
  }
  if (error) {
    return <ErrorState message={error} onRetry={refresh} />;
  }
  if (recommendations.length === 0) {
    return <EmptyState message="No analysis techniques could be evaluated for this dataset yet." />;
  }

  const grouped = {};
  for (const rec of recommendations) {
    (grouped[rec.analysis_type] ??= []).push(rec);
  }

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
      <div>
        <p className="mb-3 font-mono text-xs uppercase tracking-wider text-ink-500">
          Analysis techniques considered ({recommendations.length})
        </p>
        <div className="space-y-3">
          {Object.entries(grouped).map(([type, recs]) => (
            <div key={type}>
              <p className="mb-1 font-mono text-[11px] uppercase tracking-wider text-ink-700">{TYPE_LABELS[type]}</p>
              <div className="space-y-2">
                {recs.map((rec) => (
                  <AnalysisRecommendationCard
                    key={rec.id} rec={rec} onRun={handleSelect}
                    isSelected={selectedRec?.id === rec.id}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div>
        {!selectedRec && (
          <EmptyState message="Pick a recommended analysis on the left to configure and run it." />
        )}

        {selectedRec && (
          <div className="space-y-4">
            <div className="rounded-lg border border-graphite-border bg-graphite-800 p-4">
              <p className="font-display text-sm text-ink-100">{selectedRec.title}</p>

              {selectedRec.analysis_type === "regression" && (
                <ParamSelect label="Target column" options={selectedRec.preview?.candidate_targets ?? []}
                  value={params.target} onChange={(v) => setParams({ ...params, target: v })} />
              )}
              {selectedRec.analysis_type === "hypothesis_test" && (
                <ParamSelect label="Outcome column" options={selectedRec.preview?.candidate_outcomes ?? []}
                  value={params.outcome_column} onChange={(v) => setParams({ ...params, outcome_column: v })} />
              )}
              {(selectedRec.analysis_type === "trend_analysis" || selectedRec.analysis_type === "forecasting") && (
                <ParamSelect label="Value column (optional — leave blank for row count)"
                  options={["", ...numericColumnGuess]}
                  value={params.value_column ?? ""} onChange={(v) => setParams({ ...params, value_column: v || undefined })} />
              )}

              <button
                onClick={handleRun}
                disabled={running}
                className="mt-3 rounded border border-signal-amber/50 bg-signal-amber/10 px-4 py-1.5 font-mono text-xs text-signal-amber transition-colors hover:bg-signal-amber/20 disabled:opacity-50"
              >
                {running ? "Running…" : "Run analysis"}
              </button>
              {runError && <div className="mt-2"><ErrorState message={runError} compact /></div>}
            </div>

            {runResult && (
              <div className="rounded-lg border border-graphite-border bg-graphite-800 p-4">
                <p className="font-mono text-[11px] uppercase tracking-wider text-ink-500">Result</p>
                <p className="mt-1 text-sm text-ink-100">{runResult.summary}</p>
                {runResult.limitations.length > 0 && (
                  <ul className="mt-2 list-inside list-disc space-y-0.5 font-mono text-[11px] text-signal-amber/80">
                    {runResult.limitations.map((l, i) => <li key={i}>{l}</li>)}
                  </ul>
                )}
                <div className="mt-3 border-t border-graphite-border pt-3">
                  <ResultView value={runResult.result} />
                </div>

                <button
                  onClick={handleGenerateInsight}
                  disabled={insightLoading}
                  className="mt-3 rounded border border-graphite-border bg-graphite-900 px-4 py-1.5 font-mono text-xs text-ink-300 transition-colors hover:border-signal-amber hover:text-signal-amber disabled:opacity-50"
                >
                  {insightLoading ? "Generating…" : "Generate AI Insight"}
                </button>
                {insightError && <div className="mt-2"><ErrorState message={insightError} compact /></div>}
              </div>
            )}

            {insight && <AIInsightCard insight={insight} />}
          </div>
        )}
      </div>
    </div>
  );
}

function ParamSelect({ label, options, value, onChange }) {
  return (
    <div className="mt-2 flex items-center gap-2">
      <span className="font-mono text-[11px] uppercase tracking-wider text-ink-500">{label}</span>
      <select
        value={value ?? ""}
        onChange={(e) => onChange(e.target.value)}
        className="rounded border border-graphite-border bg-graphite-900 px-2 py-1 font-mono text-[11px] text-ink-300 focus:border-signal-amber focus:outline-none"
      >
        {options.map((o) => <option key={o || "(none)"} value={o}>{o || "(row count)"}</option>)}
      </select>
    </div>
  );
}
