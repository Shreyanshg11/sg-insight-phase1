import { useCallback, useEffect, useMemo, useState } from "react";
import { getMLRecommendations, trainModel } from "../../api/client";
import ErrorState from "../ui/ErrorState";
import LoadingState from "../ui/LoadingState";
import EmptyState from "../ui/EmptyState";
import ResultView from "../analysis/ResultView";

const MODEL_OPTIONS = {
  classification: ["random_forest", "logistic_regression"],
  regression: ["random_forest", "linear_regression"],
};

export default function MLWorkspace({ datasetId, onModelCompleted }) {
  const [recommendations, setRecommendations] = useState(null);
  const [task, setTask] = useState("classification");
  const [target, setTarget] = useState("");
  const [features, setFeatures] = useState([]);
  const [modelType, setModelType] = useState("random_forest");
  const [testSize, setTestSize] = useState("0.2");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [training, setTraining] = useState(false);
  const [error, setError] = useState(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await getMLRecommendations(datasetId);
      setRecommendations(response.recommendations);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [datasetId]);
  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => {
    if (recommendations?.length && !recommendations.some((item) => item.task === task)) {
      setTask(recommendations[0].task);
    }
  }, [recommendations, task]);

  const recommendation = useMemo(
    () => recommendations?.find((item) => item.task === task),
    [recommendations, task],
  );
  const candidates = useMemo(() => recommendation?.target_candidates ?? [], [recommendation]);
  const featureCandidates = useMemo(
    () => (recommendation?.feature_candidates ?? []).filter((name) => name !== target),
    [recommendation, target],
  );
  const selectedFeatures = useMemo(
    () => features.filter((name) => featureCandidates.includes(name)),
    [features, featureCandidates],
  );

  useEffect(() => {
    if (!candidates.includes(target)) setTarget(candidates[0] ?? "");
  }, [candidates, target]);
  useEffect(() => {
    setFeatures((current) => {
      const valid = current.filter((name) => featureCandidates.includes(name));
      return valid.length ? valid : featureCandidates.slice(0, Math.min(5, featureCandidates.length));
    });
    setModelType(MODEL_OPTIONS[task][0]);
    setResult(null);
  }, [task, target, featureCandidates]);

  const toggleFeature = (name) => setFeatures((current) =>
    current.includes(name) ? current.filter((item) => item !== name) : [...current, name]);

  const handleTrain = async () => {
    setTraining(true);
    setError(null);
    setResult(null);
    try {
      const response = await trainModel({
        dataset_id: datasetId, task, target, features: selectedFeatures, model_type: modelType,
        test_size: Number(testSize),
      });
      setResult(response);
      const summary = task === "classification"
        ? `${modelType} achieved ${Math.round(response.metrics.accuracy * 1000) / 10}% accuracy on ${response.test_rows} held-out rows.`
        : `${modelType} achieved RMSE ${response.metrics.rmse} on ${response.test_rows} held-out rows.`;
      onModelCompleted?.({
        id: `ml:${task}:${target}:${Date.now()}`, analysis_type: `${task}_model`,
        columns: [target, ...selectedFeatures], summary, result: response.metrics,
        limitations: response.limitations, ai_insight: null,
      });
    } catch (e) {
      setError(e.message);
    } finally {
      setTraining(false);
    }
  };

  if (loading) return <LoadingState message="Checking suitable supervised models…" />;
  if (error && !recommendations) return <ErrorState message={error} onRetry={refresh} />;
  if (!recommendations?.length) return <EmptyState title="No supervised model fits this dataset" message="Choose a target with enough usable values and numeric or categorical feature columns." />;

  return (
    <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
      <section className="space-y-4 rounded-lg border border-graphite-border bg-graphite-900 p-4">
        <div>
          <h2 className="font-display text-base text-ink-100">Train and evaluate a model</h2>
          <p className="mt-1 text-xs text-ink-500">The workbench holds out test rows and fits imputation and encoding on training rows only. Training does not change your dataset.</p>
        </div>
        <Field label="Task">
          <select value={task} onChange={(event) => setTask(event.target.value)} className={selectClass}>
            {recommendations.map((item) => <option key={item.task} value={item.task}>{item.task}</option>)}
          </select>
        </Field>
        {recommendation ? <>
          <Field label="Target column">
            <select value={target} onChange={(event) => setTarget(event.target.value)} className={selectClass}>
              {candidates.map((name) => <option key={name} value={name}>{name}</option>)}
            </select>
          </Field>
          <Field label="Model">
            <select value={modelType} onChange={(event) => setModelType(event.target.value)} className={selectClass}>
              {MODEL_OPTIONS[task].map((model) => <option key={model} value={model}>{model.replaceAll("_", " ")}</option>)}
            </select>
          </Field>
          <Field label="Held-out test split">
            <select value={testSize} onChange={(event) => setTestSize(event.target.value)} className={selectClass}>
              <option value="0.2">20%</option><option value="0.25">25%</option><option value="0.3">30%</option>
            </select>
          </Field>
          <fieldset>
            <legend className="font-mono text-[11px] uppercase tracking-wider text-ink-500">Feature columns</legend>
            {featureCandidates.length ? <div className="mt-2 grid grid-cols-1 gap-1 sm:grid-cols-2">
              {featureCandidates.map((name) => <label key={name} className="flex min-w-0 items-center gap-2 text-xs text-ink-300">
                <input type="checkbox" checked={selectedFeatures.includes(name)} onChange={() => toggleFeature(name)} className="accent-signal-amber" />
                <span className="truncate">{name}</span>
              </label>)}
            </div> : <p className="mt-1 text-xs text-ink-600">No other suitable feature columns were detected.</p>}
          </fieldset>
          {recommendation.notes.map((note) => <p key={note} className="text-[11px] text-ink-600">{note}</p>)}
          <button onClick={handleTrain} disabled={training || !target || !selectedFeatures.length}
            className="rounded border border-signal-amber/50 bg-signal-amber/10 px-4 py-2 font-mono text-xs text-signal-amber hover:bg-signal-amber/20 disabled:opacity-50">
            {training ? "Training and evaluating…" : "Train and evaluate"}
          </button>
        </> : <EmptyState message={`No ${task} target candidates were detected.`} />}
        {error && <ErrorState message={error} compact />}
      </section>

      <section className="space-y-4">
        {!result && <EmptyState message="Choose a target and features, then run a holdout evaluation. Results will appear here." />}
        {result && <div className="space-y-4 rounded-lg border border-graphite-border bg-graphite-900 p-4">
          <div>
            <p className="font-mono text-xs uppercase tracking-wider text-signal-green">Evaluation complete</p>
            <p className="mt-1 text-sm text-ink-200">{result.model_type.replaceAll("_", " ")} · {result.train_rows} training rows · {result.test_rows} held-out rows</p>
          </div>
          <div className="border-t border-graphite-border pt-3"><ResultView value={result.metrics} /></div>
          {result.cross_validation && (
            <p className="font-mono text-[11px] text-ink-500">
              {result.cross_validation.folds}-fold cross-validation on the training rows ({result.cross_validation.metric}): {result.cross_validation.mean} ± {result.cross_validation.std}
            </p>
          )}
          {result.feature_importance.length > 0 && <div>
            <h3 className="mb-2 font-mono text-[11px] uppercase tracking-wider text-ink-500">Top feature contributions</h3>
            <ResultView value={result.feature_importance} />
          </div>}
          {result.predictions.length > 0 && <div>
            <h3 className="mb-2 font-mono text-[11px] uppercase tracking-wider text-ink-500">Held-out predictions (up to 25)</h3>
            <ResultView value={result.predictions} />
          </div>}
          <ul className="list-inside list-disc space-y-1 text-[11px] text-signal-amber/80">
            {result.limitations.map((item) => <li key={item}>{item}</li>)}
          </ul>
        </div>}
      </section>
    </div>
  );
}

const selectClass = "rounded border border-graphite-border bg-graphite-800 px-2 py-1.5 text-xs text-ink-100 focus:border-signal-amber focus:outline-none";
function Field({ label, children }) {
  return <label className="flex flex-wrap items-center justify-between gap-3 text-xs text-ink-400"><span>{label}</span>{children}</label>;
}
