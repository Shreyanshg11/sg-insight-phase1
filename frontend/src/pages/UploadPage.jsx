import { useState, useRef, useEffect } from "react";
import {
  uploadDataset,
  listSamples,
  loadSample,
  exampleReportUrl,
  getAiProviderStatus,
} from "../api/client";
import ErrorState from "../components/ui/ErrorState";
import { describeAiStatus, TONE_DOT_CLASS } from "../lib/aiStatus";

export default function UploadPage({ onUploaded }) {
  const [goal, setGoal] = useState("");
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [samples, setSamples] = useState([]);
  const [sampleLoadingId, setSampleLoadingId] = useState(null);
  const [aiStatus, setAiStatus] = useState({ loading: true });
  const inputRef = useRef(null);

  useEffect(() => {
    listSamples()
      .then((r) => setSamples(r.samples))
      .catch(() => {});
  }, []);

  useEffect(() => {
    let active = true;

    const checkAI = async () => {
      try {
        const result = await getAiProviderStatus();
        if (active) setAiStatus({ loading: false, ...result });
      } catch {
        if (active) {
          setAiStatus({
            loading: false,
            connected: false,
            model_available: false,
            status: "disconnected",
          });
        }
      }
    };

    checkAI();
    const interval = setInterval(checkAI, 30000);

    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  const handleFile = async (file) => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const result = await uploadDataset(file, goal.trim() || null);
      onUploaded(result);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleLoadSample = async (sampleId) => {
    setSampleLoadingId(sampleId);
    setError(null);
    try {
      const result = await loadSample(sampleId);
      onUploaded(result);
    } catch (e) {
      setError(e.message);
    } finally {
      setSampleLoadingId(null);
    }
  };

  const ai = describeAiStatus(aiStatus);

  return (
    <div className="mx-auto max-w-3xl px-4 py-10 sm:px-6 sm:py-20">
      <p className="font-mono text-xs uppercase tracking-widest text-signal-amber">
        SG-Insight
      </p>

      <h1 className="mt-2 font-display text-2xl font-semibold text-ink-100 sm:text-3xl">
        Upload a dataset to begin
      </h1>

      <p className="mt-2 max-w-lg text-sm text-ink-500">
        CSV or Excel. Every cleaning step from here is proposed with a reason
        and a confidence reading. You confirm before anything changes.
      </p>

      <div className="mt-5 flex items-center gap-2 rounded border border-graphite-border bg-graphite-900 px-3 py-2.5">
        <span
          className={`h-2 w-2 shrink-0 rounded-full ${TONE_DOT_CLASS[ai.tone]}`}
        />
        <div className="min-w-0">
          <p className="font-mono text-xs text-ink-200">
            {ai.headline}
          </p>
          <p className="mt-0.5 text-[10px] text-ink-500">
            {ai.detail}
          </p>
        </div>
      </div>

      <label className="mt-8 block text-xs font-medium uppercase tracking-wider text-ink-500">
        Investigation goal{" "}
        <span className="text-ink-700">(optional, but recommended)</span>
      </label>

      <input
        value={goal}
        onChange={(e) => setGoal(e.target.value)}
        placeholder="e.g. find why revenue dropped in Q2"
        className="mt-2 w-full rounded border border-graphite-border bg-graphite-800 px-3 py-2 text-sm text-ink-100 placeholder:text-ink-700 focus:border-signal-amber focus:outline-none"
      />

      <div
        role="button"
        tabIndex={0}
        aria-label="Upload dataset - drop a CSV or Excel file, or press Enter to browse"
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          handleFile(e.dataTransfer.files?.[0]);
        }}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        className={`mt-6 flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed px-4 py-10 text-center transition-colors focus-visible:outline-none sm:px-6 sm:py-14 ${
          dragging
            ? "border-signal-amber bg-graphite-800"
            : "border-graphite-border bg-graphite-900"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".csv,.xlsx,.xls"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
        <p className="font-mono text-sm text-ink-300">
          {loading
            ? "Uploading..."
            : "Drop a .csv or .xlsx file, or click to browse"}
        </p>
      </div>

      {error && (
        <div className="mt-4">
          <ErrorState message={error} onRetry={() => setError(null)} />
        </div>
      )}

      {samples.length > 0 && (
        <div className="mt-10">
          <p className="font-mono text-xs uppercase tracking-wider text-ink-500">
            Or try a sample dataset
          </p>

          <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-3">
            {samples.map((s) => (
              <div
                key={s.id}
                className="rounded-lg border border-graphite-border bg-graphite-900 p-3"
              >
                <p className="font-display text-sm text-ink-100">
                  {s.name}
                </p>

                <p className="mt-1 line-clamp-3 text-[11px] text-ink-500">
                  {s.description}
                </p>

                <div className="mt-2 flex flex-wrap gap-1">
                  {s.highlights.map((h) => (
                    <span
                      key={h}
                      className="rounded-full bg-graphite-700 px-2 py-0.5 font-mono text-[10px] text-ink-300"
                    >
                      {h}
                    </span>
                  ))}
                </div>

                <div className="mt-3 flex gap-2">
                  <button
                    onClick={() => handleLoadSample(s.id)}
                    disabled={sampleLoadingId === s.id}
                    className="flex-1 rounded border border-signal-amber/50 bg-signal-amber/10 px-2 py-1.5 font-mono text-[11px] text-signal-amber hover:bg-signal-amber/20 disabled:opacity-50"
                  >
                    {sampleLoadingId === s.id
                      ? "Loading..."
                      : "Load into workbench"}
                  </button>

                  <a
                    href={exampleReportUrl(s.id)}
                    target="_blank"
                    rel="noreferrer"
                    className="rounded border border-graphite-border px-2 py-1.5 font-mono text-[11px] text-ink-300 hover:bg-graphite-700"
                  >
                    View example report
                  </a>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
