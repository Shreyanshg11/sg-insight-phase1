import { useState } from "react";
import { generateReport } from "../../api/client";
import ErrorState from "../ui/ErrorState";

const FORMATS = [
  { value: "markdown", label: "Markdown (.md)" },
  { value: "html", label: "HTML (.html)" },
  { value: "pdf", label: "PDF (.pdf)" },
  { value: "notebook", label: "Jupyter Notebook (.ipynb)" },
];

/**
 * The last pipeline stage — composes a report from everything already done
 * in this session (profile, quality, the cleaning/feature-engineering log,
 * EDA findings, pinned Dashboard charts) plus whichever analyses the user
 * actually ran in the Analysis tab and chooses to include here. Never
 * re-runs or invents an analysis on its own.
 */
export default function ReportWorkspace({ datasetId, completedAnalyses }) {
  const [format, setFormat] = useState("markdown");
  const [selectedIds, setSelectedIds] = useState(() => new Set(completedAnalyses.map((a) => a.id)));
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState(null);
  const [lastDownloaded, setLastDownloaded] = useState(null);

  const toggle = (id) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const handleGenerate = async () => {
    setGenerating(true);
    setError(null);
    try {
      const included = completedAnalyses
        .filter((a) => selectedIds.has(a.id))
        .map(({ analysis_type, columns, result, summary, limitations, ai_insight }) => ({
          analysis_type, columns, result, summary, limitations, ai_insight,
        }));
      const { blob, filename } = await generateReport({ dataset_id: datasetId, format, included_analyses: included });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url; a.download = filename;
      document.body.appendChild(a); a.click(); a.remove();
      URL.revokeObjectURL(url);
      setLastDownloaded(filename);
    } catch (e) {
      setError(e.message);
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className="max-w-xl">
      <p className="mb-3 font-mono text-xs uppercase tracking-wider text-ink-500">
        Generate report — composed from everything done in this session so far
      </p>

      <div className="rounded-lg border border-graphite-border bg-graphite-800 p-4">
        <p className="font-mono text-[11px] uppercase tracking-wider text-ink-500">Format</p>
        <div className="mt-2 flex gap-2">
          {FORMATS.map((f) => (
            <button
              key={f.value}
              onClick={() => setFormat(f.value)}
              className={`rounded border px-3 py-1.5 font-mono text-xs transition-colors ${
                format === f.value
                  ? "border-signal-amber bg-signal-amber/10 text-signal-amber"
                  : "border-graphite-border bg-graphite-900 text-ink-300 hover:border-signal-amber/50"
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>

        {format === "notebook" ? (
          <p className="mt-4 font-mono text-[11px] text-ink-700">
            Reproduces the cleaning/feature-engineering pipeline as a real, executable Jupyter
            notebook — one cell per confirmed action, in order, with the reason and confidence
            you saw at the time. Doesn't include EDA/Analysis/Visualization (those were
            exploratory, not a replayable pipeline) or the analyses selected below.
          </p>
        ) : (
          <>
            <p className="mt-4 font-mono text-[11px] uppercase tracking-wider text-ink-500">
              Analyses to include ({completedAnalyses.length} run this session)
            </p>
            {completedAnalyses.length === 0 ? (
              <p className="mt-1 text-xs text-ink-700">
                None yet — anything you run in the Analysis tab shows up here to include or leave out.
              </p>
            ) : (
              <div className="mt-2 space-y-1.5">
                {completedAnalyses.map((a) => (
                  <label key={a.id} className="flex items-center gap-2 text-xs text-ink-300">
                    <input
                      type="checkbox"
                      checked={selectedIds.has(a.id)}
                      onChange={() => toggle(a.id)}
                      className="accent-signal-amber"
                    />
                    {a.analysis_type} ({a.columns.join(", ")}){a.ai_insight && " — with AI insight"}
                  </label>
                ))}
              </div>
            )}

            <p className="mt-4 font-mono text-[11px] text-ink-700">
              Also includes: dataset overview, data quality report, the full cleaning/feature-engineering
              log, EDA findings, every pinned Dashboard chart, and outstanding recommendations —
              all pulled from what's already been done in this session, nothing recomputed from scratch.
            </p>
          </>
        )}

        <button
          onClick={handleGenerate}
          disabled={generating}
          className="mt-4 rounded border border-signal-amber/50 bg-signal-amber/10 px-4 py-1.5 font-mono text-xs text-signal-amber transition-colors hover:bg-signal-amber/20 disabled:opacity-50"
        >
          {generating ? "Generating…" : "Generate & Download"}
        </button>
        {error && <div className="mt-2"><ErrorState message={error} onRetry={handleGenerate} compact /></div>}
        {lastDownloaded && !error && (
          <p className="mt-2 font-mono text-xs text-signal-green">Downloaded {lastDownloaded}</p>
        )}
      </div>
    </div>
  );
}
