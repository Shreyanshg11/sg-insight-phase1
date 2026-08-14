import { useState, useEffect, useCallback, useRef } from "react";
import { draftReportPlan, previewReport, generateReport } from "../../api/client";
import SectionOutline from "./SectionOutline";
import LoadingState from "../ui/LoadingState";
import ErrorState from "../ui/ErrorState";

const FORMATS = [
  { value: "markdown", label: "Markdown (.md)" },
  { value: "html", label: "HTML (.html)" },
  { value: "pdf", label: "PDF (.pdf)" },
  { value: "notebook", label: "Jupyter Notebook (.ipynb)" },
];

// Mirrors report_engine.py's SECTION_REGISTRY — the client-side default
// shown before any AI draft runs (or if the user never asks for one). The
// backend is still the source of truth for which keys are actually valid;
// an unrecognized key here would just be silently ignored server-side.
const DEFAULT_SECTIONS = [
  { key: "executive_summary", label: "Executive Summary" },
  { key: "dataset_overview", label: "Dataset Overview" },
  { key: "quality_report", label: "Data Quality" },
  { key: "health_summary", label: "Health Dashboard" },
  { key: "action_log", label: "Cleaning & Feature Engineering Log" },
  { key: "eda_findings", label: "EDA Findings" },
  { key: "deep_analysis", label: "Deep Analysis" },
  { key: "visualizations", label: "Visualizations" },
  { key: "recommendations", label: "Recommendations" },
  { key: "limitations", label: "Limitations" },
  { key: "appendix", label: "Appendix" },
].map((s) => ({ ...s, include: true, reason: "" }));

/**
 * The last pipeline stage, now a Smart Report Builder (session 13): an
 * optional AI draft proposes which sections/analyses to include and why,
 * but every one of those is just a starting point — drag to reorder,
 * untick to exclude, or skip the draft entirely and use the default
 * full-report outline. The live preview (markdown only, for speed) always
 * reflects the current outline exactly as Generate would produce it.
 */
export default function ReportWorkspace({ datasetId, completedAnalyses }) {
  const [format, setFormat] = useState("markdown");
  const [sections, setSections] = useState(DEFAULT_SECTIONS);
  const [selectedIds, setSelectedIds] = useState(() => new Set(completedAnalyses.map((a) => a.id)));

  const [userRequest, setUserRequest] = useState("");
  const [drafting, setDrafting] = useState(false);
  const [draftError, setDraftError] = useState(null);
  const [plan, setPlan] = useState(null);

  const [preview, setPreview] = useState("");
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState(null);
  const debounceRef = useRef(null);

  const [generating, setGenerating] = useState(false);
  const [genError, setGenError] = useState(null);
  const [lastDownloaded, setLastDownloaded] = useState(null);

  const toggleAnalysis = (id) => {
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

  const includedAnalyses = () =>
    completedAnalyses
      .filter((a) => selectedIds.has(a.id))
      .map(({ analysis_type, columns, result, summary, limitations, ai_insight }) => ({
        analysis_type, columns, result, summary, limitations, ai_insight,
      }));

  const handleDraft = async () => {
    setDrafting(true);
    setDraftError(null);
    try {
      const availableAnalyses = completedAnalyses.map((a, i) => ({
        index: i, analysis_type: a.analysis_type, columns: a.columns, summary: a.summary,
      }));
      const res = await draftReportPlan(datasetId, availableAnalyses, userRequest.trim() || null);
      setPlan(res);
      setSections(res.sections.map((s) => ({ ...s })));
      setSelectedIds(new Set(res.selected_analysis_indices.map((i) => completedAnalyses[i]?.id).filter(Boolean)));
    } catch (e) {
      setDraftError(e.message);
    } finally {
      setDrafting(false);
    }
  };

  const refreshPreview = useCallback(() => {
    if (format === "notebook") return;
    setPreviewLoading(true);
    setPreviewError(null);
    previewReport({
      dataset_id: datasetId, format: "markdown",
      included_analyses: includedAnalyses(),
      section_order: sections.filter((s) => s.include).map((s) => s.key),
    })
      .then(setPreview)
      .catch((e) => setPreviewError(e.message))
      .finally(() => setPreviewLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetId, sections, selectedIds, format]);

  useEffect(() => {
    clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(refreshPreview, 400);
    return () => clearTimeout(debounceRef.current);
  }, [refreshPreview]);

  const handleGenerate = async () => {
    setGenerating(true);
    setGenError(null);
    try {
      const payload = {
        dataset_id: datasetId, format, included_analyses: includedAnalyses(),
        section_order: format === "notebook" ? null : sections.filter((s) => s.include).map((s) => s.key),
      };
      const { blob, filename } = await generateReport(payload);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url; a.download = filename;
      document.body.appendChild(a); a.click(); a.remove();
      URL.revokeObjectURL(url);
      setLastDownloaded(filename);
    } catch (e) {
      setGenError(e.message);
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <div>
        <p className="mb-3 font-mono text-xs uppercase tracking-wider text-ink-500">
          Smart Report Builder — draft with AI, then adjust anything below
        </p>

        <div className="rounded-lg border border-graphite-border bg-graphite-800 p-4">
          <div className="flex flex-wrap gap-2">
            <input
              value={userRequest} onChange={(e) => setUserRequest(e.target.value)}
              placeholder="Optional: focus this on… (e.g. regional performance)"
              className="flex-1 rounded border border-graphite-border bg-graphite-900 px-2 py-1.5 font-mono text-xs text-ink-100 placeholder:text-ink-700 focus:border-signal-amber focus:outline-none"
            />
            <button
              onClick={handleDraft} disabled={drafting}
              className="rounded border border-signal-amber/50 bg-signal-amber/10 px-3 py-1.5 font-mono text-xs text-signal-amber hover:bg-signal-amber/20 disabled:opacity-50"
            >
              {drafting ? "Drafting…" : "Draft with AI"}
            </button>
          </div>
          {draftError && <div className="mt-2"><ErrorState message={draftError} onRetry={handleDraft} compact /></div>}
          {plan && !draftError && (
            <div className="mt-2 rounded border border-graphite-border bg-graphite-900 p-2">
              <p className="font-mono text-[11px] text-ink-300">{plan.rationale}</p>
              {!plan.grounded && (
                <p className="mt-1 font-mono text-[10px] text-signal-red">
                  ⚠ unverified: {plan.unverified_figures.map((f) => f.figure).join(", ")}
                </p>
              )}
            </div>
          )}

          <p className="mt-4 font-mono text-[11px] uppercase tracking-wider text-ink-500">
            Sections — drag to reorder, untick to exclude
          </p>
          <div className="mt-2">
            <SectionOutline sections={sections} onChange={setSections} />
          </div>

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
                  <input type="checkbox" checked={selectedIds.has(a.id)} onChange={() => toggleAnalysis(a.id)} className="accent-signal-amber" />
                  {a.analysis_type} ({a.columns.join(", ")}){a.ai_insight && " — with AI insight"}
                </label>
              ))}
            </div>
          )}

          <p className="mt-4 font-mono text-[11px] uppercase tracking-wider text-ink-500">Format</p>
          <div className="mt-2 flex flex-wrap gap-2">
            {FORMATS.map((f) => (
              <button
                key={f.value} onClick={() => setFormat(f.value)}
                className={`rounded border px-3 py-1.5 font-mono text-xs transition-colors ${
                  format === f.value ? "border-signal-amber bg-signal-amber/10 text-signal-amber"
                                     : "border-graphite-border bg-graphite-900 text-ink-300 hover:border-signal-amber/50"
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>
          {format === "notebook" && (
            <p className="mt-2 font-mono text-[11px] text-ink-700">
              The notebook reproduces the cleaning/feature-engineering pipeline as executable code —
              the section outline above doesn't apply to it (it isn't a summary report).
            </p>
          )}

          <button
            onClick={handleGenerate} disabled={generating}
            className="mt-4 rounded border border-signal-amber/50 bg-signal-amber/10 px-4 py-1.5 font-mono text-xs text-signal-amber transition-colors hover:bg-signal-amber/20 disabled:opacity-50"
          >
            {generating ? "Generating…" : "Generate & Download"}
          </button>
          {genError && <div className="mt-2"><ErrorState message={genError} onRetry={handleGenerate} compact /></div>}
          {lastDownloaded && !genError && (
            <p className="mt-2 font-mono text-xs text-signal-green">Downloaded {lastDownloaded}</p>
          )}
        </div>
      </div>

      <div>
        <p className="mb-3 font-mono text-xs uppercase tracking-wider text-ink-500">
          Live preview {format === "notebook" && <span className="text-ink-700">(not available for notebooks — download to view)</span>}
        </p>
        <div className="max-h-[70vh] overflow-y-auto rounded-lg border border-graphite-border bg-graphite-900 p-4">
          {format === "notebook" ? (
            <p className="font-mono text-xs text-ink-700">Switch to another format to preview, or Generate to download the notebook directly.</p>
          ) : previewLoading && !preview ? (
            <LoadingState message="Building preview…" compact />
          ) : previewError ? (
            <ErrorState message={previewError} onRetry={refreshPreview} compact />
          ) : (
            <pre className="whitespace-pre-wrap font-mono text-[11px] leading-relaxed text-ink-300">{preview}</pre>
          )}
        </div>
      </div>
    </div>
  );
}
