import { useState } from "react";
import { exportVersionUrl } from "../api/client";
import ErrorState from "./ui/ErrorState";

function formatTimestamp(iso) {
  if (!iso) return null;
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

function CompareResult({ result }) {
  return (
    <div className="mt-3 rounded border border-graphite-border bg-graphite-900 p-3">
      <div className="flex flex-wrap gap-x-6 gap-y-1 font-mono text-xs text-ink-300">
        <span>Rows: {result.rows_a} → {result.rows_b} ({result.row_delta >= 0 ? "+" : ""}{result.row_delta})</span>
        <span>Columns: {result.columns_a} → {result.columns_b}</span>
        {result.columns_added.length > 0 && <span className="text-signal-green">+ {result.columns_added.join(", ")}</span>}
        {result.columns_removed.length > 0 && <span className="text-signal-red">− {result.columns_removed.join(", ")}</span>}
      </div>

      {result.column_changes.length === 0 ? (
        <p className="mt-2 text-xs text-ink-500">No cell-level changes detected in common columns.</p>
      ) : (
        <div className="mt-2 space-y-2">
          {result.column_changes.map((cc) => (
            <div key={cc.column} className="rounded border border-graphite-border bg-graphite-800 p-2">
              <p className="font-mono text-xs text-ink-100">
                '{cc.column}' — {cc.cells_changed} cell(s) changed
              </p>
              <div className="mt-1 space-y-0.5">
                {cc.samples.map((s, i) => (
                  <p key={i} className="font-mono text-[11px] text-ink-500">
                    row {s.row}: <span className="text-ink-300">{String(s.before)}</span> → <span className="text-signal-green">{String(s.after)}</span>
                  </p>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
      <p className="mt-2 font-mono text-[10px] italic text-ink-700">{result.note}</p>
    </div>
  );
}

export default function VersionHistory({ datasetId, versions, currentVersion, onRestore, onCompare }) {
  const [compareA, setCompareA] = useState(versions[0]?.version ?? 1);
  const [compareB, setCompareB] = useState(currentVersion);
  const [compareResult, setCompareResult] = useState(null);
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareError, setCompareError] = useState(null);
  const [confirmingRestore, setConfirmingRestore] = useState(null);

  const handleCompare = async () => {
    setCompareLoading(true);
    setCompareResult(null);
    setCompareError(null);
    try {
      const result = await onCompare(compareA, compareB);
      setCompareResult(result);
    } catch (e) {
      setCompareError(e.message);
    } finally {
      setCompareLoading(false);
    }
  };

  return (
    <div className="rounded-lg border border-graphite-border bg-graphite-800 p-4">
      <p className="font-mono text-xs uppercase tracking-wider text-ink-500">Dataset Versions</p>

      <div className="mt-3 space-y-2">
        {versions.map((v) => (
          <div key={v.version} className={`rounded border px-3 py-2 ${v.version === currentVersion ? "border-signal-amber/40 bg-graphite-900" : "border-graphite-border bg-graphite-900"}`}>
            <div className="flex flex-wrap items-center justify-between gap-1">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-mono text-xs font-medium text-ink-100">{v.label}</span>
                {v.version === currentVersion && (
                  <span className="rounded-full bg-signal-amber/15 px-1.5 py-0.5 font-mono text-[10px] uppercase text-signal-amber">current</span>
                )}
                {formatTimestamp(v.timestamp) && <span className="font-mono text-[11px] text-ink-700">{formatTimestamp(v.timestamp)}</span>}
              </div>
              <span className="font-mono text-[11px] text-ink-500">{v.rows} rows · {v.columns} cols</span>
            </div>
            {v.result_summary && <p className="mt-1 text-xs text-ink-300">{v.result_summary}</p>}

            <div className="mt-2 flex flex-wrap items-center gap-2">
              <a
                href={exportVersionUrl(datasetId, v.version)}
                download
                className="rounded border border-graphite-border px-2 py-1 font-mono text-[11px] text-ink-300 hover:bg-graphite-700"
              >
                Export CSV
              </a>
              <a href={exportVersionUrl(datasetId, v.version, "xlsx")} download className="rounded border border-graphite-border px-2 py-1 font-mono text-[11px] text-ink-300 hover:bg-graphite-700">Export XLSX</a>
              {v.version !== currentVersion && (
                confirmingRestore === v.version ? (
                  <>
                    <span className="font-mono text-[11px] text-signal-red">Discards V{v.version + 1}+. Confirm?</span>
                    <button
                      onClick={() => { onRestore(v.version); setConfirmingRestore(null); }}
                      className="rounded bg-signal-red px-2 py-1 font-mono text-[11px] text-graphite-950 hover:bg-signal-red/90"
                    >
                      Yes, restore
                    </button>
                    <button onClick={() => setConfirmingRestore(null)} className="font-mono text-[11px] text-ink-500 hover:text-ink-300">
                      Cancel
                    </button>
                  </>
                ) : (
                  <button
                    onClick={() => setConfirmingRestore(v.version)}
                    className="rounded border border-graphite-border px-2 py-1 font-mono text-[11px] text-ink-300 hover:bg-graphite-700"
                  >
                    Restore
                  </button>
                )
              )}
            </div>
          </div>
        ))}
      </div>

      <div className="mt-4 border-t border-graphite-border pt-3">
        <p className="font-mono text-[11px] uppercase tracking-wider text-ink-500">Compare versions</p>
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <select value={compareA} onChange={(e) => setCompareA(Number(e.target.value))}
                  className="rounded border border-graphite-border bg-graphite-900 px-2 py-1 font-mono text-xs text-ink-300 focus:border-signal-amber focus:outline-none">
            {versions.map((v) => <option key={v.version} value={v.version}>{v.label}</option>)}
          </select>
          <span className="font-mono text-xs text-ink-500">vs</span>
          <select value={compareB} onChange={(e) => setCompareB(Number(e.target.value))}
                  className="rounded border border-graphite-border bg-graphite-900 px-2 py-1 font-mono text-xs text-ink-300 focus:border-signal-amber focus:outline-none">
            {versions.map((v) => <option key={v.version} value={v.version}>{v.label}</option>)}
          </select>
          <button
            onClick={handleCompare}
            disabled={compareLoading}
            className="rounded bg-signal-amber px-3 py-1 font-mono text-xs font-medium text-graphite-950 hover:bg-signal-amber/90 disabled:opacity-50"
          >
            {compareLoading ? "Comparing…" : "Compare"}
          </button>
        </div>
        {compareError && <div className="mt-2"><ErrorState message={compareError} onRetry={handleCompare} compact /></div>}
        {compareResult && <CompareResult result={compareResult} />}
      </div>
    </div>
  );
}
