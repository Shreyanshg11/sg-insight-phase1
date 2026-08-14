import { useState, useEffect, useCallback } from "react";
import { useFilters } from "../../context/FilterContext";
import { getFilteredRows } from "../../api/client";
import LoadingState from "../ui/LoadingState";
import ErrorState from "../ui/ErrorState";

const PAGE_SIZE = 20;

export default function MatchingRowsPanel({ datasetId }) {
  const { filters, drillDownOpen, setDrillDownOpen } = useFilters();
  const [page, setPage] = useState(1);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    getFilteredRows(datasetId, filters, page, PAGE_SIZE)
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [datasetId, filters, page]);

  useEffect(() => {
    if (drillDownOpen) load();
  }, [drillDownOpen, load]);

  useEffect(() => { setPage(1); }, [filters]);

  if (!drillDownOpen) return null;

  const totalPages = data ? Math.max(1, Math.ceil(data.total_matching / PAGE_SIZE)) : 1;

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-graphite-950/70 sm:items-center" onClick={() => setDrillDownOpen(false)}>
      <div
        className="max-h-[80vh] w-full max-w-4xl overflow-hidden rounded-t-lg border border-graphite-border bg-graphite-900 sm:rounded-lg"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-graphite-border px-4 py-3">
          <p className="font-mono text-xs uppercase tracking-wider text-ink-500">
            Matching rows {data && <span className="text-ink-100">— {data.total_matching} of {data.total_rows}</span>}
          </p>
          <button onClick={() => setDrillDownOpen(false)} aria-label="Close" className="font-mono text-ink-500 hover:text-ink-100">✕</button>
        </div>

        <div className="max-h-[60vh] overflow-auto p-4">
          {loading && <LoadingState message="Loading matching rows…" />}
          {error && <ErrorState message={error} onRetry={load} />}
          {data && !loading && data.rows.length === 0 && (
            <p className="font-mono text-sm text-ink-500">No rows match this filter.</p>
          )}
          {data && !loading && data.rows.length > 0 && (
            <table className="w-full border-collapse text-left font-mono text-[11px]">
              <thead>
                <tr className="border-b border-graphite-border">
                  {data.columns.map((c) => (
                    <th key={c} className="whitespace-nowrap px-2 py-1.5 text-ink-500">{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.rows.map((row, i) => (
                  <tr key={i} className="border-b border-graphite-border/50">
                    {data.columns.map((c) => (
                      <td key={c} className="whitespace-nowrap px-2 py-1.5 text-ink-300">
                        {row[c] === null || row[c] === undefined ? <span className="text-ink-700">—</span> : String(row[c])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {data && totalPages > 1 && (
          <div className="flex items-center justify-between border-t border-graphite-border px-4 py-2">
            <button onClick={() => setPage((p) => Math.max(1, p - 1))} disabled={page <= 1}
              className="rounded border border-graphite-border px-2 py-1 font-mono text-[11px] text-ink-300 hover:bg-graphite-700 disabled:opacity-40">
              Prev
            </button>
            <span className="font-mono text-[11px] text-ink-500">Page {page} of {totalPages}</span>
            <button onClick={() => setPage((p) => Math.min(totalPages, p + 1))} disabled={page >= totalPages}
              className="rounded border border-graphite-border px-2 py-1 font-mono text-[11px] text-ink-300 hover:bg-graphite-700 disabled:opacity-40">
              Next
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
