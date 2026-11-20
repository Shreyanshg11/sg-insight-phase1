import { useEffect, useState } from "react";
import { getDatasetIntelligence, getAiProviderStatus } from "../../api/client";
import { getAiProviderLabel } from "../../lib/aiStatus";

const number = (value, digits = 2) => {
  if (value === null || value === undefined || value === "") return "—";
  const parsed = Number(value);
  return Number.isFinite(parsed)
    ? parsed.toLocaleString(undefined, { maximumFractionDigits: digits })
    : String(value);
};

const label = (value) =>
  String(value || "Unknown")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());

function entries(value) {
  if (Array.isArray(value)) {
    return value.map((item, index) => [String(index), item]);
  }
  if (value && typeof value === "object") {
    return Object.entries(value);
  }
  return [];
}

function rowsFrom(value) {
  if (Array.isArray(value)) return value;
  if (!value || typeof value !== "object") return [];

  for (const key of ["top_segments", "top_values", "top_categories", "distribution", "values", "items"]) {
    if (Array.isArray(value[key])) return value[key];
  }

  return [];
}

function Section({ title, children }) {
  return (
    <section className="mt-5 rounded-lg border border-graphite-border bg-graphite-800 p-4">
      <h3 className="mb-3 font-mono text-xs uppercase tracking-wider text-signal-amber">
        {title}
      </h3>
      {children}
    </section>
  );
}

function JsonDetails({ title, value }) {
  if (!value || (typeof value === "object" && !Object.keys(value).length)) return null;

  return (
    <details className="mt-3 rounded-lg border border-graphite-border p-3">
      <summary className="cursor-pointer font-mono text-xs text-ink-300">
        {title}
      </summary>
      <pre className="mt-3 overflow-x-auto whitespace-pre-wrap break-words text-xs text-ink-400">
        {typeof value === "string" ? value : JSON.stringify(value, null, 2)}
      </pre>
    </details>
  );
}

function MetricCard({ label: title, value, detail }) {
  return (
    <div className="rounded-lg border border-graphite-border bg-graphite-900 p-3">
      <p className="font-mono text-[10px] uppercase text-ink-500">{title}</p>
      <p className="mt-1 text-xl text-ink-100 tabular-nums">{value ?? "—"}</p>
      {detail && <p className="mt-1 text-xs text-ink-500">{detail}</p>}
    </div>
  );
}

export default function EDADatasetIntelligence({ datasetId }) {
  const [goal, setGoal] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [aiProviderStatus, setAiProviderStatus] = useState({ loading: true });

  useEffect(() => {
    let active = true;
    getAiProviderStatus()
      .then((status) => {
        if (active) setAiProviderStatus(status);
      })
      .catch(() => {
        if (active) setAiProviderStatus(null);
      });
    return () => {
      active = false;
    };
  }, []);

  async function runIntelligence() {
    if (!datasetId || loading) return;

    setLoading(true);
    setError("");

    try {
      const data = await getDatasetIntelligence(datasetId, goal.trim());
      setResult(data);
    } catch (err) {
      setError(err.message || "Dataset Intelligence request failed.");
    } finally {
      setLoading(false);
    }
  }

  const findings = result?.findings || {};
  const profile = result?.dataset_profile || {};
  const business = findings.business_insights || {};

  const cards = [
    { label: "Rows", value: findings.rows },
    { label: "Columns", value: findings.columns },
    { label: "Missing Values", value: findings.total_missing_values },
    { label: "Duplicate Rows", value: findings.duplicate_rows },
    { label: "Numeric Columns", value: findings.numeric_column_count },
    { label: "Categorical Columns", value: findings.categorical_column_count },
    { label: "Outlier Occurrences", value: findings.total_outlier_column_occurrences },
  ];

  const segments = entries(business.segment_intelligence);
  const spending = entries(business.spending_intelligence);
  const comparisons = Array.isArray(business.category_numeric_comparisons)
    ? business.category_numeric_comparisons
    : [];
  const opportunities = Array.isArray(business.business_opportunities)
    ? business.business_opportunities
    : [];

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-graphite-border bg-graphite-800 p-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 className="font-display text-lg text-ink-100">Dataset Intelligence</h2>
          <span className="rounded border border-graphite-border px-2 py-1 font-mono text-[10px] uppercase text-ink-400">
{getAiProviderLabel(aiProviderStatus)}
          </span>
        </div>

        <p className="mt-2 text-sm text-ink-400">
          Explore your dataset using deterministic Pandas profiling and AI-assisted business interpretation.
        </p>

        <label className="mt-4 block font-mono text-xs text-ink-400">
          Investigation goal (optional)
        </label>
        <textarea
          value={goal}
          onChange={(e) => setGoal(e.target.value)}
          placeholder="Example: Identify customer behaviour patterns and unusual purchasing activity..."
          rows={3}
          disabled={loading}
          className="mt-2 w-full rounded-lg border border-graphite-border bg-graphite-900 p-3 text-sm text-ink-100 outline-none focus:border-signal-amber disabled:opacity-50"
        />

        <div className="mt-3 flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={runIntelligence}
            disabled={loading || !datasetId}
            className="rounded-lg bg-signal-amber px-4 py-2 font-mono text-xs font-semibold text-graphite-950 transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? "Analysing dataset..." : result ? "Run Again" : "Run Dataset Intelligence"}
          </button>
          {loading && (
            <span className="text-xs text-ink-400">
              AI-assisted interpretation may take a few minutes.
            </span>
          )}
        </div>
      </div>

      {error && (
        <div role="alert" className="rounded-lg border border-red-500/40 bg-red-500/10 p-4 text-sm text-red-300">
          <p>{error}</p>
          <button
            type="button"
            onClick={runIntelligence}
            disabled={loading}
            className="mt-2 font-mono text-xs underline"
          >
            Retry
          </button>
        </div>
      )}

      {result && (
        <>
          <Section title="Verified Dataset Findings">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
              {cards.map((card) => (
                <MetricCard key={card.label} {...card} />
              ))}
            </div>
            <JsonDetails title="Outlier counts by column (1.5× IQR)" value={findings.outlier_columns} />
            <JsonDetails title="Missing values by column" value={findings.missing_columns} />
            <p className="mt-3 text-xs leading-relaxed text-ink-400">
              Outlier counts represent column-level IQR occurrences, not necessarily unique rows or data-entry errors.
            </p>
          </Section>

          <Section title="Customer Segment Intelligence">
            {segments.length === 0 ? (
              <p className="text-sm text-ink-400">No segment distribution findings are available.</p>
            ) : (
              <div className="space-y-4">
                {segments.map(([column, data]) => {
                  const segmentRows = rowsFrom(data);
                  return (
                    <div key={column}>
                      <h4 className="mb-2 text-sm font-semibold text-ink-200">
                        {label(column)}
                      </h4>
                      {segmentRows.length > 0 ? (
                        <div className="overflow-x-auto">
                          <table className="w-full text-left text-xs">
                            <thead className="text-ink-500">
                              <tr>
                                <th className="py-2 pr-3">Category</th>
                                <th className="py-2 pr-3">Count</th>
                                <th className="py-2">Share</th>
                              </tr>
                            </thead>
                            <tbody>
                              {segmentRows.map((item, index) => (
                                <tr key={`${column}-${index}`} className="border-t border-graphite-border text-ink-300">
                                  <td className="py-2 pr-3">{item.value ?? item.category ?? item.name ?? item.label ?? "—"}</td>
                                  <td className="py-2 pr-3 tabular-nums">{number(item.count)}</td>
                                  <td className="py-2 tabular-nums">{item.share_pct == null ? "—" : `${number(item.share_pct)}%`}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      ) : (
                        <p className="text-xs text-ink-500">Distribution details are not available in a recognised format.</p>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </Section>

          <Section title="Spending Analysis">
            {spending.length === 0 ? (
              <p className="text-sm text-ink-400">No recognised spending metrics were identified.</p>
            ) : (
              <div className="space-y-4">
                {spending.map(([metric, stats]) => (
                  <div key={metric} className="rounded-lg border border-graphite-border bg-graphite-900 p-3">
                    <h4 className="mb-3 text-sm font-semibold text-ink-100">{label(metric)}</h4>
                    <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                      {["count", "mean", "median", "min", "max", "q1", "q3", "skewness", "outlier_count"].map((key) => (
                        stats?.[key] !== undefined && (
                          <MetricCard
                            key={key}
                            label={label(key)}
                            value={number(stats[key])}
                          />
                        )
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
            <p className="mt-3 text-xs text-ink-500">
              Spending metrics are selected by recognised column names. Outliers and skewness are descriptive indicators, not proof of errors or fraud.
            </p>
          </Section>

          <Section title="Category Comparisons">
            {comparisons.length === 0 ? (
              <p className="text-sm text-ink-400">No eligible category comparisons are available.</p>
            ) : (
              <div className="space-y-3">
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead className="text-ink-500">
                      <tr>
                        <th className="py-2 pr-3">Category</th>
                        <th className="py-2 pr-3">Metric</th>
                        <th className="py-2">Minimum group size</th>
                      </tr>
                    </thead>
                    <tbody>
                      {comparisons.map((item, index) => (
                        <tr key={`${item.category_column}-${item.metric_column}-${index}`} className="border-t border-graphite-border text-ink-300">
                          <td className="py-2 pr-3">{item.category_column || "—"}</td>
                          <td className="py-2 pr-3">{item.metric_column || "—"}</td>
                          <td className="py-2 tabular-nums">{number(item.minimum_group_size, 0)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {comparisons.map((item, index) => {
                  const groups = Array.isArray(item.groups) ? item.groups : [];

                  return (
                    <article
                      key={`comparison-${index}`}
                      className="rounded-lg border border-graphite-border bg-graphite-900 p-3"
                    >
                      <div className="mb-3">
                        <h4 className="text-sm font-semibold text-ink-100">
                          {item.category_column || "Category"} · {item.metric_column || "Metric"}
                        </h4>
                        <p className="mt-1 text-xs text-ink-500">
                          Minimum group size: {number(item.minimum_group_size, 0)}
                        </p>
                      </div>

                      {groups.length === 0 ? (
                        <p className="text-xs text-ink-400">
                          No group statistics available.
                        </p>
                      ) : (
                        <div className="overflow-x-auto">
                          <table className="w-full text-left text-xs">
                            <thead className="text-ink-500">
                              <tr>
                                <th className="py-2 pr-3">Group</th>
                                <th className="py-2 pr-3">Count</th>
                                <th className="py-2 pr-3">Mean</th>
                                <th className="py-2">Median</th>
                              </tr>
                            </thead>
                            <tbody>
                              {groups.map((group, groupIndex) => (
                                <tr
                                  key={`${group.category}-${groupIndex}`}
                                  className="border-t border-graphite-border text-ink-300"
                                >
                                  <td className="py-2 pr-3 font-medium text-ink-100">
                                    {group.category ?? "—"}
                                  </td>
                                  <td className="py-2 pr-3 tabular-nums">
                                    {number(group.count, 0)}
                                  </td>
                                  <td className="py-2 pr-3 tabular-nums">
                                    {number(group.mean)}
                                  </td>
                                  <td className="py-2 tabular-nums">
                                    {number(group.median)}
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      )}
                    </article>
                  );
                })}
              </div>
            )}
          </Section>

          <Section title="Business Opportunities">
            {opportunities.length === 0 ? (
              <p className="text-sm text-ink-400">No rule-based business opportunities were identified.</p>
            ) : (
              <div className="space-y-3">
                {opportunities.map((item, index) => (
                  <article key={`${item.type}-${item.column}-${index}`} className="rounded-lg border border-graphite-border bg-graphite-900 p-3">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <h4 className="text-sm font-semibold text-ink-100">
                        {label(item.type)}
                      </h4>
                      {item.column && (
                        <span className="rounded border border-graphite-border px-2 py-1 font-mono text-[10px] text-signal-amber">
                          {item.column}
                        </span>
                      )}
                    </div>
                    {item.evidence && (
                      <div className="mt-3 grid grid-cols-2 gap-2">
                        {entries(item.evidence).map(([key, value]) => (
                          <MetricCard
                            key={key}
                            label={label(key)}
                            value={typeof value === "number" ? number(value) : String(value)}
                          />
                        ))}
                      </div>
                    )}
                    {item.recommendation && (
                      <p className="mt-3 text-sm leading-relaxed text-ink-200">
                        <strong className="text-signal-amber">Suggested action: </strong>
                        {item.recommendation}
                      </p>
                    )}
                    {item.caveat && (
                      <p className="mt-2 text-xs leading-relaxed text-ink-500">
                        <strong>Caveat: </strong>{item.caveat}
                      </p>
                    )}
                  </article>
                ))}
              </div>
            )}
          </Section>

          <Section title="Dataset Observation">
            <p className="text-sm leading-relaxed text-ink-200">{result.observation || "No observation available."}</p>
          </Section>

          <Section title="Business Interpretation">
            <p className="text-sm leading-relaxed text-ink-200">
              {result.business_meaning || "AI-generated business interpretation is unavailable. Refer to deterministic findings."}
            </p>
          </Section>

          {result.possible_explanations?.length > 0 && (
            <Section title="Possible Explanations">
              <ul className="list-disc space-y-2 pl-5 text-sm text-ink-300">
                {result.possible_explanations.map((item, index) => (
                  <li key={index}>{item}</li>
                ))}
              </ul>
            </Section>
          )}

          {result.limitations?.length > 0 && (
            <Section title="Limitations">
              <ul className="list-disc space-y-2 pl-5 text-sm text-ink-300">
                {result.limitations.map((item, index) => (
                  <li key={index}>{item}</li>
                ))}
              </ul>
            </Section>
          )}

          <Section title="Reliability & Verification">
            <div className="flex flex-wrap gap-3 text-xs">
              <span className="text-ink-300">
                Validation status: <strong>{result.grounded ? "No issues detected" : "Review required"}</strong>
              </span>
              <span className="text-ink-300">
                Checks: <strong>Numeric + semantic</strong>
              </span>
              <span className="text-ink-300">
                Model: <strong>{result.model || "Unknown"}</strong>
              </span>
            </div>

            {result.unverified_figures?.length > 0 && (
              <JsonDetails title="Unverified claims" value={result.unverified_figures} />
            )}

            {!result.grounded && (
              <p className="mt-3 text-xs text-amber-400">
                Some numerical or semantic claims could not be verified. Review flagged claims before relying on the interpretation.
              </p>
            )}

            {result.grounded && (
              <p className="mt-3 text-xs text-ink-400">
                No issues were detected by current validation checks. This does not guarantee that every AI interpretation is factually correct.
              </p>
            )}
          </Section>

          <JsonDetails title="Complete verified dataset profile" value={profile} />
          <JsonDetails title="Evidence supplied to the AI" value={result.evidence} />

          <p className="text-xs text-ink-500">
            Read-only analysis. No dataset values were modified.
            {findings.read_only ? " Read-only verification: passed." : ""}
          </p>
        </>
      )}
    </div>
  );
}



