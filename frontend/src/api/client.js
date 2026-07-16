const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function handle(res) {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}

export async function uploadDataset(file, investigationGoal) {
  const form = new FormData();
  form.append("file", file);
  if (investigationGoal) form.append("investigation_goal", investigationGoal);
  const res = await fetch(`${BASE_URL}/api/dataset/upload`, { method: "POST", body: form });
  return handle(res);
}

export async function getProfile(datasetId) {
  const res = await fetch(`${BASE_URL}/api/dataset/${datasetId}/profile`);
  return handle(res);
}

export async function getQuality(datasetId) {
  const res = await fetch(`${BASE_URL}/api/dataset/${datasetId}/quality`);
  return handle(res);
}

export async function applyAction(payload) {
  const res = await fetch(`${BASE_URL}/api/dataset/apply-action`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handle(res);
}

export async function undoAction(datasetId) {
  const res = await fetch(`${BASE_URL}/api/dataset/${datasetId}/undo`, { method: "POST" });
  return handle(res);
}

export async function getHistory(datasetId) {
  const res = await fetch(`${BASE_URL}/api/dataset/${datasetId}/history`);
  return handle(res);
}

export async function getWarnings(datasetId) {
  const res = await fetch(`${BASE_URL}/api/dataset/${datasetId}/warnings`);
  return handle(res);
}

export async function getHealth(datasetId) {
  const res = await fetch(`${BASE_URL}/api/dataset/${datasetId}/health`);
  return handle(res);
}

export async function compareMethods(datasetId, column) {
  const res = await fetch(`${BASE_URL}/api/dataset/${datasetId}/compare-methods?column=${encodeURIComponent(column)}`);
  return handle(res);
}

export async function getVersions(datasetId) {
  const res = await fetch(`${BASE_URL}/api/dataset/${datasetId}/versions`);
  return handle(res);
}

export async function compareVersions(datasetId, a, b) {
  const res = await fetch(`${BASE_URL}/api/dataset/${datasetId}/versions/compare?a=${a}&b=${b}`);
  return handle(res);
}

export async function restoreVersion(datasetId, versionNumber) {
  const res = await fetch(`${BASE_URL}/api/dataset/${datasetId}/versions/${versionNumber}/restore`, { method: "POST" });
  return handle(res);
}

export function exportVersionUrl(datasetId, versionNumber) {
  return `${BASE_URL}/api/dataset/${datasetId}/versions/${versionNumber}/export`;
}

// ---------- Feature Engineering Center (#12) ----------

export async function getFeatureRecommendations(datasetId) {
  const res = await fetch(`${BASE_URL}/api/dataset/${datasetId}/feature-recommendations`);
  return handle(res);
}

export async function applyFeatureAction(payload) {
  const res = await fetch(`${BASE_URL}/api/dataset/feature-action`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handle(res);
}

// ---------- EDA module navigation (#13) ----------

export async function getEDAOverview(datasetId) {
  const res = await fetch(`${BASE_URL}/api/dataset/${datasetId}/eda/overview`);
  return handle(res);
}

export async function getEDADistribution(datasetId, column) {
  const res = await fetch(`${BASE_URL}/api/dataset/${datasetId}/eda/distribution?column=${encodeURIComponent(column)}`);
  return handle(res);
}

export async function getEDACorrelation(datasetId) {
  const res = await fetch(`${BASE_URL}/api/dataset/${datasetId}/eda/correlation`);
  return handle(res);
}

export async function getEDAMissingPatterns(datasetId) {
  const res = await fetch(`${BASE_URL}/api/dataset/${datasetId}/eda/missing-patterns`);
  return handle(res);
}

export async function getEDARelationships(datasetId, columnX, columnY) {
  const res = await fetch(
    `${BASE_URL}/api/dataset/${datasetId}/eda/relationships?column_x=${encodeURIComponent(columnX)}&column_y=${encodeURIComponent(columnY)}`
  );
  return handle(res);
}

export async function getEDACategoryAnalysis(datasetId, column) {
  const res = await fetch(`${BASE_URL}/api/dataset/${datasetId}/eda/category-analysis?column=${encodeURIComponent(column)}`);
  return handle(res);
}

export async function getEDATimeAnalysis(datasetId, column) {
  const res = await fetch(`${BASE_URL}/api/dataset/${datasetId}/eda/time-analysis?column=${encodeURIComponent(column)}`);
  return handle(res);
}

// ---------- Analysis Recommendation Engine (#14) + AI Insight (#15) ----------

export async function getAnalysisRecommendations(datasetId) {
  const res = await fetch(`${BASE_URL}/api/dataset/${datasetId}/analysis-recommendations`);
  return handle(res);
}

export async function runAnalysis(payload) {
  const res = await fetch(`${BASE_URL}/api/dataset/run-analysis`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handle(res);
}

export async function getAIInsight(payload) {
  const res = await fetch(`${BASE_URL}/api/dataset/ai-insight`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handle(res);
}

// ---------- Visualization Engine (#16) + Dashboard Builder (#17) ----------

export async function getChartRecommendations(datasetId) {
  const res = await fetch(`${BASE_URL}/api/dataset/${datasetId}/chart-recommendations`);
  return handle(res);
}

export async function explainChart(payload) {
  const res = await fetch(`${BASE_URL}/api/dataset/explain-chart`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handle(res);
}

export async function pinChart(payload) {
  const res = await fetch(`${BASE_URL}/api/dataset/dashboard/pin`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handle(res);
}

export async function unpinChart(datasetId, chartId) {
  const res = await fetch(`${BASE_URL}/api/dataset/dashboard/unpin`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset_id: datasetId, chart_id: chartId }),
  });
  return handle(res);
}

export async function getDashboard(datasetId) {
  const res = await fetch(`${BASE_URL}/api/dataset/${datasetId}/dashboard`);
  return handle(res);
}

// ---------- Report Generation (final pipeline stage) ----------

export async function generateReport(payload) {
  const res = await fetch(`${BASE_URL}/api/dataset/report`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch (_) {}
    throw new Error(detail);
  }
  const disposition = res.headers.get("Content-Disposition") || "";
  const match = disposition.match(/filename="?([^"]+)"?/);
  const filename = match ? match[1] : `report.${payload.format === "markdown" ? "md" : payload.format}`;
  const blob = await res.blob();
  return { blob, filename };
}
