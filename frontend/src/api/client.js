const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const apiFetch = (input, init = {}) => fetch(input, { credentials: "include", ...init });

export async function getAuthStatus() {
  return handle(await apiFetch(`${BASE_URL}/api/auth/status`));
}

export async function login(password) {
  return handle(await apiFetch(`${BASE_URL}/api/auth/login`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ password }),
  }));
}

export async function logout() {
  return handle(await apiFetch(`${BASE_URL}/api/auth/logout`, { method: "POST" }));
}

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
  const res = await apiFetch(`${BASE_URL}/api/dataset/upload`, { method: "POST", body: form });
  return handle(res);
}

export async function listSamples() {
  const res = await apiFetch(`${BASE_URL}/api/dataset/samples`);
  return handle(res);
}

export async function loadSample(sampleId) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/samples/${sampleId}/load`, { method: "POST" });
  return handle(res);
}

export function exampleReportUrl(sampleId) {
  return `${BASE_URL}/api/dataset/samples/${sampleId}/example-report`;
}

export async function getProfile(datasetId) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/${datasetId}/profile`);
  return handle(res);
}

export async function getQuality(datasetId) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/${datasetId}/quality`);
  return handle(res);
}

export async function applyAction(payload) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/apply-action`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handle(res);
}

export async function undoAction(datasetId) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/${datasetId}/undo`, { method: "POST" });
  return handle(res);
}

export async function getHistory(datasetId) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/${datasetId}/history`);
  return handle(res);
}

export async function getWarnings(datasetId) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/${datasetId}/warnings`);
  return handle(res);
}

export async function getHealth(datasetId) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/${datasetId}/health`);
  return handle(res);
}

export async function compareMethods(datasetId, column) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/${datasetId}/compare-methods?column=${encodeURIComponent(column)}`);
  return handle(res);
}

// Cleaning-recommendation lifecycle. Status is persisted on the server; none of these can change data.
export async function listRecommendations(datasetId, status = "all") {
  const res = await apiFetch(`${BASE_URL}/api/dataset/${datasetId}/recommendations?status=${encodeURIComponent(status)}`);
  return handle(res);
}

export async function dismissRecommendation(datasetId, recKey) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/${datasetId}/recommendations/${encodeURIComponent(recKey)}/dismiss`, {
    method: "POST",
  });
  return handle(res);
}

export async function restoreRecommendation(datasetId, recKey) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/${datasetId}/recommendations/${encodeURIComponent(recKey)}/restore`, {
    method: "POST",
  });
  return handle(res);
}

export async function getVersions(datasetId) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/${datasetId}/versions`);
  return handle(res);
}

export async function compareVersions(datasetId, a, b) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/${datasetId}/versions/compare?a=${a}&b=${b}`);
  return handle(res);
}

export async function restoreVersion(datasetId, versionNumber) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/${datasetId}/versions/${versionNumber}/restore`, { method: "POST" });
  return handle(res);
}

export function exportVersionUrl(datasetId, versionNumber, format = "csv") {
  return `${BASE_URL}/api/dataset/${datasetId}/versions/${versionNumber}/export?format=${encodeURIComponent(format)}`;
}

// ---------- Feature Engineering Center (#12) ----------

export async function getFeatureRecommendations(datasetId) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/${datasetId}/feature-recommendations`);
  return handle(res);
}

export async function applyFeatureAction(payload) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/feature-action`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handle(res);
}

// ---------- EDA module navigation (#13) ----------

// `filters`: optional array of {column, op, value} — the active Interactive
// Investigation filter (session 12). Serialized as a JSON query param since
// these are GET requests; empty/omitted means "whole dataset", same as before.
function buildQuery(params) {
  const entries = Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== "");
  if (!entries.length) return "";
  return "?" + entries.map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join("&");
}

function filtersQueryValue(filters) {
  return filters && filters.length ? JSON.stringify(filters) : undefined;
}

export async function getEDAOverview(datasetId, filters) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/${datasetId}/eda/overview${buildQuery({ filters: filtersQueryValue(filters) })}`);
  return handle(res);
}

export async function getEDADistribution(datasetId, column, filters) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/${datasetId}/eda/distribution${buildQuery({ column, filters: filtersQueryValue(filters) })}`);
  return handle(res);
}

export async function getEDACorrelation(datasetId, filters) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/${datasetId}/eda/correlation${buildQuery({ filters: filtersQueryValue(filters) })}`);
  return handle(res);
}

export async function getEDAMissingPatterns(datasetId, filters) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/${datasetId}/eda/missing-patterns${buildQuery({ filters: filtersQueryValue(filters) })}`);
  return handle(res);
}

export async function getEDARelationships(datasetId, columnX, columnY, filters) {
  const res = await apiFetch(
    `${BASE_URL}/api/dataset/${datasetId}/eda/relationships${buildQuery({ column_x: columnX, column_y: columnY, filters: filtersQueryValue(filters) })}`
  );
  return handle(res);
}

export async function getEDACategoryAnalysis(datasetId, column, filters) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/${datasetId}/eda/category-analysis${buildQuery({ column, filters: filtersQueryValue(filters) })}`);
  return handle(res);
}

export async function getEDATimeAnalysis(datasetId, column, filters) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/${datasetId}/eda/time-analysis${buildQuery({ column, filters: filtersQueryValue(filters) })}`);
  return handle(res);
}

// ---------- Analysis Recommendation Engine (#14) + AI Insight (#15) ----------

export async function getAnalysisRecommendations(datasetId) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/${datasetId}/analysis-recommendations`);
  return handle(res);
}

export async function runAnalysis(payload) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/run-analysis`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handle(res);
}

export async function getAIInsight(payload) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/ai-insight`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handle(res);
}

// ---------- Visualization Engine (#16) + Dashboard Builder (#17) ----------

export async function getChartRecommendations(datasetId, filters) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/${datasetId}/chart-recommendations${buildQuery({ filters: filtersQueryValue(filters) })}`);
  return handle(res);
}

export async function getFilteredRows(datasetId, filters, page = 1, pageSize = 25) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/filtered-rows`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset_id: datasetId, filters: filters || [], page, page_size: pageSize }),
  });
  return handle(res);
}

export async function buildChartFiltered(datasetId, chartType, columns, filters) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/build-chart`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset_id: datasetId, chart_type: chartType, columns, filters: filters || [] }),
  });
  return handle(res);
}

export async function explainChart(payload) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/explain-chart`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handle(res);
}

export async function pinChart(payload) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/dashboard/pin`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handle(res);
}

export async function unpinChart(datasetId, chartId) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/dashboard/unpin`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset_id: datasetId, chart_id: chartId }),
  });
  return handle(res);
}

export async function sendCopilotMessage(datasetId, message, history) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/copilot/message`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset_id: datasetId, message, history }),
  });
  return handle(res);
}

// Confirmation is by proposal_id: the server executes only the action it stored when
// it created the proposal, never parameters sent back from the browser.
export async function confirmCopilotAction(datasetId, proposedAction) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/copilot/confirm-action`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(
      proposedAction?.proposal_id
        ? { dataset_id: datasetId, proposal_id: proposedAction.proposal_id }
        : { dataset_id: datasetId, proposed_action: proposedAction },
    ),
  });
  return handle(res);
}

export async function dismissCopilotAction(datasetId, proposalId) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/copilot/dismiss-action`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset_id: datasetId, proposal_id: proposalId }),
  });
  return handle(res);
}

export async function getDashboard(datasetId) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/${datasetId}/dashboard`);
  return handle(res);
}

// ---------- Report Generation (final pipeline stage) ----------

export async function draftReportPlan(datasetId, availableAnalyses, userRequest) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/report/draft-plan`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset_id: datasetId, available_analyses: availableAnalyses, user_request: userRequest || null }),
  });
  return handle(res);
}

export async function previewReport(payload) {
  // Same endpoint as generateReport, but returns text directly rather than
  // triggering a file download — used for the Smart Report Builder's live
  // preview (session 13), which re-fetches on every outline change.
  const res = await apiFetch(`${BASE_URL}/api/dataset/report`, {
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
  return res.text();
}

export async function generateReport(payload) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/report`, {
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

export async function getAiProviderStatus() {
  const res = await apiFetch(`${BASE_URL}/api/ai/status`);
  return handle(res);
}

// Backward-compatible alias for existing integrations.
export const getOllamaStatus = getAiProviderStatus;

// ---------- Dataset Intelligence ----------
export async function getDatasetIntelligence(datasetId, investigationGoal = "") {
  const res = await apiFetch(
    `${BASE_URL}/api/dataset/${datasetId}/dataset-intelligence`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ investigation_goal: investigationGoal || null }),
      signal: AbortSignal.timeout(660000),
    }
  );
  return handle(res);
}
// ---------- Supervised model recommendation and evaluation ----------
export async function getMLRecommendations(datasetId) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/${datasetId}/ml-recommendations`);
  return handle(res);
}

export async function trainModel(payload) {
  const res = await apiFetch(`${BASE_URL}/api/dataset/train-model`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handle(res);
}
