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
