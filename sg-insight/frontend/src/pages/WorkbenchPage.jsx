import { useEffect, useState, useCallback } from "react";
import { getProfile, getQuality, applyAction, undoAction, getHistory } from "../api/client";
import ProfileSummary from "../components/ProfileSummary";
import QualityScorePanel from "../components/QualityScorePanel";
import RecommendationCard from "../components/RecommendationCard";

export default function WorkbenchPage({ dataset }) {
  const [profile, setProfile] = useState(null);
  const [quality, setQuality] = useState(null);
  const [history, setHistory] = useState(null);
  const [dismissed, setDismissed] = useState(new Set());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [p, q, h] = await Promise.all([
        getProfile(dataset.dataset_id),
        getQuality(dataset.dataset_id),
        getHistory(dataset.dataset_id),
      ]);
      setProfile(p);
      setQuality(q);
      setHistory(h);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [dataset.dataset_id]);

  useEffect(() => { refresh(); }, [refresh]);

  const handleConfirm = async (rec) => {
    await applyAction({
      dataset_id: dataset.dataset_id,
      step: rec.step,
      action: rec.recommended_action,
      column: rec.column,
      reason: rec.reason,
      confidence: rec.confidence,
    });
    await refresh();
  };

  const handleUndo = async () => {
    await undoAction(dataset.dataset_id);
    await refresh();
  };

  if (loading && !profile) {
    return <div className="p-10 font-mono text-sm text-ink-500">Loading dataset…</div>;
  }
  if (error) {
    return <div className="p-10 font-mono text-sm text-signal-red">{error}</div>;
  }

  const activeRecs = quality?.recommendations.filter((r) => !dismissed.has(r.id)) ?? [];

  return (
    <div className="mx-auto max-w-5xl px-6 py-10">
      <div className="flex items-center justify-between">
        <div>
          <p className="font-mono text-xs uppercase tracking-widest text-signal-amber">SG-Insight</p>
          <h1 className="mt-1 font-display text-2xl font-semibold text-ink-100">
            {dataset.filename}
          </h1>
          {dataset.investigation_goal && (
            <p className="mt-1 text-sm text-ink-500">
              Investigation goal: <span className="text-ink-300">{dataset.investigation_goal}</span>
            </p>
          )}
        </div>
        <div className="flex items-center gap-3">
          <span className="font-mono text-xs text-ink-500">
            v{history?.current_version ?? 1}
          </span>
          <button
            onClick={handleUndo}
            disabled={!history || history.current_version <= 1}
            className="rounded border border-graphite-border px-3 py-1.5 text-xs text-ink-300 hover:bg-graphite-700 disabled:opacity-40"
          >
            Undo last action
          </button>
        </div>
      </div>

      <div className="mt-6 grid grid-cols-3 gap-4">
        <div className="col-span-2">{profile && <ProfileSummary profile={profile} />}</div>
        <div>{quality && <QualityScorePanel score={quality.score} />}</div>
      </div>

      <div className="mt-8">
        <p className="font-mono text-xs uppercase tracking-wider text-ink-500">
          Cleaning recommendations ({activeRecs.length})
        </p>
        <div className="mt-3 space-y-3">
          {activeRecs.length === 0 && (
            <p className="rounded-lg border border-graphite-border bg-graphite-800 p-4 text-sm text-ink-500">
              No outstanding recommendations — dataset looks clean, or every action has been handled.
            </p>
          )}
          {activeRecs.map((rec) => (
            <RecommendationCard
              key={rec.id}
              recommendation={rec}
              onConfirm={handleConfirm}
              onDismiss={(r) => setDismissed((prev) => new Set(prev).add(r.id))}
            />
          ))}
        </div>
      </div>

      {history && history.history.length > 0 && (
        <div className="mt-8">
          <p className="font-mono text-xs uppercase tracking-wider text-ink-500">Action history</p>
          <div className="mt-3 space-y-2">
            {history.history.map((h, i) => (
              <div key={i} className="flex items-center justify-between rounded border border-graphite-border bg-graphite-900 px-3 py-2 font-mono text-xs">
                <span className="text-ink-500">v{h.version}</span>
                <span className="text-ink-100">{h.step} · {h.action}{h.column ? ` · ${h.column}` : ""}</span>
                <span className="text-ink-500">{h.confidence != null ? `${Math.round(h.confidence * 100)}%` : ""}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
