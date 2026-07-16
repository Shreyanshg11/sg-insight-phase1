import { useEffect, useState, useCallback } from "react";
import {
  getProfile, getQuality, applyAction, undoAction, getHistory, getWarnings, getHealth,
  compareMethods, getVersions, compareVersions, restoreVersion,
} from "../api/client";
import ProfileSummary from "../components/ProfileSummary";
import QualityScorePanel from "../components/QualityScorePanel";
import RecommendationCard from "../components/RecommendationCard";
import WarningsPanel from "../components/WarningsPanel";
import HealthDashboard from "../components/HealthDashboard";
import VersionHistory from "../components/VersionHistory";
import TopBar from "../components/layout/TopBar";
import PipelineNav from "../components/layout/PipelineNav";
import SessionContextPanel from "../components/layout/SessionContextPanel";
import RecentActionsPanel from "../components/layout/RecentActionsPanel";
import ComingSoonPanel from "../components/layout/ComingSoonPanel";
import FeatureEngineeringCenter from "../components/FeatureEngineeringCenter";
import EDAWorkspace from "../components/EDAWorkspace";
import AnalysisWorkspace from "../components/analysis/AnalysisWorkspace";
import VisualizationWorkspace from "../components/charts/VisualizationWorkspace";
import DashboardWorkspace from "../components/dashboard/DashboardWorkspace";
import ReportWorkspace from "../components/report/ReportWorkspace";
import LoadingState from "../components/ui/LoadingState";
import ErrorState from "../components/ui/ErrorState";
import EmptyState from "../components/ui/EmptyState";

const SORT_OPTIONS = [
  { value: "goal", label: "Goal relevance (default)" },
  { value: "missing_pct", label: "Missing % (high to low)" },
  { value: "confidence", label: "Confidence (high to low)" },
];

const STAGE_LABELS = {
  profile: "Profile", quality: "Quality", cleaning: "Cleaning",
  feature_engineering: "Feature Engineering", eda: "EDA", analysis: "Analysis",
  visualization: "Visualization", dashboard: "Dashboard", report: "Report",
};

function sortRecs(recs, sortBy) {
  if (sortBy === "missing_pct") {
    return [...recs].sort((a, b) => (b.preview?.missing_pct ?? -1) - (a.preview?.missing_pct ?? -1));
  }
  if (sortBy === "confidence") {
    return [...recs].sort((a, b) => b.confidence - a.confidence);
  }
  return recs; // "goal" — already sorted goal-relevant-first by the backend
}

export default function WorkbenchPage({ dataset }) {
  const [profile, setProfile] = useState(null);
  const [quality, setQuality] = useState(null);
  const [history, setHistory] = useState(null);
  const [warnings, setWarnings] = useState(null);
  const [health, setHealth] = useState(null);
  const [versions, setVersions] = useState(null);
  const [dismissed, setDismissed] = useState(new Set());
  const [sortBy, setSortBy] = useState("goal");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [activeView, setActiveView] = useState("pipeline"); // "pipeline" | "versions"
  const [activeStage, setActiveStage] = useState("profile");
  const [completedAnalyses, setCompletedAnalyses] = useState([]);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [mobileAsideOpen, setMobileAsideOpen] = useState(false);

  const handleAnalysisCompleted = useCallback((entry) => {
    setCompletedAnalyses((prev) => {
      const idx = prev.findIndex((a) => a.id === entry.id);
      if (idx === -1) return [...prev, entry];
      const next = [...prev];
      next[idx] = entry;
      return next;
    });
  }, []);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [p, q, h, w, hl, v] = await Promise.all([
        getProfile(dataset.dataset_id),
        getQuality(dataset.dataset_id),
        getHistory(dataset.dataset_id),
        getWarnings(dataset.dataset_id),
        getHealth(dataset.dataset_id),
        getVersions(dataset.dataset_id),
      ]);
      setProfile(p);
      setQuality(q);
      setHistory(h);
      setWarnings(w);
      setHealth(hl);
      setVersions(v);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [dataset.dataset_id]);

  useEffect(() => { refresh(); }, [refresh]);

  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === "Escape") {
        setMobileNavOpen(false);
        setMobileAsideOpen(false);
      }
    };
    window.addEventListener("keydown", handleEscape);
    return () => window.removeEventListener("keydown", handleEscape);
  }, []);

  const handleConfirm = async (rec, overrides = {}) => {
    await applyAction({
      dataset_id: dataset.dataset_id,
      step: rec.step,
      action: overrides.action || rec.recommended_action,
      column: rec.column,
      params: overrides.params || {},
      reason: rec.reason,
      confidence: rec.confidence,
    });
    await refresh();
  };

  const handleCompareMethods = (column) => compareMethods(dataset.dataset_id, column);
  const handleRestore = async (versionNumber) => { await restoreVersion(dataset.dataset_id, versionNumber); await refresh(); };
  const handleCompareVersions = (a, b) => compareVersions(dataset.dataset_id, a, b);
  const handleUndo = async () => { await undoAction(dataset.dataset_id); await refresh(); };

  if (loading && !profile) {
    return (
      <div className="flex h-screen items-center justify-center bg-graphite-950">
        <LoadingState message="Loading dataset…" />
      </div>
    );
  }
  if (error) {
    return (
      <div className="flex h-screen items-center justify-center bg-graphite-950 p-6">
        <div className="w-full max-w-md">
          <ErrorState message={error} onRetry={refresh} />
        </div>
      </div>
    );
  }

  const activeRecs = sortRecs(quality?.recommendations.filter((r) => !dismissed.has(r.id)) ?? [], sortBy);
  const goal = profile?.investigation_goal || dataset.investigation_goal;

  const renderStageContent = () => {
    if (activeStage === "profile") {
      return (
        <div className="space-y-4">
          {health && <HealthDashboard health={health} />}
          {profile && <ProfileSummary profile={profile} />}
        </div>
      );
    }
    if (activeStage === "quality") {
      return <div className="max-w-md">{quality && <QualityScorePanel score={quality.score} />}</div>;
    }
    if (activeStage === "cleaning") {
      return (
        <div>
          <div className="flex items-center justify-between">
            <p className="font-mono text-xs uppercase tracking-wider text-ink-500">
              Cleaning recommendations ({activeRecs.length})
              {goal && sortBy === "goal" && <span className="ml-2 text-ink-700">— goal-relevant items shown first</span>}
            </p>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="rounded border border-graphite-border bg-graphite-800 px-2 py-1 font-mono text-[11px] text-ink-300 focus:border-signal-amber focus:outline-none"
            >
              {SORT_OPTIONS.map((opt) => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
            </select>
          </div>
          <div className="mt-3 space-y-3">
            {activeRecs.length === 0 && (
              <EmptyState message="No outstanding recommendations — dataset looks clean, or every action has been handled." />
            )}
            {activeRecs.map((rec) => (
              <RecommendationCard
                key={rec.id}
                recommendation={rec}
                onConfirm={handleConfirm}
                onDismiss={(r) => setDismissed((prev) => new Set(prev).add(r.id))}
                onCompareMethods={handleCompareMethods}
              />
            ))}
          </div>
        </div>
      );
    }
    if (activeStage === "feature_engineering") {
      return <FeatureEngineeringCenter datasetId={dataset.dataset_id} onDatasetChanged={refresh} />;
    }
    if (activeStage === "eda") {
      return <EDAWorkspace datasetId={dataset.dataset_id} />;
    }
    if (activeStage === "analysis") {
      return <AnalysisWorkspace datasetId={dataset.dataset_id} investigationGoal={goal} onAnalysisCompleted={handleAnalysisCompleted} />;
    }
    if (activeStage === "visualization") {
      return <VisualizationWorkspace datasetId={dataset.dataset_id} />;
    }
    if (activeStage === "dashboard") {
      return <DashboardWorkspace datasetId={dataset.dataset_id} />;
    }
    if (activeStage === "report") {
      return <ReportWorkspace datasetId={dataset.dataset_id} completedAnalyses={completedAnalyses} />;
    }
    return <ComingSoonPanel stageKey={activeStage} label={STAGE_LABELS[activeStage]} />;
  };

  return (
    <div className="flex h-screen flex-col bg-graphite-950">
      <TopBar
        dataset={dataset}
        currentVersion={versions?.current_version ?? history?.current_version ?? 1}
        qualityScore={quality?.score?.overall ?? null}
        warningCount={warnings?.warnings?.length ?? 0}
        goal={goal}
        onUndo={handleUndo}
        canUndo={!!history && history.current_version > 1}
        onToggleNav={() => setMobileNavOpen((v) => !v)}
        onToggleAside={() => setMobileAsideOpen((v) => !v)}
      />

      <div className="flex flex-1 overflow-hidden">
        <PipelineNav
          activeStage={activeStage}
          onSelectStage={(stage) => { setActiveStage(stage); setActiveView("pipeline"); }}
          activeView={activeView}
          onSelectView={setActiveView}
          mobileOpen={mobileNavOpen}
          onCloseMobile={() => setMobileNavOpen(false)}
        />

        <main className="flex-1 overflow-y-auto px-4 py-6 sm:px-6">
          {activeView === "versions" ? (
            versions && (
              <VersionHistory
                datasetId={dataset.dataset_id}
                versions={versions.versions}
                currentVersion={versions.current_version}
                onRestore={handleRestore}
                onCompare={handleCompareVersions}
              />
            )
          ) : renderStageContent()}
        </main>

        {mobileAsideOpen && (
          <div
            className="fixed inset-0 z-30 bg-graphite-950/70 lg:hidden"
            onClick={() => setMobileAsideOpen(false)}
            aria-hidden="true"
          />
        )}
        <aside
          className={`fixed inset-y-0 right-0 z-40 w-72 max-w-[85vw] flex-shrink-0 space-y-4 overflow-y-auto border-l border-graphite-border bg-graphite-900 p-4 transition-transform duration-200 lg:static lg:z-auto lg:max-w-none lg:translate-x-0 ${
            mobileAsideOpen ? "translate-x-0" : "translate-x-full"
          }`}
        >
          <SessionContextPanel goal={goal} goalRelevantColumns={profile?.goal_relevant_columns} />
          {warnings && <WarningsPanel warnings={warnings.warnings} />}
          <RecentActionsPanel history={history?.history} />
        </aside>
      </div>
    </div>
  );
}
