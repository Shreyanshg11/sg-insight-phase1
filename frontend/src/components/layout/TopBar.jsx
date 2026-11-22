import { useState } from "react";
import { useOperatorAuth } from "../authContext";

function Badge({ children, tone = "default" }) {
  const tones = {
    default: "border-graphite-border text-ink-300",
    amber: "border-signal-amber/40 text-signal-amber",
    red: "border-signal-red/40 text-signal-red",
    green: "border-signal-green/40 text-signal-green",
  };
  return (
    <span className={`rounded border px-2 py-0.5 font-mono text-[11px] ${tones[tone]}`}>
      {children}
    </span>
  );
}

/**
 * Persistent top bar per the brief: Dataset Name, Version, Quality Score,
 * Session Goal, Warnings — always visible regardless of which pipeline
 * stage is active, so the analyst never loses track of dataset-level state
 * while drilling into a specific stage.
 */
function IconButton({ onClick, label, children }) {
  return (
    <button
      onClick={onClick}
      aria-label={label}
      className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded border border-graphite-border text-ink-300 hover:bg-graphite-700 focus-visible:outline-none lg:hidden"
    >
      {children}
    </button>
  );
}

export default function TopBar({
  dataset, currentVersion, qualityScore, warningCount, goal, onUndo, canUndo,
  onToggleNav, onToggleAside,
}) {
  const { enabled, signOut } = useOperatorAuth();
  const [authError, setAuthError] = useState("");
  const scoreTone = qualityScore == null ? "default" : qualityScore >= 80 ? "green" : qualityScore >= 60 ? "amber" : "red";
  const handleSignOut = async () => {
    setAuthError("");
    try { await signOut(); } catch (error) { setAuthError(error.message || "Could not sign out."); }
  };

  return (
    <div className="flex flex-wrap items-center justify-between gap-2 border-b border-graphite-border bg-graphite-900 px-3 py-3 sm:px-6">
      <div className="flex min-w-0 items-center gap-2 sm:gap-3">
        <IconButton onClick={onToggleNav} label="Toggle workflow navigation">
          <span aria-hidden="true" className="font-mono text-sm">☰</span>
        </IconButton>
        <span className="hidden font-mono text-xs uppercase tracking-widest text-signal-amber whitespace-nowrap sm:inline">SG-Insight</span>
        <span className="truncate font-display text-sm font-medium text-ink-100">{dataset.filename}</span>
        {goal && (
          <span className="hidden truncate font-mono text-xs text-ink-500 md:inline">
            — goal: <span className="text-ink-300">{goal}</span>
          </span>
        )}
      </div>
      <div className="flex flex-shrink-0 flex-wrap items-center gap-2">
        <Badge>v{currentVersion ?? 1}</Badge>
        {qualityScore != null && <Badge tone={scoreTone}>quality {qualityScore}</Badge>}
        {warningCount > 0 && <Badge tone="amber">{warningCount} warning{warningCount !== 1 ? "s" : ""}</Badge>}
        <button
          onClick={onUndo}
          disabled={!canUndo}
          className="rounded border border-graphite-border px-3 py-1 font-mono text-xs text-ink-300 hover:bg-graphite-700 disabled:opacity-40 focus-visible:outline-none"
        >
          Undo
        </button>
        {enabled && <button onClick={handleSignOut} aria-describedby={authError ? "signout-error" : undefined}
          className="rounded border border-graphite-border px-3 py-1 font-mono text-xs text-ink-300 hover:bg-graphite-700 focus-visible:outline-none">Sign out</button>}
        <IconButton onClick={onToggleAside} label="Toggle session context panel">
          <span aria-hidden="true" className="font-mono text-sm">i</span>
        </IconButton>
      </div>
      {authError && <span id="signout-error" role="alert" className="w-full text-right text-xs text-red-300">{authError}</span>}
    </div>
  );
}
