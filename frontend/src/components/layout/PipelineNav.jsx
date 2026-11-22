import TabNav from "../ui/TabNav";

export const PIPELINE_STAGES = [
  { key: "profile", label: "Profile", implemented: true },
  { key: "quality", label: "Quality", implemented: true },
  { key: "cleaning", label: "Cleaning", implemented: true },
  { key: "feature_engineering", label: "Feature Engineering", implemented: true },
  { key: "eda", label: "EDA", implemented: true },
  { key: "analysis", label: "Analysis", implemented: true },
  { key: "machine_learning", label: "Machine Learning", implemented: true },
  { key: "visualization", label: "Visualization", implemented: true },
  { key: "dashboard", label: "Dashboard", implemented: true },
  { key: "report", label: "Report", implemented: true },
];

function StageItem(stage, isActive) {
  return (
    <>
      <span
        className={`h-1.5 w-1.5 flex-shrink-0 rounded-full ${
          isActive ? "bg-signal-amber" : stage.implemented ? "bg-signal-green" : "bg-ink-700"
        }`}
      />
      <span className="flex-1 truncate">{stage.label}</span>
      {!stage.implemented && <span className="flex-shrink-0 text-[9px] text-ink-700">soon</span>}
    </>
  );
}

/**
 * Shows the FULL intended 10-stage pipeline (per the brief), not just what's
 * built — unimplemented stages are visibly present but disabled with a phase
 * tag, so the shape of the whole workbench is honest about what's here today
 * vs coming later, rather than only showing the 3 stages that currently work.
 *
 * Renders as an ARIA tablist (arrow keys move between stages, Home/End jump
 * to first/last). On small screens it's an off-canvas drawer controlled by
 * `mobileOpen`/`onCloseMobile`; on `lg` and up it's the static sidebar.
 */
export default function PipelineNav({ activeStage, onSelectStage, activeView, onSelectView, mobileOpen, onCloseMobile }) {
  const items = PIPELINE_STAGES.map((s) => ({ ...s, disabled: !s.implemented }));
  const activeKey = activeView === "pipeline" ? activeStage : null;

  return (
    <>
      {mobileOpen && (
        <div
          className="fixed inset-0 z-30 bg-graphite-950/70 lg:hidden"
          onClick={onCloseMobile}
          aria-hidden="true"
        />
      )}
      <nav
        className={`fixed inset-y-0 left-0 z-40 flex w-64 max-w-[85vw] flex-shrink-0 flex-col overflow-y-auto border-r border-graphite-border bg-graphite-900 py-4 transition-transform duration-200 lg:static lg:z-auto lg:w-52 lg:max-w-none lg:translate-x-0 ${
          mobileOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <p className="px-4 pb-2 font-mono text-[11px] uppercase tracking-wider text-ink-500">Workflow</p>
        <TabNav
          items={items}
          activeKey={activeKey}
          onChange={(key) => { onSelectStage(key); onCloseMobile?.(); }}
          orientation="vertical"
          ariaLabel="Pipeline stages"
          idPrefix="stage"
          className="flex flex-col"
          renderItem={StageItem}
          itemClassName={(item, isActive) =>
            `group flex items-center gap-2 px-4 py-2 text-left font-mono text-xs transition-colors focus-visible:outline-none ${
              isActive
                ? "bg-graphite-800 text-signal-amber"
                : item.implemented
                ? "text-ink-300 hover:bg-graphite-800 hover:text-ink-100"
                : "text-ink-700 cursor-not-allowed"
            }`
          }
        />

        <div className="mt-4 border-t border-graphite-border pt-3">
          <button
            onClick={() => { onSelectView("versions"); onCloseMobile?.(); }}
            aria-current={activeView === "versions" ? "page" : undefined}
            className={`flex w-full items-center gap-2 px-4 py-2 text-left font-mono text-xs transition-colors focus-visible:outline-none ${
              activeView === "versions" ? "bg-graphite-800 text-signal-amber" : "text-ink-300 hover:bg-graphite-800 hover:text-ink-100"
            }`}
          >
            <span className={`h-1.5 w-1.5 flex-shrink-0 rounded-full ${activeView === "versions" ? "bg-signal-amber" : "bg-ink-500"}`} />
            Dataset Versions
          </button>
        </div>
      </nav>
    </>
  );
}
