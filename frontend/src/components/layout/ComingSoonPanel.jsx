const PHASE_BY_STAGE = {};

export default function ComingSoonPanel({ stageKey, label }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-graphite-border bg-graphite-900 px-6 py-16 text-center">
      <p className="font-mono text-xs uppercase tracking-wider text-ink-700">{PHASE_BY_STAGE[stageKey] || "Coming later"}</p>
      <p className="mt-2 font-display text-lg text-ink-300">{label} isn't built yet</p>
      <p className="mt-1 max-w-sm text-sm text-ink-700">
        This stage is part of the full workflow but hasn't been implemented in this build.
        It's shown here so the pipeline reflects the whole intended workbench, not just what exists today.
      </p>
    </div>
  );
}
