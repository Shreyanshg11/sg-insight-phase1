/**
 * Sibling panels to QualityScorePanel — together the three make up the app's
 * "three constructs" view (Data Quality / Analytical Risk / Readiness).
 * Deliberately mirrors QualityScorePanel's exact structure (header, big
 * number, breakdown-with-hover-tooltip) so the three read as one family
 * rather than three different UI patterns bolted together.
 *
 * Score direction differs between the two: Analytical Risk is a risk meter
 * (0 = none found, 100 = severe — colored red as it climbs), Readiness reads
 * like Data Quality (0 = not ready, 100 = fully ready — colored green as it
 * climbs). Each header states its own direction so it's never ambiguous at
 * a glance.
 */
function riskColor(score) {
  if (score >= 50) return "text-signal-red";
  if (score >= 20) return "text-signal-amber";
  return "text-signal-green";
}

function readinessColor(score) {
  if (score >= 70) return "text-signal-green";
  if (score >= 40) return "text-signal-amber";
  return "text-signal-red";
}

function ScorePanel({ title, subtitle, score, colorClass, breakdown, pointsLabel }) {
  return (
    <div className="rounded-lg border border-graphite-border bg-graphite-800 p-5">
      <div className="flex items-baseline justify-between">
        <div>
          <p className="font-mono text-xs uppercase tracking-wider text-ink-500">{title}</p>
          <p className="mt-0.5 text-[11px] text-ink-700">{subtitle}</p>
        </div>
        <p className={`font-display text-3xl font-semibold tabular-nums ${colorClass}`}>
          {score}<span className="text-base text-ink-500">/100</span>
        </p>
      </div>

      {breakdown?.length > 0 ? (
        <div className="mt-5 border-t border-graphite-border pt-4">
          <p className="mb-2 font-mono text-[11px] uppercase tracking-wider text-ink-500">
            Why the score is {score}
          </p>
          <div className="space-y-1.5">
            {breakdown.map((item) => (
              <div key={item.label} tabIndex={0} className="group relative flex items-center justify-between rounded focus-visible:outline-none">
                <span className="font-mono text-xs text-ink-300">{item.label}</span>
                <span className="font-mono text-xs tabular-nums text-ink-100">
                  {pointsLabel(item.points)}
                </span>
                <div className="pointer-events-none absolute left-0 top-full z-10 mt-1 hidden w-64 rounded border border-graphite-border bg-graphite-900 p-2 text-[11px] leading-snug text-ink-500 shadow-lg group-hover:block group-focus-within:block">
                  {item.reason}
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <p className="mt-4 border-t border-graphite-border pt-4 font-mono text-xs text-ink-500">
          Nothing contributing to this score.
        </p>
      )}
    </div>
  );
}

export function AnalyticalRiskPanel({ risk }) {
  if (!risk) return null;
  return (
    <ScorePanel
      title="Analytical Risk"
      subtitle="0 = no risk signals found, 100 = severe"
      score={risk.overall}
      colorClass={riskColor(risk.overall)}
      breakdown={risk.breakdown}
      pointsLabel={(p) => `+${p}`}
    />
  );
}

export function ReadinessPanel({ readiness }) {
  if (!readiness) return null;
  return (
    <ScorePanel
      title="Readiness"
      subtitle="0 = not structurally ready, 100 = fully ready"
      score={readiness.overall}
      colorClass={readinessColor(readiness.overall)}
      breakdown={readiness.breakdown}
      pointsLabel={(p) => (p > 0 ? `+${p}` : "0")}
    />
  );
}
