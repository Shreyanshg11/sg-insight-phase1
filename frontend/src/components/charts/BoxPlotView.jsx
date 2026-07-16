export default function BoxPlotView({ groups }) {
  if (!groups || groups.length === 0) {
    return <p className="font-mono text-xs text-ink-500">No groups to plot.</p>;
  }
  const width = 480, height = 260, pad = 30;
  const globalMin = Math.min(...groups.map((g) => g.min));
  const globalMax = Math.max(...groups.map((g) => g.max));
  const scaleY = (v) => height - pad - ((v - globalMin) / (globalMax - globalMin || 1)) * (height - 2 * pad);
  const slotWidth = (width - 2 * pad) / groups.length;
  const boxWidth = Math.min(40, slotWidth * 0.5);

  return (
    <div>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full max-w-xl rounded border border-graphite-border bg-graphite-900">
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="#2A2F3A" />
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="#2A2F3A" />
        {groups.map((g, i) => {
          const cx = pad + slotWidth * (i + 0.5);
          return (
            <g key={i}>
              <line x1={cx} y1={scaleY(g.min)} x2={cx} y2={scaleY(g.max)} stroke="#8A93A6" strokeWidth={1} />
              <rect x={cx - boxWidth / 2} y={scaleY(g.q3)} width={boxWidth} height={Math.max(1, scaleY(g.q1) - scaleY(g.q3))}
                    fill="#E8A33D" fillOpacity={0.35} stroke="#E8A33D" />
              <line x1={cx - boxWidth / 2} y1={scaleY(g.median)} x2={cx + boxWidth / 2} y2={scaleY(g.median)} stroke="#E8A33D" strokeWidth={2} />
              <text x={cx} y={height - pad + 14} textAnchor="middle" fontSize="9" fill="#8A93A6" fontFamily="monospace">
                {g.category.length > 10 ? g.category.slice(0, 9) + "…" : g.category}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="mt-1.5 flex flex-wrap gap-3">
        {groups.map((g, i) => (
          <span key={i} className="font-mono text-[11px] text-ink-500">
            {g.category}: n={g.n}, median={g.median}{g.outlier_count > 0 && `, ${g.outlier_count} outlier(s)`}
          </span>
        ))}
      </div>
    </div>
  );
}
