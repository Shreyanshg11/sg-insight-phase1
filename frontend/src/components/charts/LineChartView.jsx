export default function LineChartView({ buckets, trend }) {
  const width = 480, height = 220, pad = 30;
  const counts = buckets.map((b) => b.count);
  const max = Math.max(1, ...counts);
  const stepX = buckets.length > 1 ? (width - 2 * pad) / (buckets.length - 1) : 0;
  const scaleY = (v) => height - pad - (v / max) * (height - 2 * pad);
  const points = buckets.map((b, i) => `${pad + i * stepX},${scaleY(b.count)}`).join(" ");

  return (
    <div>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full max-w-xl rounded border border-graphite-border bg-graphite-900">
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="#2A2F3A" />
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="#2A2F3A" />
        <polyline points={points} fill="none" stroke="#E8A33D" strokeWidth={2} />
        {buckets.map((b, i) => (
          <circle key={i} cx={pad + i * stepX} cy={scaleY(b.count)} r={2.5} fill="#E8A33D" />
        ))}
      </svg>
      {trend && (
        <p className="mt-1.5 font-mono text-[11px] text-ink-500">
          Trend: <span className={trend.direction === "increasing" ? "text-signal-green" : trend.direction === "decreasing" ? "text-signal-red" : "text-ink-300"}>
            {trend.direction}
          </span> ({trend.slope_per_period}/period, R²={trend.r_squared})
        </p>
      )}
    </div>
  );
}
