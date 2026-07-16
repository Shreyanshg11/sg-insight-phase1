export default function ScatterPlotView({ points }) {
  const width = 480, height = 260, pad = 24;
  if (points.length === 0) {
    return <p className="font-mono text-xs text-ink-500">No points to plot.</p>;
  }
  const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const scaleX = (x) => pad + ((x - xMin) / (xMax - xMin || 1)) * (width - 2 * pad);
  const scaleY = (y) => height - pad - ((y - yMin) / (yMax - yMin || 1)) * (height - 2 * pad);

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full max-w-xl rounded border border-graphite-border bg-graphite-900">
      <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="#2A2F3A" />
      <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="#2A2F3A" />
      {points.filter((p) => !p.is_outlier).map((p, i) => (
        <circle key={i} cx={scaleX(p.x)} cy={scaleY(p.y)} r={2.5} fill="#E8A33D" fillOpacity={0.55} />
      ))}
      {points.filter((p) => p.is_outlier).map((p, i) => (
        <circle key={`o-${i}`} cx={scaleX(p.x)} cy={scaleY(p.y)} r={3.5} fill="#D9695F" fillOpacity={0.85} />
      ))}
    </svg>
  );
}
