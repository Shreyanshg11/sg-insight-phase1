const SEGMENTS = 10;

function levelColor(confidence) {
  if (confidence >= 0.8) return "bg-signal-green";
  if (confidence >= 0.6) return "bg-signal-amber";
  return "bg-signal-red";
}

/**
 * The recurring "signature" element of SG-Insight: every AI recommendation
 * carries a confidence reading rendered as a segmented signal meter rather
 * than a bare percentage — a visual reminder that confidence is a measured
 * quantity with an explainable source, not a vibe.
 */
export default function ConfidenceMeter({ confidence, label = "confidence" }) {
  const filled = Math.round(confidence * SEGMENTS);
  const color = levelColor(confidence);

  return (
    <div className="flex items-center gap-2">
      <div className="flex gap-[3px]" role="img" aria-label={`${label}: ${Math.round(confidence * 100)}%`}>
        {Array.from({ length: SEGMENTS }).map((_, i) => (
          <span
            key={i}
            className={`h-3 w-[3px] rounded-sm ${i < filled ? color : "bg-graphite-700"}`}
          />
        ))}
      </div>
      <span className="font-mono text-xs text-ink-300 tabular-nums">
        {Math.round(confidence * 100)}%
      </span>
    </div>
  );
}
