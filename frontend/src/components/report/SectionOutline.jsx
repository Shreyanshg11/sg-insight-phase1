import { useRef, useState } from "react";

/**
 * Reorderable, toggleable list of report sections. Drag-and-drop is
 * hand-rolled on the Pointer Events API (not the HTML5 Drag-and-Drop API)
 * specifically because Pointer Events unify mouse and touch — this app has
 * real mobile support (session 4's polish pass), and HTML5 DnD notoriously
 * doesn't work on touch devices. `setPointerCapture` means move/up events
 * keep routing to the same handle element even once the cursor leaves it,
 * which is what makes a smooth drag possible without a library.
 */
export default function SectionOutline({ sections, onChange }) {
  const [dragIndex, setDragIndex] = useState(null);
  const [overIndex, setOverIndex] = useState(null);
  const itemRefs = useRef([]);

  const startDrag = (index) => (e) => {
    setDragIndex(index);
    setOverIndex(index);
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const onMove = (e) => {
    if (dragIndex === null) return;
    const y = e.clientY;
    let closest = overIndex;
    let closestDist = Infinity;
    itemRefs.current.forEach((el, i) => {
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const mid = rect.top + rect.height / 2;
      const dist = Math.abs(y - mid);
      if (dist < closestDist) { closestDist = dist; closest = i; }
    });
    if (closest !== overIndex) setOverIndex(closest);
  };

  const endDrag = () => {
    if (dragIndex !== null && overIndex !== null && overIndex !== dragIndex) {
      const next = [...sections];
      const [moved] = next.splice(dragIndex, 1);
      next.splice(overIndex, 0, moved);
      onChange(next);
    }
    setDragIndex(null);
    setOverIndex(null);
  };

  const toggleInclude = (key) => {
    onChange(sections.map((s) => (s.key === key ? { ...s, include: !s.include } : s)));
  };

  return (
    <div className="space-y-1">
      {sections.map((s, i) => (
        <div
          key={s.key}
          ref={(el) => { itemRefs.current[i] = el; }}
          className={`flex items-center gap-2 rounded border bg-graphite-900 px-2 py-1.5 transition-colors ${
            dragIndex === i ? "opacity-50" : ""
          } ${overIndex === i && dragIndex !== null && dragIndex !== i ? "border-signal-amber" : "border-graphite-border"} ${
            !s.include ? "opacity-50" : ""
          }`}
        >
          <span
            onPointerDown={startDrag(i)}
            onPointerMove={onMove}
            onPointerUp={endDrag}
            className="touch-none cursor-grab select-none px-1 font-mono text-ink-700 hover:text-ink-500"
            aria-label={`Drag to reorder ${s.label}`}
          >
            ⠿
          </span>
          <input
            type="checkbox"
            checked={s.include}
            onChange={() => toggleInclude(s.key)}
            className="accent-signal-amber"
            aria-label={`Include ${s.label} section`}
          />
          <div className="min-w-0 flex-1">
            <p className="font-mono text-xs text-ink-100">{s.label}</p>
            {s.reason && <p className="truncate font-mono text-[10px] text-ink-500">{s.reason}</p>}
          </div>
        </div>
      ))}
    </div>
  );
}
