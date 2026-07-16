import { useRef } from "react";

/**
 * Shared tab navigation used everywhere the app has an "array of sections +
 * one active key" pattern: PipelineNav's 9 workflow stages, EDA's 7
 * sub-sections, and the Analysis/Visualization/Dashboard sub-navs.
 *
 * Implements the standard ARIA tabs pattern with automatic activation:
 * - role="tablist" / role="tab" / aria-selected / aria-controls
 * - roving tabindex (only the active tab is in the tab order)
 * - ArrowLeft/ArrowRight (horizontal) or ArrowUp/ArrowDown (vertical) move
 *   focus AND select, skipping disabled items
 * - Home/End jump to the first/last enabled item
 *
 * `items`: [{ key, label, disabled, badge, indicator }]
 * `renderItem`: optional (item, isActive) => node, for fully custom content
 * (e.g. PipelineNav's colored dot + "soon" tag) while TabNav still owns all
 * the roving-tabindex/keyboard/ARIA wiring.
 */
export default function TabNav({
  items,
  activeKey,
  onChange,
  orientation = "horizontal",
  ariaLabel,
  renderItem,
  className = "",
  itemClassName,
  idPrefix = "tab",
}) {
  const refs = useRef({});

  const enabledKeys = items.filter((i) => !i.disabled).map((i) => i.key);

  const focusAndSelect = (key) => {
    onChange(key);
    refs.current[key]?.focus();
  };

  const handleKeyDown = (e) => {
    const isHorizontal = orientation === "horizontal";
    const nextKey = isHorizontal ? "ArrowRight" : "ArrowDown";
    const prevKey = isHorizontal ? "ArrowLeft" : "ArrowUp";

    if (![nextKey, prevKey, "Home", "End"].includes(e.key)) return;
    e.preventDefault();

    const currentIdx = enabledKeys.indexOf(activeKey);
    if (e.key === "Home") return focusAndSelect(enabledKeys[0]);
    if (e.key === "End") return focusAndSelect(enabledKeys[enabledKeys.length - 1]);

    if (e.key === nextKey) {
      const next = enabledKeys[(currentIdx + 1) % enabledKeys.length];
      return focusAndSelect(next);
    }
    if (e.key === prevKey) {
      const prev = enabledKeys[(currentIdx - 1 + enabledKeys.length) % enabledKeys.length];
      return focusAndSelect(prev);
    }
  };

  return (
    <div
      role="tablist"
      aria-label={ariaLabel}
      aria-orientation={orientation}
      className={className}
      onKeyDown={handleKeyDown}
    >
      {items.map((item) => {
        const isActive = item.key === activeKey;
        return (
          <button
            key={item.key}
            ref={(el) => { refs.current[item.key] = el; }}
            role="tab"
            id={`${idPrefix}-${item.key}`}
            aria-selected={isActive}
            aria-controls={`${idPrefix}-panel-${item.key}`}
            disabled={item.disabled}
            tabIndex={isActive ? 0 : -1}
            onClick={() => !item.disabled && onChange(item.key)}
            className={itemClassName ? itemClassName(item, isActive) : defaultItemClass(item, isActive, orientation)}
          >
            {renderItem ? renderItem(item, isActive) : item.label}
          </button>
        );
      })}
    </div>
  );
}

function defaultItemClass(item, isActive, orientation) {
  const base = "font-mono text-xs transition-colors focus-visible:outline-none";
  const spacing = orientation === "horizontal" ? "px-3 py-1.5" : "px-4 py-2 text-left";
  if (item.disabled) return `${base} ${spacing} cursor-not-allowed text-ink-700`;
  return `${base} ${spacing} ${
    isActive
      ? "bg-graphite-800 text-signal-amber"
      : "text-ink-300 hover:bg-graphite-800 hover:text-ink-100"
  }`;
}
