import { createContext, useContext, useState, useCallback } from "react";

const FilterContext = createContext(null);

/**
 * Global filter state (session 12) — one active filter, shared across every
 * tab until cleared. A filter is a VIEW, never a dataset version: it's never
 * sent through apply-action/push_version, just passed as a query/body param
 * to whichever filter-aware endpoint is being called (EDA, chart
 * recommendations, filtered-rows).
 *
 * One condition per column: adding a new condition on a column that already
 * has one replaces it, rather than stacking (keeps the filter bar's chips
 * simple to reason about — "region is East", not "region is East AND
 * region is West").
 */
export function FilterProvider({ children }) {
  const [filters, setFilters] = useState([]);
  const [drillDownOpen, setDrillDownOpen] = useState(false);

  const addFilter = useCallback((condition) => {
    setFilters((prev) => [...prev.filter((f) => f.column !== condition.column), condition]);
    setDrillDownOpen(true);
  }, []);

  const removeFilter = useCallback((column) => {
    setFilters((prev) => prev.filter((f) => f.column !== column));
  }, []);

  const clearFilters = useCallback(() => {
    setFilters([]);
    setDrillDownOpen(false);
  }, []);

  return (
    <FilterContext.Provider value={{ filters, addFilter, removeFilter, clearFilters, drillDownOpen, setDrillDownOpen }}>
      {children}
    </FilterContext.Provider>
  );
}

export function useFilters() {
  const ctx = useContext(FilterContext);
  if (!ctx) throw new Error("useFilters() must be called within a FilterProvider");
  return ctx;
}
