// Pure, framework-free helpers for recommendation status and Apply eligibility.
// Kept free of React so they can be unit-tested with `node --test` (npm test).

// Mirror of backend/app/core/action_policy.py REVIEW_ONLY_ACTIONS. The backend also sends
// `review_only` / `executable` on every cleaning recommendation; this list is only the
// fallback for a recommendation that lacks those fields (e.g. an older backend).
export const REVIEW_ONLY_ACTIONS = ["flag_for_review"];

const normalizeAction = (action) => (typeof action === "string" ? action.trim().toLowerCase() : "");

/** Advice only: it never changes data, so it must never offer Apply. */
export function isReviewOnly(rec) {
  if (!rec) return false;
  return rec.review_only === true || REVIEW_ONLY_ACTIONS.includes(normalizeAction(rec.recommended_action));
}

/** The only gate for showing / sending an Apply action for a recommendation's own recommended action. */
export function canApply(rec) {
  if (!rec) return false;
  return !isReviewOnly(rec) && rec.executable !== false;
}

export const recStatus = (rec) => (rec?.status === "dismissed" ? "dismissed" : "open");

export function splitByStatus(recs) {
  const list = Array.isArray(recs) ? recs : [];
  return {
    open: list.filter((r) => recStatus(r) === "open"),
    dismissed: list.filter((r) => recStatus(r) === "dismissed"),
  };
}

export function recommendationCounts(recs) {
  const { open, dismissed } = splitByStatus(recs);
  return { open: open.length, dismissed: dismissed.length, total: open.length + dismissed.length };
}

/**
 * What the Cleaning tab should say:
 *   "outstanding"   - at least one open recommendation: render the cards
 *   "all_dismissed" - there are recommendations but the user dismissed every one
 *   "none"          - the backend produced no recommendations at all
 */
export function emptyStateKind(counts) {
  const open = counts?.open ?? 0;
  const dismissed = counts?.dismissed ?? 0;
  if (open > 0) return "outstanding";
  return dismissed > 0 ? "all_dismissed" : "none";
}

/** Immutable update of one recommendation's status inside a /quality response. */
export function setRecommendationStatus(quality, id, status) {
  if (!quality || !Array.isArray(quality.recommendations)) return quality;
  if (!quality.recommendations.some((r) => r.id === id)) return quality;
  const recommendations = quality.recommendations.map((r) => (r.id === id ? { ...r, status } : r));
  return { ...quality, recommendations, recommendation_counts: recommendationCounts(recommendations) };
}
