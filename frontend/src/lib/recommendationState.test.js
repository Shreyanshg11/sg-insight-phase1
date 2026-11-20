import test from "node:test";
import assert from "node:assert/strict";
import {
  canApply, isReviewOnly, splitByStatus, recommendationCounts, emptyStateKind, setRecommendationStatus,
} from "./recommendationState.js";

const rec = (over = {}) => ({ id: "rk1_a", step: "missing_value_center", recommended_action: "impute_median", ...over });

test("Apply is never offered for flag_for_review, in any casing or via the backend flags", () => {
  for (const action of ["flag_for_review", "FLAG_FOR_REVIEW", " Flag_For_Review "]) {
    assert.equal(canApply(rec({ step: "outlier_center", recommended_action: action })), false, action);
    assert.equal(isReviewOnly(rec({ recommended_action: action })), true, action);
  }
  assert.equal(canApply(rec({ review_only: true })), false);
  assert.equal(canApply(rec({ executable: false })), false);
  assert.equal(canApply(rec({ review_only: true, executable: true })), false);   // review_only always wins
});

test("Apply is offered for supported actions, including recommendations without the new fields", () => {
  assert.equal(canApply(rec()), true);
  assert.equal(canApply(rec({ executable: true, review_only: false })), true);
  assert.equal(canApply({ id: "f1", category: "ratio", recommended_action: "create_ratio" }), true);   // feature recs
  assert.equal(canApply(rec({ recommended_action: "drop_duplicates", step: "duplicate_manager" })), true);
});

test("missing or malformed recommendations cannot be applied", () => {
  for (const bad of [null, undefined]) assert.equal(canApply(bad), false);
  assert.equal(isReviewOnly(null), false);
  assert.equal(isReviewOnly(rec({ recommended_action: undefined })), false);
});

test("splitByStatus / counts treat anything not dismissed as open", () => {
  const recs = [rec({ id: "1" }), rec({ id: "2", status: "dismissed" }), rec({ id: "3", status: "open" }), rec({ id: "4", status: "weird" })];
  const { open, dismissed } = splitByStatus(recs);
  assert.deepEqual(open.map((r) => r.id), ["1", "3", "4"]);
  assert.deepEqual(dismissed.map((r) => r.id), ["2"]);
  assert.deepEqual(recommendationCounts(recs), { open: 3, dismissed: 1, total: 4 });
  assert.deepEqual(splitByStatus(undefined), { open: [], dismissed: [] });
  assert.deepEqual(splitByStatus("nope"), { open: [], dismissed: [] });
});

test("the three Cleaning-tab states are distinguishable", () => {
  assert.equal(emptyStateKind({ open: 2, dismissed: 1, total: 3 }), "outstanding");
  assert.equal(emptyStateKind({ open: 1, dismissed: 0, total: 1 }), "outstanding");
  assert.equal(emptyStateKind({ open: 0, dismissed: 3, total: 3 }), "all_dismissed");
  assert.equal(emptyStateKind({ open: 0, dismissed: 0, total: 0 }), "none");
  assert.equal(emptyStateKind(undefined), "none");
});

test("dismissing every recommendation yields 'all dismissed', never 'none'", () => {
  let q = { recommendations: [rec({ id: "1" }), rec({ id: "2" }), rec({ id: "3" })] };
  for (const id of ["1", "2", "3"]) q = setRecommendationStatus(q, id, "dismissed");
  assert.deepEqual(q.recommendation_counts, { open: 0, dismissed: 3, total: 3 });
  assert.equal(emptyStateKind(recommendationCounts(q.recommendations)), "all_dismissed");
});

test("setRecommendationStatus is immutable, reversible (rollback/restore) and ignores unknown ids", () => {
  const q = { score: { total: 80 }, recommendations: [rec({ id: "1" }), rec({ id: "2" })] };
  const dismissed = setRecommendationStatus(q, "1", "dismissed");
  assert.notEqual(dismissed, q);
  assert.equal(q.recommendations[0].status, undefined);                       // original untouched
  assert.equal(dismissed.recommendations[0].status, "dismissed");
  assert.equal(dismissed.recommendations[1].status, undefined);
  assert.deepEqual(dismissed.score, q.score);                                 // other fields preserved
  const restored = setRecommendationStatus(dismissed, "1", "open");
  assert.equal(restored.recommendations[0].status, "open");
  assert.deepEqual(restored.recommendation_counts, { open: 2, dismissed: 0, total: 2 });
  assert.equal(setRecommendationStatus(q, "missing", "dismissed"), q);        // unknown id: same object
  assert.equal(setRecommendationStatus(null, "1", "dismissed"), null);
  assert.deepEqual(setRecommendationStatus({}, "1", "dismissed"), {});
});
