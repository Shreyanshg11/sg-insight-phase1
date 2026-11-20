"""
Stable identity for cleaning recommendations.

Recommendations are recomputed from the dataframe on every request, so their
old random uuid4 ids changed on every call and nothing could be remembered
about them (e.g. "the user dismissed this one"). The identity below is a
deterministic hash of the fields that define *what the recommendation is
about*, not of the evidence that supports it:

  duplicates ...... step, action
  missing values .. step, column            (NOT the suggested method: median ->
                                             mode must not resurrect a dismissal)
  outliers ........ step, column, action    (counts/fences are evidence, not identity)
  normalize case .. step, column, action
  fuzzy merge ..... step, column, action, match_type and the sorted, normalized
                    member values            (a different group is a different
                                              recommendation)

Dataset scoping is NOT part of the hash: the key says what a recommendation is,
and persisted status is always keyed by (dataset_id, rec_key), so the same key
in two datasets never interferes.

`evidence_fingerprint` hashes the human-readable issue text (which carries the
counts). It does not affect identity; it lets a dismissed recommendation report
"changed since dismissed" when the evidence moved, without silently resurfacing.
"""
import hashlib
import json

KEY_PREFIX = "rk1_"   # identity scheme version; a future scheme would use rk2_


def _norm(value) -> str:
    return " ".join(str(value).split()).casefold()


def identity_fields(rec) -> list:
    step, column, action = rec.step, rec.column, rec.recommended_action
    if step == "missing_value_center":
        return ["cleaning", step, column]
    if step == "category_merge_center" and action == "fuzzy_merge":
        preview = rec.preview or {}
        members = sorted({_norm(v) for v in preview.get("values", [])})
        return ["cleaning", step, column, action, preview.get("match_type"), members]
    return ["cleaning", step, column, action]


def rec_key(rec) -> str:
    payload = json.dumps(identity_fields(rec), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return KEY_PREFIX + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def evidence_fingerprint(rec) -> str:
    text = f"{rec.recommended_action}|{rec.issue}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def assign_stable_ids(recs) -> None:
    """Sets rec.id to the stable key, in place. If two recommendations in one
    response ever hash identically, later ones get a deterministic ~n suffix so
    React keys and dismissals never collide."""
    seen: dict[str, int] = {}
    for rec in recs:
        key = rec_key(rec)
        seen[key] = seen.get(key, 0) + 1
        rec.id = key if seen[key] == 1 else f"{key}~{seen[key]}"
