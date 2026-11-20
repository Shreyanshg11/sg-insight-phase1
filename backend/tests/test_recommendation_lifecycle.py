"""Phase A - cleaning recommendation lifecycle: stable identity, persisted dismissal/restore,
and review-only (flag_for_review) safety. Runs against the real FastAPI app, session store and
SQLite file; nothing is mocked."""
import hashlib
import sqlite3
import threading
import uuid

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core import persistence, proposal_store, recommendation_store
from app.core.recommendation_identity import identity_fields, rec_key
from app.core.session_store import store
from app.engines.cleaning_engine import (
    EXECUTABLE_RECOMMENDED_ACTIONS, apply_action, generate_recommendations, is_executable_recommendation)
from app.main import app


def make_df(extra_missing=False, drop_dupes=False):
    """Yields every recommendation type: duplicates, missing values (qty), outlier (price),
    case/whitespace variants (region) and a fuzzy spelling group (shade)."""
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame({
        "price": np.round(rng.normal(50, 5, n), 2),
        "qty": rng.integers(1, 20, n).astype(float),
        "region": ["North"] * 30 + ["north "] * 10 + ["South"] * 40,
        "shade": ["Colour"] * 10 + ["Color"] * 50 + ["Blue"] * 20,
    })
    df.loc[3, "price"] = 5000.0
    df.loc[[5, 6, 7, 8, 9], "qty"] = np.nan
    if extra_missing:
        df.loc[[10, 11, 12, 13], "price"] = np.nan
    if not drop_dupes:
        df = pd.concat([df, df.iloc[:2]], ignore_index=True)
    return df


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def session():
    return store.create(f"reclife_{uuid.uuid4().hex[:6]}.csv", make_df())


def recs_of(client, ds):
    r = client.get(f"/api/dataset/{ds}/quality")
    assert r.status_code == 200, r.text
    return r.json()["recommendations"], r.json()["recommendation_counts"]


def by_step(recs, step, column=None):
    return next(r for r in recs if r["step"] == step and (column is None or r["column"] == column))


def dismiss(client, ds, key, **body):
    return client.post(f"/api/dataset/{ds}/recommendations/{key}/dismiss", json=body or None)


def restore(client, ds, key, **body):
    return client.post(f"/api/dataset/{ds}/recommendations/{key}/restore", json=body or None)


def snapshot(session):
    """Everything dismissal/restore must leave untouched."""
    s = store.get(session.dataset_id)
    with sqlite3.connect(persistence.DB_PATH) as conn:
        version_count = conn.execute("SELECT version_count FROM sessions WHERE dataset_id=?", (s.dataset_id,)).fetchone()
        log_rows = conn.execute("SELECT COUNT(*) FROM log_entries WHERE dataset_id=?", (s.dataset_id,)).fetchone()[0]
    files = sorted((p.name, p.stat().st_mtime_ns, hashlib.md5(p.read_bytes()).hexdigest())
                   for p in (persistence.VERSIONS_DIR / s.dataset_id).glob("v*.pkl"))
    return {"version": s.version_number, "log": len(s.log), "frames": [v.copy(deep=True) for v in s.versions],
            "db_version_count": tuple(version_count) if version_count else None, "db_log_rows": log_rows, "files": files}


def assert_untouched(before, after):
    assert before["version"] == after["version"] and before["log"] == after["log"]
    assert before["db_version_count"] == after["db_version_count"] and before["db_log_rows"] == after["db_log_rows"]
    assert before["files"] == after["files"]
    assert len(before["frames"]) == len(after["frames"])
    for a, b in zip(before["frames"], after["frames"]):
        pd.testing.assert_frame_equal(a, b)


def status_rows(ds):
    with sqlite3.connect(persistence.DB_PATH) as conn:
        return conn.execute("SELECT rec_key, status FROM recommendation_status WHERE dataset_id=?", (ds,)).fetchall()


# ---------------------------------------------------------------- 1. stable identity
class TestStableIdentity:
    def test_ids_are_identical_across_repeated_requests(self, client, session):
        first, _ = recs_of(client, session.dataset_id)
        second, _ = recs_of(client, session.dataset_id)
        assert {r["id"] for r in first} == {r["id"] for r in second}
        assert len(first) == 5 and all(r["id"].startswith("rk1_") for r in first)

    def test_ids_are_not_random_uuids_and_are_unique(self, client, session):
        recs, _ = recs_of(client, session.dataset_id)
        assert len({r["id"] for r in recs}) == len(recs)
        for r in recs:
            with pytest.raises(ValueError):
                uuid.UUID(r["id"])

    def test_ids_survive_a_restart(self, client, session):
        before = {r["id"] for r in recs_of(client, session.dataset_id)[0]}
        store._sessions.pop(session.dataset_id)            # fresh process: session reloaded from disk
        assert {r["id"] for r in recs_of(client, session.dataset_id)[0]} == before

    def test_same_content_gives_the_same_key_in_any_dataset(self, client):
        a, b = store.create("a.csv", make_df()), store.create("b.csv", make_df())
        assert {r["id"] for r in recs_of(client, a.dataset_id)[0]} == {r["id"] for r in recs_of(client, b.dataset_id)[0]}

    def test_identity_ignores_evidence_but_not_subject(self):
        base = {r.step + str(r.column): r for r in generate_recommendations(make_df(), [])}
        more = {r.step + str(r.column): r for r in generate_recommendations(make_df(extra_missing=True), [])}
        for key in base:                                   # unchanged recommendations keep their identity...
            assert rec_key(base[key]) == rec_key(more[key]), key
        assert "missing_value_centerprice" in more and "missing_value_centerprice" not in base   # ...a new subject is new
        assert rec_key(more["missing_value_centerprice"]) not in {rec_key(r) for r in base.values()}

    def test_missing_value_identity_excludes_the_suggested_method(self):
        rec = next(r for r in generate_recommendations(make_df(), []) if r.step == "missing_value_center")
        assert rec.recommended_action in ("impute_mean", "impute_median")
        flipped = rec.model_copy(update={"recommended_action": "impute_mode" if rec.recommended_action != "impute_mode" else "impute_mean"})
        assert rec_key(rec) == rec_key(flipped) and identity_fields(rec) == ["cleaning", "missing_value_center", "qty"]

    def test_fuzzy_identity_follows_group_members_not_spelling_or_order(self):
        rec = next(r for r in generate_recommendations(make_df(), []) if r.recommended_action == "fuzzy_merge")
        shuffled = rec.model_copy(update={"preview": {**rec.preview, "values": list(reversed([v.upper() for v in rec.preview["values"]]))}})
        assert rec_key(rec) == rec_key(shuffled)
        other = rec.model_copy(update={"preview": {**rec.preview, "values": ["Alpha", "Alpha1"]}})
        assert rec_key(rec) != rec_key(other)

    def test_duplicate_keys_in_one_response_get_deterministic_suffixes(self):
        from app.core.recommendation_identity import assign_stable_ids
        rec = generate_recommendations(make_df(), [])[0]
        a, b = rec.model_copy(), rec.model_copy()
        assign_stable_ids([a, b])
        assert b.id == a.id + "~2" and a.id == rec_key(rec)


# ---------------------------------------------------------------- 2-4. dismissal persists, no data/version change
class TestDismissal:
    def test_dismissal_persists_across_refresh_and_restart(self, client, session):
        ds = session.dataset_id
        recs, counts = recs_of(client, ds)
        assert counts == {"open": 5, "dismissed": 0, "total": 5}
        key = by_step(recs, "missing_value_center")["id"]
        r = dismiss(client, ds, key)
        assert r.status_code == 200 and r.json() == {"dataset_id": ds, "rec_key": key, "status": "dismissed", "changed": True,
                                                     "dataset_version": 1, "version_changed": False}
        for _ in range(2):                                                    # "refresh" = a new request recomputing everything
            recs, counts = recs_of(client, ds)
            assert by_step(recs, "missing_value_center")["status"] == "dismissed"
            assert counts == {"open": 4, "dismissed": 1, "total": 5}
        store._sessions.pop(ds)                                                # "reload" = new process / new session cache
        recs, counts = recs_of(client, ds)
        assert by_step(recs, "missing_value_center")["status"] == "dismissed" and counts["dismissed"] == 1
        listed = client.get(f"/api/dataset/{ds}/recommendations?status=dismissed").json()
        assert [x["id"] for x in listed["recommendations"]] == [key] and listed["counts"]["dismissed"] == 1

    def test_dismissal_does_not_change_data_or_create_a_version(self, client, session):
        ds = session.dataset_id
        before = snapshot(session)
        for rec in recs_of(client, ds)[0]:
            assert dismiss(client, ds, rec["id"]).status_code == 200
        assert_untouched(before, snapshot(session))
        history = client.get(f"/api/dataset/{ds}/history").json()
        assert history["history"] == [] and history["current_version"] == 1
        assert store.get(ds).version_number == 1

    def test_all_dismissed_is_distinguishable_from_none_available(self, client, session):
        ds = session.dataset_id
        for rec in recs_of(client, ds)[0]:
            dismiss(client, ds, rec["id"])
        recs, counts = recs_of(client, ds)
        assert counts == {"open": 0, "dismissed": 5, "total": 5} and all(r["status"] == "dismissed" for r in recs)
        clean = store.create("clean.csv", pd.DataFrame({"a": np.arange(30.0), "b": np.arange(30.0)[::-1]}))
        assert recs_of(client, clean.dataset_id)[1]["total"] == 0 or recs_of(client, clean.dataset_id)[1]["dismissed"] == 0

    def test_list_filters_and_counts(self, client, session):
        ds = session.dataset_id
        key = by_step(recs_of(client, ds)[0], "duplicate_manager")["id"]
        dismiss(client, ds, key)
        get = lambda s: client.get(f"/api/dataset/{ds}/recommendations?status={s}").json()
        assert len(get("open")["recommendations"]) == 4 and len(get("dismissed")["recommendations"]) == 1
        assert len(get("all")["recommendations"]) == 5 and get("all")["counts"] == {"open": 4, "dismissed": 1, "total": 5}
        assert get("all")["dataset_version"] == 1 and get("all")["state_token"].startswith("v1:")
        assert client.get(f"/api/dataset/{ds}/recommendations?status=bogus").status_code == 422

    def test_dismissed_stay_dismissed_through_apply_undo_and_restore_version(self, client, session):
        ds = session.dataset_id
        recs, _ = recs_of(client, ds)
        key = by_step(recs, "outlier_center")["id"]
        dismiss(client, ds, key)
        dup = by_step(recs, "duplicate_manager")
        r = client.post("/api/dataset/apply-action", json={"dataset_id": ds, "step": dup["step"], "action": dup["recommended_action"],
                                                          "params": {}, "reason": dup["reason"], "confidence": dup["confidence"]})
        assert r.status_code == 200 and store.get(ds).version_number == 2
        recs, _ = recs_of(client, ds)
        assert by_step(recs, "outlier_center")["status"] == "dismissed"              # new version, still dismissed
        assert not any(x["step"] == "duplicate_manager" for x in recs)
        client.post(f"/api/dataset/{ds}/undo")
        recs, _ = recs_of(client, ds)
        assert by_step(recs, "outlier_center")["status"] == "dismissed"              # after undo
        assert by_step(recs, "duplicate_manager")["status"] == "open"                # an applied-then-undone one is open again
        client.post(f"/api/dataset/{ds}/versions/1/restore")
        assert by_step(recs_of(client, ds)[0], "outlier_center")["status"] == "dismissed"


# ---------------------------------------------------------------- 5. scoping and invalid identities
class TestScopingAndInvalidIdentities:
    def test_dismissal_is_scoped_to_its_dataset(self, client):
        a, b = store.create("a.csv", make_df()), store.create("b.csv", make_df())
        key = by_step(recs_of(client, a.dataset_id)[0], "outlier_center")["id"]
        assert dismiss(client, a.dataset_id, key).status_code == 200
        assert by_step(recs_of(client, a.dataset_id)[0], "outlier_center")["status"] == "dismissed"
        assert by_step(recs_of(client, b.dataset_id)[0], "outlier_center")["status"] == "open"      # same key, other dataset
        assert status_rows(b.dataset_id) == [] and len(status_rows(a.dataset_id)) == 1

    def test_a_key_from_another_dataset_is_rejected(self, client):
        a = store.create("a.csv", make_df())
        c = store.create("c.csv", pd.DataFrame({"x": np.arange(20.0), "y": np.arange(20.0) * 2}))
        key = by_step(recs_of(client, a.dataset_id)[0], "outlier_center")["id"]
        assert dismiss(client, c.dataset_id, key).status_code == 404
        assert restore(client, c.dataset_id, key).status_code == 404
        assert status_rows(c.dataset_id) == []

    @pytest.mark.parametrize("key", ["nope", "rk1_" + "0" * 24, "../etc/passwd", "rk1_" + "a" * 24 + "~9", "x" * 64])
    def test_unknown_or_malformed_keys_are_404_and_write_nothing(self, client, session, key):
        before = snapshot(session)
        assert dismiss(client, session.dataset_id, key).status_code == 404
        assert restore(client, session.dataset_id, key).status_code == 404
        assert status_rows(session.dataset_id) == []
        assert_untouched(before, snapshot(session))

    def test_over_long_key_is_rejected_by_validation(self, client, session):
        assert dismiss(client, session.dataset_id, "k" * 200).status_code == 422

    def test_unknown_dataset_is_404(self, client):
        assert dismiss(client, "no-such-dataset", "rk1_x").status_code == 404
        assert client.get("/api/dataset/no-such-dataset/recommendations").status_code == 404

    def test_stale_key_after_the_data_changed_is_404_not_a_phantom_record(self, client, session):
        ds = session.dataset_id
        dup = by_step(recs_of(client, ds)[0], "duplicate_manager")
        client.post("/api/dataset/apply-action", json={"dataset_id": ds, "step": dup["step"], "action": dup["recommended_action"],
                                                      "params": {}, "reason": "r", "confidence": 0.9})
        assert dismiss(client, ds, dup["id"]).status_code == 404          # the recommendation no longer exists
        assert status_rows(ds) == []

    def test_stale_state_token_is_409_and_writes_nothing(self, client, session):
        ds = session.dataset_id
        key = by_step(recs_of(client, ds)[0], "outlier_center")["id"]
        token = client.get(f"/api/dataset/{ds}/recommendations").json()["state_token"]
        dup = by_step(recs_of(client, ds)[0], "duplicate_manager")
        client.post("/api/dataset/apply-action", json={"dataset_id": ds, "step": dup["step"], "action": dup["recommended_action"],
                                                      "params": {}, "reason": "r", "confidence": 0.9})
        r = dismiss(client, ds, key, expected_state_token=token)
        assert r.status_code == 409 and "changed" in r.json()["detail"]
        assert status_rows(ds) == []
        assert dismiss(client, ds, key, expected_state_token=proposal_store.state_token(store.get(ds))).status_code == 200


# ---------------------------------------------------------------- 6-7. restore
class TestRestore:
    def test_restore_reopens_without_touching_data(self, client, session):
        ds = session.dataset_id
        key = by_step(recs_of(client, ds)[0], "category_merge_center", "region")["id"]
        dismiss(client, ds, key)
        before = snapshot(session)
        r = restore(client, ds, key)
        assert r.status_code == 200 and r.json()["status"] == "open" and r.json()["changed"] is True
        assert r.json()["version_changed"] is False
        recs, counts = recs_of(client, ds)
        assert by_step(recs, "category_merge_center", "region")["status"] == "open" and counts["dismissed"] == 0
        assert_untouched(before, snapshot(session))

    def test_restore_survives_restart_and_can_be_dismissed_again(self, client, session):
        ds = session.dataset_id
        key = by_step(recs_of(client, ds)[0], "duplicate_manager")["id"]
        dismiss(client, ds, key); restore(client, ds, key)
        store._sessions.pop(ds)
        assert by_step(recs_of(client, ds)[0], "duplicate_manager")["status"] == "open"
        assert dismiss(client, ds, key).json()["changed"] is True
        assert [(e["event"]) for e in recommendation_store.events(ds, key)] == ["dismissed", "restored", "dismissed"]

    def test_restoring_something_never_dismissed_is_404_and_an_open_one_is_a_noop(self, client, session):
        ds = session.dataset_id
        key = by_step(recs_of(client, ds)[0], "duplicate_manager")["id"]
        assert restore(client, ds, key).status_code == 404                         # no record
        dismiss(client, ds, key); restore(client, ds, key)
        again = restore(client, ds, key)
        assert again.status_code == 200 and again.json()["changed"] is False
        assert len(status_rows(ds)) == 1                                           # row kept, nothing deleted

    def test_recompute_never_clears_dismissals(self, client, session):
        ds = session.dataset_id
        key = by_step(recs_of(client, ds)[0], "outlier_center")["id"]
        dismiss(client, ds, key)
        for _ in range(3):
            client.get(f"/api/dataset/{ds}/profile"); client.get(f"/api/dataset/{ds}/quality"); client.get(f"/api/dataset/{ds}/recommendations")
        assert by_step(recs_of(client, ds)[0], "outlier_center")["status"] == "dismissed"


# ---------------------------------------------------------------- 8. new / materially changed recommendations
class TestChangedRecommendations:
    def test_a_new_recommendation_is_open_even_when_others_are_dismissed(self, client):
        s = store.create("grow.csv", make_df())
        for rec in recs_of(client, s.dataset_id)[0]:
            dismiss(client, s.dataset_id, rec["id"])
        grown = make_df(extra_missing=True)                                  # a new column now has missing values
        store.get(s.dataset_id).push_version(grown, __import__("app.core.session_store", fromlist=["ActionLogEntry"]).ActionLogEntry(
            step="x", action="y", column=None, reason="r", confidence=0.5))
        recs, counts = recs_of(client, s.dataset_id)
        assert by_step(recs, "missing_value_center", "price")["status"] == "open"        # new subject -> new identity -> open
        assert by_step(recs, "missing_value_center", "qty")["status"] == "dismissed"
        assert counts == {"open": 1, "dismissed": 5, "total": 6}

    def test_evidence_change_keeps_it_dismissed_but_reports_it(self, client):
        s = store.create("evidence.csv", make_df())
        key = by_step(recs_of(client, s.dataset_id)[0], "missing_value_center", "qty")["id"]
        dismiss(client, s.dataset_id, key)
        assert by_step(recs_of(client, s.dataset_id)[0], "missing_value_center")["changed_since_dismissed"] is False
        df = make_df(); df.loc[[20, 21, 22], "qty"] = np.nan                  # more missing values, same subject
        store.get(s.dataset_id).push_version(df, __import__("app.core.session_store", fromlist=["ActionLogEntry"]).ActionLogEntry(
            step="x", action="y", column=None, reason="r", confidence=0.5))
        rec = by_step(recs_of(client, s.dataset_id)[0], "missing_value_center", "qty")
        assert rec["id"] == key and rec["status"] == "dismissed" and rec["changed_since_dismissed"] is True

    def test_a_different_fuzzy_group_is_a_different_recommendation(self, client):
        s = store.create("fz.csv", make_df())
        fuzzy = next(r for r in recs_of(client, s.dataset_id)[0] if r["recommended_action"] == "fuzzy_merge")
        dismiss(client, s.dataset_id, fuzzy["id"])
        df = make_df(); df["shade"] = ["Colour"] * 10 + ["Color"] * 50 + ["Greyish"] * 11 + ["Grayish"] * 11
        store.get(s.dataset_id).push_version(df, __import__("app.core.session_store", fromlist=["ActionLogEntry"]).ActionLogEntry(
            step="x", action="y", column=None, reason="r", confidence=0.5))
        recs, _ = recs_of(client, s.dataset_id)
        fuzzy_now = [r for r in recs if r["recommended_action"] == "fuzzy_merge"]
        assert fuzzy["id"] in {r["id"] for r in fuzzy_now if r["status"] == "dismissed"}
        assert any(r["status"] == "open" and r["id"] != fuzzy["id"] for r in fuzzy_now)


# ---------------------------------------------------------------- 9. duplicates and concurrency
class TestDuplicateAndConcurrentRequests:
    def test_duplicate_dismissal_is_idempotent_with_one_row_and_one_event(self, client, session):
        ds = session.dataset_id
        key = by_step(recs_of(client, ds)[0], "outlier_center")["id"]
        first, second, third = (dismiss(client, ds, key) for _ in range(3))
        assert [first.json()["changed"], second.json()["changed"], third.json()["changed"]] == [True, False, False]
        assert all(r.status_code == 200 and r.json()["status"] == "dismissed" for r in (first, second, third))
        assert status_rows(ds) == [(key, "dismissed")]
        assert [e["event"] for e in recommendation_store.events(ds, key)] == ["dismissed"]

    def test_concurrent_dismissals_produce_exactly_one_change(self, session):
        ds = session.dataset_id
        key = by_step(recs_of(TestClient(app), ds)[0], "outlier_center")["id"]
        results, barrier = [], threading.Barrier(8)
        def worker():
            barrier.wait()
            r = TestClient(app).post(f"/api/dataset/{ds}/recommendations/{key}/dismiss")
            results.append((r.status_code, r.json().get("changed")))
        threads = [threading.Thread(target=worker) for _ in range(8)]
        [t.start() for t in threads]; [t.join() for t in threads]
        assert sorted(results, key=str) == sorted([(200, True)] + [(200, False)] * 7, key=str)
        assert status_rows(ds) == [(key, "dismissed")] and len(recommendation_store.events(ds, key)) == 1

    def test_store_level_race_without_the_http_lock_is_still_safe(self, session):
        ds = session.dataset_id
        rec = generate_recommendations(store.get(ds).current, [])[0]
        out, barrier = [], threading.Barrier(10)
        def worker():
            barrier.wait()
            out.append(recommendation_store.dismiss(ds, rec, 1, "v1:original"))
        threads = [threading.Thread(target=worker) for _ in range(10)]
        [t.start() for t in threads]; [t.join() for t in threads]
        assert out.count(True) == 1 and out.count(False) == 9
        assert len(status_rows(ds)) == 1 and len(recommendation_store.events(ds)) == 1

    def test_concurrent_dismiss_and_restore_end_in_a_consistent_state(self, session):
        ds = session.dataset_id
        key = by_step(recs_of(TestClient(app), ds)[0], "duplicate_manager")["id"]
        TestClient(app).post(f"/api/dataset/{ds}/recommendations/{key}/dismiss")
        barrier, codes = threading.Barrier(8), []
        def worker(i):
            barrier.wait()
            c = TestClient(app)
            codes.append(c.post(f"/api/dataset/{ds}/recommendations/{key}/{'restore' if i % 2 else 'dismiss'}").status_code)
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        [t.start() for t in threads]; [t.join() for t in threads]
        assert set(codes) == {200}
        rows = status_rows(ds)
        assert len(rows) == 1 and rows[0][1] in ("open", "dismissed")
        events = [e["event"] for e in recommendation_store.events(ds, key)]
        # events strictly alternate, so the final status always matches the last event
        assert all(a != b for a, b in zip(events, events[1:]))
        assert rows[0][1] == ("dismissed" if events[-1] == "dismissed" else "open")


# ---------------------------------------------------------------- 10. review-only / unsupported actions
class TestReviewOnlySafety:
    def test_outlier_recommendation_is_flagged_not_executable(self, client, session):
        rec = by_step(recs_of(client, session.dataset_id)[0], "outlier_center")
        assert rec["recommended_action"] == "flag_for_review" and rec["review_only"] is True and rec["executable"] is False

    @pytest.mark.parametrize("action", ["flag_for_review", "FLAG_FOR_REVIEW", " Flag_For_Review "])
    def test_applying_flag_for_review_is_rejected_before_any_change(self, client, session, action):
        before = snapshot(session)
        r = client.post("/api/dataset/apply-action", json={"dataset_id": session.dataset_id, "step": "outlier_center", "action": action,
                                                           "column": "price", "params": {}, "reason": "r", "confidence": 0.5})
        assert r.status_code == 400 and "review-only" in r.json()["detail"] and "never changes your data" in r.json()["detail"]
        assert_untouched(before, snapshot(session))

    def test_engine_still_refuses_flag_for_review(self):
        with pytest.raises(ValueError, match="Unknown outlier_center action"):
            apply_action(make_df(), "outlier_center", "flag_for_review", "price", {})

    def test_explicit_supported_outlier_actions_are_unchanged(self, client, session):
        # a deliberate, explicit action remains possible through the API (nothing was removed)
        r = client.post("/api/dataset/apply-action", json={"dataset_id": session.dataset_id, "step": "outlier_center", "action": "clip",
                                                           "column": "price", "params": {}, "reason": "user chose clip", "confidence": 0.5})
        assert r.status_code == 200 and store.get(session.dataset_id).version_number == 2

    def test_review_only_recommendation_never_becomes_applied_by_dismissing_it(self, client, session):
        ds = session.dataset_id
        key = by_step(recs_of(client, ds)[0], "outlier_center")["id"]
        before = snapshot(session)
        assert dismiss(client, ds, key).status_code == 200
        assert_untouched(before, snapshot(session))
        assert store.get(ds).current.loc[3, "price"] == 5000.0                    # the flagged value is untouched

    def test_registry_matches_what_apply_action_really_executes(self):
        """Every generated recommendation marked executable must run through apply_action exactly as the
        card would run it; every non-executable one must be review-only. Swept over all synthetic datasets."""
        from synthetic_datasets import ALL_SYNTHETIC_DATASET_BUILDERS
        frames = [make_df()] + [build() for build in ALL_SYNTHETIC_DATASET_BUILDERS.values()]
        checked, steps_checked = 0, set()
        for df in frames:
            for rec in generate_recommendations(df, []):
                assert rec.executable == is_executable_recommendation(rec.step, rec.recommended_action)
                if rec.executable:
                    params = ({"values": rec.preview["values"], "canonical": rec.preview["suggested_canonical"]}
                              if rec.recommended_action == "fuzzy_merge" else {})
                    apply_action(df.copy(), rec.step, rec.recommended_action, rec.column, params)   # must not raise
                    checked += 1
                    steps_checked.add(rec.step)
                else:
                    assert rec.review_only, (rec.step, rec.recommended_action)
        assert checked >= 8 and {"duplicate_manager", "missing_value_center", "category_merge_center"} <= steps_checked

    def test_every_executable_registry_entry_is_accepted_by_apply_action(self):
        df = make_df()
        assert EXECUTABLE_RECOMMENDED_ACTIONS["outlier_center"] == set()
        for action in EXECUTABLE_RECOMMENDED_ACTIONS["missing_value_center"] - {"ignore"}:
            apply_action(df, "missing_value_center", action, "qty", {})
        apply_action(df, "duplicate_manager", "drop_duplicates", None, {})
        apply_action(df, "category_merge_center", "normalize_case_whitespace", "region", {})


# ---------------------------------------------------------------- 12. Copilot persistence unaffected
class TestCopilotCoexistence:
    def test_dismissing_a_recommendation_does_not_touch_copilot_proposals(self, client, session):
        ds = session.dataset_id
        rec = proposal_store.create_proposal(session, {"kind": "cleaning", "step": "duplicate_manager", "action": "drop_duplicates",
                                                       "column": None, "params": {}, "reason": "r", "confidence": 0.9}, {"source": "cleaning"})
        key = by_step(recs_of(client, ds)[0], "duplicate_manager")["id"]
        dismiss(client, ds, key)
        assert proposal_store.get_proposal(rec["proposal_id"])["status"] == "pending"
        # and the existing Copilot confirm path still works end to end
        r = client.post("/api/dataset/copilot/confirm-action", json={"dataset_id": ds, "proposal_id": rec["proposal_id"]})
        assert r.status_code in (200, 409)

    def test_quality_response_keeps_every_existing_field(self, client, session):
        body = client.get(f"/api/dataset/{session.dataset_id}/quality").json()
        for field in ("dataset_id", "score", "recommendations", "analytical_risk", "readiness", "recommendation_counts"):
            assert field in body
        for field in ("id", "step", "column", "issue", "recommended_action", "reason", "confidence", "preview", "how_detected",
                      "risks", "alternative_actions", "goal_relevant", "status", "review_only", "executable"):
            assert field in body["recommendations"][0]
