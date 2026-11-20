"""Phase 2 - Copilot proposal authenticity (F3), review-only enforcement (F4) and
server-side lifecycle (F5). Runs against the real FastAPI app, session store and
SQLite registry; only the LLM call (send_message) is replaced by a stub."""
import threading

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core import action_policy, proposal_store
from app.core.session_store import store
from app.engines import copilot_proposals
from app.engines.copilot_proposals import candidate_key, trusted_candidates
from app.main import app
from app.routers import copilot as copilot_router

DUP_ACTION = {"kind": "cleaning", "step": "duplicate_manager", "action": "drop_duplicates", "column": None,
              "params": {}, "reason": "AI invented reason", "confidence": 0.99}


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def dup_session():
    return store.create("proposal_test.csv", pd.DataFrame({"name": ["A", "B", "A", "C"], "value": [10, 20, 10, 30]}))


def _stub_llm(monkeypatch, proposed_action, reply="Here is my suggestion."):
    async def fake(**kwargs):
        return {"reply": reply, "proposed_action": proposed_action, "has_more_suggestions": True,
                "unverified_figures": [], "grounded": True}
    monkeypatch.setattr(copilot_router, "send_message", fake)


def _ask(client, session, monkeypatch, ai_action):
    _stub_llm(monkeypatch, ai_action)
    r = client.post("/api/dataset/copilot/message", json={"dataset_id": session.dataset_id, "message": "what next?"})
    assert r.status_code == 200, r.text
    return r.json()


def _confirm(client, session, **body):
    return client.post("/api/dataset/copilot/confirm-action", json={"dataset_id": session.dataset_id, **body})


def _dismiss(client, dataset_id, proposal_id):
    return client.post("/api/dataset/copilot/dismiss-action", json={"dataset_id": dataset_id, "proposal_id": proposal_id})


def _unchanged(session, version=1):
    s = store.get(session.dataset_id)
    assert s.version_number == version and len(s.log) == version - 1


# ---------------------------------------------------------------- policy (F4)
class TestActionPolicy:
    @pytest.mark.parametrize("a", ["flag_for_review", "FLAG_FOR_REVIEW", " Flag_For_Review "])
    def test_review_only_is_case_insensitive(self, a):
        assert action_policy.is_review_only(a) and not action_policy.is_copilot_executable(a)

    def test_fuzzy_merge_blocked_for_copilot(self):
        assert action_policy.is_copilot_blocked("fuzzy_merge") and not action_policy.is_copilot_executable("Fuzzy_Merge")

    @pytest.mark.parametrize("a", [None, "", 5, ["clip"]])
    def test_invalid_actions_not_executable(self, a):
        assert not action_policy.is_copilot_executable(a)

    def test_normal_actions_executable(self):
        assert action_policy.is_copilot_executable("drop_duplicates")


# ---------------------------------------------------------------- valid flow (F3/F5)
class TestValidCleaningFlow:
    def test_proposal_is_rebuilt_from_backend_recommendation(self, client, dup_session, monkeypatch):
        body = _ask(client, dup_session, monkeypatch, DUP_ACTION)
        p = body["proposed_action"]
        assert p["proposal_id"] and p["status"] == "pending" and p["dataset_version"] == 1 and p["expires_at"]
        assert p["reason"] != "AI invented reason" and p["confidence"] != 0.99      # backend values, not the LLM's
        assert p["params"] == {} and p["step"] == "duplicate_manager" and p["action"] == "drop_duplicates"
        assert body["proposal_note"] is None
        _unchanged(dup_session)                                                       # proposing never mutates

    def test_confirm_by_proposal_id_applies_once(self, client, dup_session, monkeypatch):
        pid = _ask(client, dup_session, monkeypatch, DUP_ACTION)["proposed_action"]["proposal_id"]
        r = _confirm(client, dup_session, proposal_id=pid)
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["version"] == 2 and d["rows"] == 3 and d["status"] == "confirmed" and d["proposal_id"] == pid and d["result_summary"]
        s = store.get(dup_session.dataset_id)
        assert len(s.log) == 1 and s.log[0].reason != "AI invented reason"           # notebook logs the trusted reason
        assert proposal_store.get_proposal(pid)["status"] == "confirmed"

    def test_echoing_the_proposal_back_is_allowed(self, client, dup_session, monkeypatch):
        p = _ask(client, dup_session, monkeypatch, DUP_ACTION)["proposed_action"]
        assert _confirm(client, dup_session, proposal_id=p["proposal_id"], proposed_action=p).status_code == 200

    def test_legacy_confirm_still_works_but_ignores_client_metadata(self, client, dup_session):
        r = _confirm(client, dup_session, proposed_action={**DUP_ACTION, "reason": "malicious", "confidence": 0.01})
        assert r.status_code == 200, r.text
        log = store.get(dup_session.dataset_id).log[0]
        assert log.reason != "malicious" and log.confidence != 0.01

    def test_confirm_requires_something_to_confirm(self, client, dup_session):
        assert _confirm(client, dup_session).status_code == 400


# ---------------------------------------------------------------- authenticity (F3)
class TestFabricatedAndAltered:
    @pytest.mark.parametrize("fake", [
        {"kind": "cleaning", "step": "missing_value_center", "action": "drop_column", "column": "revenue", "params": {}},
        {"kind": "cleaning", "step": "duplicate_manager", "action": "drop_rows", "params": {}},
        {"kind": "cleaning", "step": "made_up_step", "action": "drop_duplicates"},
        {"kind": "feature_engineering", "category": "ratio", "action": "create_ratio", "columns": ["name", "nope"], "params": {}},
        {"kind": "delete_everything", "action": "x"},
        {"kind": "cleaning", "step": "duplicate_manager", "action": "drop_duplicates", "params": {"keep": "last"}},  # altered params
    ])
    def test_fabricated_proposal_is_dropped_with_a_reason(self, client, dup_session, monkeypatch, fake):
        body = _ask(client, dup_session, monkeypatch, fake)
        assert body["proposed_action"] is None and body["proposal_note"]
        assert "Note: no executable action was offered" in body["reply"]

    @pytest.mark.parametrize("junk", ["drop everything", ["a", "b"], 42, {}, {"kind": "cleaning"}])
    def test_invalid_ai_output_cannot_bypass_validation(self, client, dup_session, monkeypatch, junk):
        body = _ask(client, dup_session, monkeypatch, junk)
        assert body["proposed_action"] is None
        _unchanged(dup_session)

    def test_llm_failure_is_a_503_not_a_mutation(self, client, dup_session, monkeypatch):
        async def boom(**kw): raise RuntimeError("Ollama down")
        monkeypatch.setattr(copilot_router, "send_message", boom)
        r = client.post("/api/dataset/copilot/message", json={"dataset_id": dup_session.dataset_id, "message": "x"})
        assert r.status_code == 503
        _unchanged(dup_session)

    def test_modified_action_on_confirm_is_rejected_and_proposal_stays_pending(self, client, dup_session, monkeypatch):
        pid = _ask(client, dup_session, monkeypatch, DUP_ACTION)["proposed_action"]["proposal_id"]
        for evil in ({**DUP_ACTION, "action": "drop_duplicates", "step": "missing_value_center", "column": "value"},
                     {**DUP_ACTION, "params": {"keep": "last"}}):
            assert _confirm(client, dup_session, proposal_id=pid, proposed_action=evil).status_code == 400
        _unchanged(dup_session)
        assert proposal_store.get_proposal(pid)["status"] == "pending"
        assert _confirm(client, dup_session, proposal_id=pid).status_code == 200

    def test_legacy_confirm_with_fabricated_action_is_rejected(self, client, dup_session):
        r = _confirm(client, dup_session, proposed_action={"kind": "cleaning", "step": "missing_value_center",
                                                          "action": "drop_column", "column": "value"})
        assert r.status_code == 400 and r.json()["detail"]
        _unchanged(dup_session)

    def test_unknown_proposal_id(self, client, dup_session):
        assert _confirm(client, dup_session, proposal_id="does-not-exist").status_code == 404

    def test_wrong_dataset_is_rejected(self, client, dup_session, monkeypatch):
        other = store.create("other.csv", pd.DataFrame({"name": ["A", "B", "A"], "value": [1, 2, 1]}))
        pid = _ask(client, dup_session, monkeypatch, DUP_ACTION)["proposed_action"]["proposal_id"]
        assert _confirm(client, other, proposal_id=pid).status_code == 404
        assert _dismiss(client, other.dataset_id, pid).status_code == 404
        _unchanged(other)
        _unchanged(dup_session)
        assert proposal_store.get_proposal(pid)["status"] == "pending"


# ---------------------------------------------------------------- stale / expiry (F5)
class TestStaleAndExpiry:
    def test_stale_after_dataset_changes(self, client, dup_session, monkeypatch):
        pid = _ask(client, dup_session, monkeypatch, DUP_ACTION)["proposed_action"]["proposal_id"]
        # a different, manual mutation moves the dataset on
        r = client.post("/api/dataset/apply-action", json={"dataset_id": dup_session.dataset_id, "step": "duplicate_manager",
                                                           "action": "drop_duplicates", "params": {}, "reason": "manual", "confidence": 0.9})
        assert r.status_code == 200
        c = _confirm(client, dup_session, proposal_id=pid)
        assert c.status_code == 409 and "stale" in c.json()["detail"]
        assert proposal_store.get_proposal(pid)["status"] == "expired"
        _unchanged(dup_session, 2)

    def test_same_version_number_after_undo_and_new_action_is_still_stale(self, client, monkeypatch):
        df = pd.DataFrame({"name": ["A", "B", "A", "C"], "value": [10.0, np.nan, 10.0, 30.0]})
        s = store.create("undo_stale.csv", df)
        client.post("/api/dataset/apply-action", json={"dataset_id": s.dataset_id, "step": "duplicate_manager", "action": "drop_duplicates",
                                                       "params": {}, "reason": "m", "confidence": 0.9})
        cand = next(c for c in trusted_candidates(store.get(s.dataset_id).current, []) if c["kind"] == "cleaning")
        ai = {k: cand[k] for k in ("kind", "step", "action", "column")} | {"params": {}}
        pid = _ask(client, s, monkeypatch, ai)["proposed_action"]["proposal_id"]          # created at v2
        client.post(f"/api/dataset/{s.dataset_id}/undo")                                      # back to v1 ...
        client.post("/api/dataset/apply-action", json={"dataset_id": s.dataset_id, "step": "duplicate_manager", "action": "drop_duplicates",
                                                       "params": {}, "reason": "again", "confidence": 0.9})  # ... and v2 again, new state
        assert store.get(s.dataset_id).version_number == 2
        assert _confirm(client, s, proposal_id=pid).status_code == 409

    def test_ttl_expiry(self, client, dup_session, monkeypatch):
        monkeypatch.setattr(proposal_store, "PROPOSAL_TTL_SECONDS", -5)
        pid = _ask(client, dup_session, monkeypatch, DUP_ACTION)["proposed_action"]["proposal_id"]
        r = _confirm(client, dup_session, proposal_id=pid)
        assert r.status_code == 410 and "expired" in r.json()["detail"]
        _unchanged(dup_session)

    def test_dataset_changing_during_llm_call_drops_the_proposal(self, client, dup_session, monkeypatch):
        async def slow(**kw):
            store.get(dup_session.dataset_id).push_version(
                store.get(dup_session.dataset_id).current.copy(),
                __import__("app.core.session_store", fromlist=["ActionLogEntry"]).ActionLogEntry(
                    step="x", action="y", column=None, reason="r", confidence=0.5))
            return {"reply": "ok", "proposed_action": DUP_ACTION, "has_more_suggestions": False, "unverified_figures": [], "grounded": True}
        monkeypatch.setattr(copilot_router, "send_message", slow)
        body = client.post("/api/dataset/copilot/message", json={"dataset_id": dup_session.dataset_id, "message": "x"}).json()
        assert body["proposed_action"] is None and "changed" in body["proposal_note"]

    def test_proposals_survive_a_restart(self, client, dup_session, monkeypatch):
        pid = _ask(client, dup_session, monkeypatch, DUP_ACTION)["proposed_action"]["proposal_id"]
        store._sessions.pop(dup_session.dataset_id)                   # simulate a fresh process: cache empty
        assert proposal_store.get_proposal(pid)["status"] == "pending"
        assert _confirm(client, dup_session, proposal_id=pid).status_code == 200   # session reloaded from disk


# ---------------------------------------------------------------- duplicates, replay, dismissal, concurrency
class TestLifecycle:
    def test_duplicate_confirmation_is_rejected(self, client, dup_session, monkeypatch):
        pid = _ask(client, dup_session, monkeypatch, DUP_ACTION)["proposed_action"]["proposal_id"]
        assert _confirm(client, dup_session, proposal_id=pid).status_code == 200
        again = _confirm(client, dup_session, proposal_id=pid)
        assert again.status_code == 409 and "already confirmed" in again.json()["detail"]
        _unchanged(dup_session, 2)

    def test_dismiss_then_confirm_is_rejected(self, client, dup_session, monkeypatch):
        pid = _ask(client, dup_session, monkeypatch, DUP_ACTION)["proposed_action"]["proposal_id"]
        d = _dismiss(client, dup_session.dataset_id, pid)
        assert d.status_code == 200 and d.json()["status"] == "dismissed"
        assert _dismiss(client, dup_session.dataset_id, pid).json()["status"] == "dismissed"   # idempotent
        c = _confirm(client, dup_session, proposal_id=pid)
        assert c.status_code == 409 and "dismissed" in c.json()["detail"]
        _unchanged(dup_session)

    def test_cannot_dismiss_a_confirmed_proposal(self, client, dup_session, monkeypatch):
        pid = _ask(client, dup_session, monkeypatch, DUP_ACTION)["proposed_action"]["proposal_id"]
        _confirm(client, dup_session, proposal_id=pid)
        assert _dismiss(client, dup_session.dataset_id, pid).status_code == 409
        assert proposal_store.get_proposal(pid)["status"] == "confirmed"

    def test_failed_execution_is_not_marked_confirmed(self, client, dup_session, monkeypatch):
        pid = _ask(client, dup_session, monkeypatch, DUP_ACTION)["proposed_action"]["proposal_id"]
        def boom(*a, **k): raise ValueError("engine exploded")
        monkeypatch.setattr("app.routers.cleaning.apply_action", boom)
        r = _confirm(client, dup_session, proposal_id=pid)
        assert r.status_code == 400
        assert proposal_store.get_proposal(pid)["status"] == "rejected"
        _unchanged(dup_session)

    def test_unexpected_error_releases_the_proposal(self, client, dup_session, monkeypatch):
        pid = _ask(client, dup_session, monkeypatch, DUP_ACTION)["proposed_action"]["proposal_id"]
        monkeypatch.setattr("app.routers.cleaning.apply_action", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk")))
        with pytest.raises(RuntimeError):
            TestClient(app, raise_server_exceptions=True).post("/api/dataset/copilot/confirm-action",
                                                                json={"dataset_id": dup_session.dataset_id, "proposal_id": pid})
        assert proposal_store.get_proposal(pid)["status"] == "pending"   # nothing was applied, so it stays confirmable
        _unchanged(dup_session)

    def test_concurrent_confirmations_mutate_exactly_once(self, dup_session, monkeypatch):
        c = TestClient(app)
        pid = _ask(c, dup_session, monkeypatch, DUP_ACTION)["proposed_action"]["proposal_id"]
        codes, barrier = [], threading.Barrier(8)
        def worker():
            barrier.wait()
            codes.append(TestClient(app).post("/api/dataset/copilot/confirm-action",
                                              json={"dataset_id": dup_session.dataset_id, "proposal_id": pid}).status_code)
        threads = [threading.Thread(target=worker) for _ in range(8)]
        [t.start() for t in threads]; [t.join() for t in threads]
        assert sorted(codes) == [200] + [409] * 7
        _unchanged(dup_session, 2)

    def test_concurrent_legacy_confirmations_mutate_exactly_once(self, dup_session):
        codes, barrier = [], threading.Barrier(6)
        def worker():
            barrier.wait()
            codes.append(TestClient(app).post("/api/dataset/copilot/confirm-action",
                                              json={"dataset_id": dup_session.dataset_id, "proposed_action": DUP_ACTION}).status_code)
        threads = [threading.Thread(target=worker) for _ in range(6)]
        [t.start() for t in threads]; [t.join() for t in threads]
        assert codes.count(200) == 1 and all(c in (200, 400, 409) for c in codes)
        _unchanged(dup_session, 2)


# ---------------------------------------------------------------- review-only enforcement (F4)
@pytest.fixture
def outlier_session(messy_df):
    return store.create("outlier_proposal.csv", messy_df)


class TestReviewOnly:
    def _flag(self):
        return {"kind": "cleaning", "step": "outlier_center", "action": "flag_for_review", "column": "revenue", "params": {}}

    def test_precondition_outlier_recommendation_is_review_only(self, outlier_session):
        from app.engines.cleaning_engine import generate_recommendations
        recs = [r for r in generate_recommendations(outlier_session.current, []) if r.step == "outlier_center"]
        assert recs and all(r.recommended_action == "flag_for_review" for r in recs)
        assert not any(c["step"] == "outlier_center" for c in trusted_candidates(outlier_session.current, []) if c["kind"] == "cleaning")

    @pytest.mark.parametrize("action", ["flag_for_review", "Flag_For_Review", "  FLAG_FOR_REVIEW "])
    def test_flag_for_review_never_becomes_a_proposal(self, client, outlier_session, monkeypatch, action):
        body = _ask(client, outlier_session, monkeypatch, {**self._flag(), "action": action})
        assert body["proposed_action"] is None and "review-only" in body["proposal_note"]

    def test_flag_for_review_with_missing_or_odd_kind_is_still_dropped(self, client, outlier_session, monkeypatch):
        for ai in ({"step": "outlier_center", "action": "flag_for_review", "column": "revenue"},
                   {"kind": "feature_engineering", "action": "flag_for_review"}):
            assert _ask(client, outlier_session, monkeypatch, ai)["proposed_action"] is None

    @pytest.mark.parametrize("swap", ["clip", "drop_rows", "convert_to_missing"])
    def test_cannot_swap_review_only_for_an_executable_outlier_action(self, client, outlier_session, monkeypatch, swap):
        body = _ask(client, outlier_session, monkeypatch, {**self._flag(), "action": swap})
        assert body["proposed_action"] is None and "review-only" in body["proposal_note"]
        _unchanged(outlier_session)

    @pytest.mark.parametrize("action", ["flag_for_review", "clip", "drop_rows", "convert_to_missing"])
    def test_legacy_confirm_cannot_execute_outlier_actions(self, client, outlier_session, action):
        assert _confirm(client, outlier_session, proposed_action={**self._flag(), "action": action}).status_code == 400
        _unchanged(outlier_session)

    def test_fuzzy_merge_is_not_executable_through_copilot(self, client, monkeypatch):
        s = store.create("fuzzy_cp.csv", pd.DataFrame({"c": ["Colour"] * 5 + ["Color"] * 9 + ["Blue"] * 3}))
        ai = {"kind": "cleaning", "step": "category_merge_center", "action": "fuzzy_merge", "column": "c",
              "params": {"values": ["Colour", "Color"], "canonical": "Color"}}
        body = _ask(client, s, monkeypatch, ai)
        assert body["proposed_action"] is None and "review-only" in body["proposal_note"]
        r = _confirm(client, s, proposed_action=ai)
        assert r.status_code == 400 and "fuzzy_merge" in r.json()["detail"]
        _unchanged(s)

    def test_has_more_ignores_review_only_recommendations(self, client, monkeypatch):
        df = pd.DataFrame({"x": list(np.linspace(90, 110, 60)) + [5000.0]})
        s = store.create("only_outliers.csv", df)
        from app.engines.cleaning_engine import generate_recommendations
        assert any(r.recommended_action == "flag_for_review" for r in generate_recommendations(df, []))   # review-only exists...
        monkeypatch.setattr(copilot_proposals, "generate_feature_recommendations", lambda *a, **k: [])
        body = _ask(client, s, monkeypatch, None)
        assert body["has_more_suggestions"] is False                   # ...but it is not "more suggestions"
        assert copilot_router._has_more_suggestions(s) is False

    def test_has_more_true_when_an_executable_recommendation_remains(self, client, dup_session, monkeypatch):
        assert _ask(client, dup_session, monkeypatch, None)["has_more_suggestions"] is True

    def test_llm_has_more_claim_is_not_trusted(self, client, monkeypatch):
        s = store.create("clean_cp.csv", pd.DataFrame({"x": np.linspace(0, 1, 50)}))
        monkeypatch.setattr(copilot_proposals, "generate_feature_recommendations", lambda *a, **k: [])
        assert _ask(client, s, monkeypatch, None)["has_more_suggestions"] is False   # stub said True


# ---------------------------------------------------------------- feature engineering
class TestFeatureEngineeringFlow:
    def _feature(self, df):
        cands = [c for c in trusted_candidates(df, []) if c["kind"] == "feature_engineering"]
        assert cands, "fixture should yield at least one feature recommendation"
        return cands[0]

    def test_valid_feature_proposal_confirms(self, client, messy_df, monkeypatch):
        s = store.create("feature_cp.csv", messy_df)
        c = self._feature(s.current)
        ai = {"kind": "feature_engineering", "category": c["category"], "action": c["action"], "columns": c["columns"],
              "new_column_name": c["new_column_name"], "params": {}, "reason": "x", "confidence": 1.0}
        p = _ask(client, s, monkeypatch, ai)["proposed_action"]
        assert p and p["proposal_id"] and p["reason"] == c["reason"]
        r = _confirm(client, s, proposal_id=p["proposal_id"])
        assert r.status_code == 200, r.text
        assert r.json()["columns"] > len(messy_df.columns) and r.json()["version"] == 2

    def test_altered_feature_columns_or_name_are_rejected(self, client, messy_df, monkeypatch):
        s = store.create("feature_cp2.csv", messy_df)
        c = self._feature(s.current)
        base = {"kind": "feature_engineering", "category": c["category"], "action": c["action"], "columns": c["columns"],
                "new_column_name": c["new_column_name"], "params": {}}
        for bad in ({**base, "columns": ["nope"]}, {**base, "new_column_name": "hijacked"}, {**base, "params": {"x": 1}},
                    {**base, "action": "label_encode" if c["action"] != "label_encode" else "one_hot_encode"}):
            assert _ask(client, s, monkeypatch, bad)["proposed_action"] is None
        _unchanged(s)


# ---------------------------------------------------------------- registry atomicity (cross-process safety)
class TestRegistryCompareAndSet:
    """The per-dataset lock serialises one process; the SQL compare-and-set is what keeps a second
    worker process safe, so it is tested directly, without the lock."""

    def _pending(self, dup_session):
        return proposal_store.create_proposal(dup_session, {"kind": "cleaning", "step": "duplicate_manager",
                                                            "action": "drop_duplicates", "column": None, "params": {},
                                                            "reason": "r", "confidence": 0.9}, {"source": "cleaning"})

    def test_only_one_transition_wins(self, dup_session):
        pid = self._pending(dup_session)["proposal_id"]
        assert proposal_store.transition(pid, "pending", "confirming") is True
        assert proposal_store.transition(pid, "pending", "confirming") is False
        assert proposal_store.transition(pid, "pending", "dismissed") is False
        assert proposal_store.get_proposal(pid)["status"] == "confirming"

    def test_concurrent_claims_have_exactly_one_winner_without_the_lock(self, dup_session):
        pid = self._pending(dup_session)["proposal_id"]
        wins, barrier = [], threading.Barrier(10)
        def worker():
            barrier.wait()
            wins.append(proposal_store.transition(pid, "pending", "confirming"))
        threads = [threading.Thread(target=worker) for _ in range(10)]
        [t.start() for t in threads]; [t.join() for t in threads]
        assert wins.count(True) == 1 and wins.count(False) == 9

    def test_dismiss_loses_to_a_confirmation_in_flight(self, dup_session):
        pid = self._pending(dup_session)["proposal_id"]
        assert proposal_store.transition(pid, "pending", "confirming")
        assert not proposal_store.transition(pid, "pending", "dismissed")

    def test_expiry_is_lazy_and_one_way(self, dup_session, monkeypatch):
        monkeypatch.setattr(proposal_store, "PROPOSAL_TTL_SECONDS", -1)
        pid = self._pending(dup_session)["proposal_id"]
        assert proposal_store.get_proposal(pid)["status"] == "expired"
        assert not proposal_store.transition(pid, "pending", "confirming")
