import json
import nbformat
import pytest
from app.core.session_store import SessionStore, ActionLogEntry
from app.engines.cleaning_engine import apply_action
from app.engines.report_engine import generate_report


@pytest.fixture
def report_session(tmp_path, monkeypatch, messy_df):
    monkeypatch.setenv("SG_INSIGHT_DATA_DIR", str(tmp_path))
    import app.core.persistence as persistence
    monkeypatch.setattr(persistence, "DATA_DIR", tmp_path)
    monkeypatch.setattr(persistence, "DB_PATH", tmp_path / "sessions.db")
    monkeypatch.setattr(persistence, "VERSIONS_DIR", tmp_path / "versions")

    store = SessionStore()
    session = store.create(filename="sales.csv", df=messy_df, investigation_goal="find why revenue is volatile")
    new_df, preview = apply_action(session.current, "missing_value_center", "impute_median", "cost", {})
    entry = ActionLogEntry(step="missing_value_center", action="impute_median", column="cost",
                            reason="4 missing values", confidence=0.8, result_summary=str(preview))
    session.push_version(new_df, entry)
    return session


class TestGenerateReportFormats:
    def test_markdown_contains_key_sections(self, report_session):
        content = generate_report(report_session, format="markdown")
        assert isinstance(content, str)
        assert "sales.csv" in content
        assert "find why revenue is volatile" in content

    def test_html_is_well_formed_and_contains_data(self, report_session):
        content = generate_report(report_session, format="html")
        assert "<html" in content.lower()
        assert "sales.csv" in content

    def test_pdf_produces_nonempty_bytes_with_pdf_header(self, report_session):
        content = generate_report(report_session, format="pdf")
        assert isinstance(content, (bytes, bytearray))
        assert content[:4] == b"%PDF"
        assert len(content) > 1000

    def test_notebook_produces_valid_executable_nbformat(self, report_session):
        content = generate_report(report_session, format="notebook")
        nb = nbformat.reads(content, as_version=4)
        nbformat.validate(nb)
        assert len(nb.cells) > 0

    def test_unknown_format_raises(self, report_session):
        with pytest.raises(ValueError, match="Unknown report format"):
            generate_report(report_session, format="not_a_real_format")


class TestReportIncludesSessionData:
    def test_cleaning_log_action_appears_in_report(self, report_session):
        content = generate_report(report_session, format="markdown")
        assert "impute_median" in content or "missing_value_center" in content


class TestSectionOrder:
    """Session 13 (Smart Report Builder): section_order lets the caller
    choose which sections appear and in what order, for all three summary
    formats. None (the default, used by the pre-existing generate flow)
    must still produce every section — this is what the tests above
    already exercise implicitly; these add the filtering behavior itself."""

    def test_markdown_excludes_sections_not_in_order(self, report_session):
        full = generate_report(report_session, format="markdown")
        assert "## Appendix" in full  # sanity: present by default

        limited = generate_report(report_session, format="markdown",
                                    section_order=["executive_summary", "quality_report"])
        assert "## Executive Summary" in limited
        assert "## Data Quality Report" in limited
        assert "## Appendix" not in limited
        assert "## EDA Findings" not in limited

    def test_markdown_respects_custom_order(self, report_session):
        content = generate_report(report_session, format="markdown",
                                    section_order=["limitations", "executive_summary"])
        assert content.index("## Limitations") < content.index("## Executive Summary")

    def test_html_excludes_sections_not_in_order(self, report_session):
        limited = generate_report(report_session, format="html", section_order=["executive_summary"])
        assert "Executive Summary" in limited
        assert "Appendix" not in limited
        assert "EDA Findings" not in limited

    def test_pdf_with_limited_sections_still_valid(self, report_session):
        content = generate_report(report_session, format="pdf", section_order=["executive_summary", "limitations"])
        assert content[:4] == b"%PDF"
        assert len(content) > 500

    def test_empty_section_order_produces_header_only(self, report_session):
        # Not a crash case — an empty list is a valid (if unusual) plan.
        content = generate_report(report_session, format="markdown", section_order=[])
        assert "sales.csv" in content  # header survives
        assert "## Executive Summary" not in content

    def test_unknown_section_key_is_silently_ignored(self, report_session):
        # A stale/typo'd key shouldn't crash the whole report — just contributes nothing.
        content = generate_report(report_session, format="markdown",
                                    section_order=["executive_summary", "not_a_real_section"])
        assert "## Executive Summary" in content
