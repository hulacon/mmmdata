"""Tests for behavioral.constants."""

from behavioral.constants import (
    SUBJECT_IDS, TB_SESSIONS, TB_ENCODING_SESSIONS, SESSION_ORDER,
    EnCon, ReCon, ENCON_LABELS, RECON_LABELS,
    RESP_POSITION_MAP, RESP_CONFIDENCE_MAP,
)


def test_subject_ids():
    assert SUBJECT_IDS == ("03", "04", "05")


def test_tb_sessions_range():
    assert TB_SESSIONS[0] == "04"
    assert TB_SESSIONS[-1] == "18"
    assert len(TB_SESSIONS) == 15


def test_encoding_sessions_exclude_18():
    assert "18" not in TB_ENCODING_SESSIONS
    assert TB_ENCODING_SESSIONS[-1] == "17"
    assert len(TB_ENCODING_SESSIONS) == 14


def test_session_order_mapping():
    assert SESSION_ORDER["04"] == 0
    assert SESSION_ORDER["18"] == 14
    assert len(SESSION_ORDER) == 15


def test_encon_enum():
    assert EnCon.SINGLE == 1
    assert EnCon.REPEATS == 2
    assert EnCon.TRIPLETS == 3
    assert ENCON_LABELS[1] == "single"


def test_recon_enum():
    assert ReCon.WITHIN == 1
    assert ReCon.ACROSS == 2
    assert RECON_LABELS[1] == "within"


def test_resp_position_map():
    assert RESP_POSITION_MAP[1] == 1
    assert RESP_POSITION_MAP[2] == 1
    assert RESP_POSITION_MAP[3] == 2
    assert RESP_POSITION_MAP[4] == 2


def test_resp_confidence_map():
    assert RESP_CONFIDENCE_MAP[1] == "sure"
    assert RESP_CONFIDENCE_MAP[2] == "maybe"
    assert RESP_CONFIDENCE_MAP[3] == "maybe"
    assert RESP_CONFIDENCE_MAP[4] == "sure"
