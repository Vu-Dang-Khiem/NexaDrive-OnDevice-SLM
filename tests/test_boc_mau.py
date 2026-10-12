"""Test bốc mẫu phân tầng để thẩm định chéo + lấy lại kết quả hàm từ Prompt B."""

import pytest

from eval.boc_mau_tham_dinh import boc_phan_tang
from gen.records import CallRecord, ReplyRecord, SuggestRecord, negative_record, result_from_reply_prompt


def _kho() -> list[dict]:
    rows = []
    for i in range(200):
        rows.append(CallRecord(f"mở cốp {i}", "open_trunk", {}).to_row())
        rows.append(negative_record(f"tán gẫu {i}").to_row())
    for i in range(20):
        tra_loi = ReplyRecord(f"còn xăng {i}", "get_fuel_level", {}, {"fuel_percent": 42}, f"Còn 42%. {i}")
        rows.append(tra_loi.to_row())
        rows.append(SuggestRecord(f"nóng quá {i}", (("set_ac", {"state": "on"}),)).to_row())
    return rows


def test_boc_dung_so_luong():
    assert len(boc_phan_tang(_kho(), n=100, seed=1)) == 100


def test_moi_loai_deu_co_mat():
    """Loại hiếm (ask, reply) chỉ chiếm ~9% kho — bốc ngẫu nhiên toàn cục dễ trượt."""
    mau = boc_phan_tang(_kho(), n=40, seed=1)
    assert {r["meta"]["type"] for r in mau} == {"call", "reply", "ask"}


def test_xin_nhieu_hon_kho_thi_lay_het():
    kho = _kho()
    assert len(boc_phan_tang(kho, n=10_000, seed=1)) == len(kho)


def test_cung_seed_ra_cung_mau():
    a = boc_phan_tang(_kho(), n=50, seed=7)
    b = boc_phan_tang(_kho(), n=50, seed=7)
    assert [r["prompt"] for r in a] == [r["prompt"] for r in b]


# ── lấy lại kết quả hàm từ Prompt B (cần khi chấm lại mẫu reply) ────────────
def test_lay_lai_ket_qua_ham():
    row = ReplyRecord(
        "còn bao nhiêu xăng?", "get_fuel_level", {}, {"fuel_percent": 42, "range_km": 380}, "Xăng còn 42%."
    ).to_row()
    assert result_from_reply_prompt(row["prompt"]) == {"fuel_percent": 42, "range_km": 380}


def test_prompt_khong_co_khoi_tool_thi_bao_loi():
    row = CallRecord("mở cốp xe", "open_trunk", {}).to_row()
    with pytest.raises(ValueError):
        result_from_reply_prompt(row["prompt"])
