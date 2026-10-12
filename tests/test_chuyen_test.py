"""Test khâu chuyển tập test người viết tay (eval/chuyen_test.py)."""

import pytest

from eval.chuyen_test import doc_csv, doc_nhan, dong_thanh_row, thong_ke
from gen.teacher_prompts import TeacherResponseError
from schema.call_format import CallParseError, parse_calls, parse_readable


# ── đọc lời gọi dạng người viết ─────────────────────────────────────────────
def test_doc_loi_goi_khong_tham_so():
    assert parse_readable("open_trunk()") == [("open_trunk", {})]


def test_doc_loi_goi_co_tham_so():
    (ten, args), = parse_readable('set_ac(state="on", temperature=26, zone="driver")')
    assert ten == "set_ac"
    assert args == {"state": "on", "temperature": 26, "zone": "driver"}


def test_doc_nhieu_loi_goi():
    calls = parse_readable('set_ac(state="on"); control_window(window="all", position="down")')
    assert [t for t, _ in calls] == ["set_ac", "control_window"]


def test_ten_ham_sai_bao_loi():
    with pytest.raises(CallParseError, match="không có hàm"):
        parse_readable("bat_dieu_hoa()")


def test_tham_so_la_bao_loi():
    with pytest.raises(CallParseError, match="tham số lạ"):
        parse_readable('set_ac(state="on", mau_sac="do")')


# ── kiểm tham số theo schema ────────────────────────────────────────────────
def test_nhan_hoi_lai_co_tien_to_ask():
    calls, la_hoi_lai = doc_nhan('ask: set_ac(state="on")')
    assert la_hoi_lai is True
    assert calls == [("set_ac", {"state": "on"})]


def test_gia_tri_ngoai_khoang_bi_bat():
    with pytest.raises(TeacherResponseError, match="max"):
        doc_nhan('set_ac(state="on", temperature=99)')


def test_gia_tri_ngoai_enum_bi_bat():
    with pytest.raises(TeacherResponseError, match="enum"):
        doc_nhan('set_headlights(mode="sieu_sang")')


def test_thieu_tham_so_bat_buoc_bi_bat():
    with pytest.raises(TeacherResponseError, match="bắt buộc"):
        doc_nhan("set_ac()")


# ── ghép thành dòng dữ liệu ─────────────────────────────────────────────────
def test_row_cau_ro_rang():
    row = dong_thanh_row("Mở cốp xe", "open_trunk()")
    assert row["completion"] == "<nexa_11>()<nexa_end><|im_end|>"
    assert row["meta"]["type"] == "call"
    assert row["meta"]["nguon"] == "nguoi_viet_tay"


def test_row_cau_mo_ho_co_token_hoi_lai():
    row = dong_thanh_row("Nóng quá", 'ask: set_ac(state="on")')
    (call,) = parse_calls(row["completion"])
    assert call.suggested is True
    assert row["meta"]["type"] == "ask"


def test_row_ngoai_pham_vi():
    row = dong_thanh_row("Kể chuyện cười đi", "irrelevant_function()")
    assert row["meta"]["fn"] == "irrelevant_function"


def test_ngoai_pham_vi_khong_duoc_dung_ask():
    with pytest.raises(CallParseError):
        dong_thanh_row("Kể chuyện cười đi", "ask: irrelevant_function()")


# ── đọc cả file CSV ─────────────────────────────────────────────────────────
def _viet(tmp_path, noi_dung):
    p = tmp_path / "test.csv"
    p.write_text(noi_dung, encoding="utf-8")
    return p


def test_doc_csv_hop_le(tmp_path):
    noi_dung = 'cau,nhan,ghi_chu\nMở cốp xe,open_trunk(),\nNóng quá,"ask: set_ac(state=""on"")",\n'
    p = _viet(tmp_path, noi_dung)
    rows, loi = doc_csv(p)
    assert loi == []
    assert len(rows) == 2


def test_doc_csv_bao_loi_kem_so_dong(tmp_path):
    p = _viet(tmp_path, "cau,nhan,ghi_chu\nMở cốp xe,open_trunk(),\nBật điều hoà,ham_bia(),\n")
    rows, loi = doc_csv(p)
    assert len(rows) == 1
    assert len(loi) == 1
    assert "dòng 3" in loi[0]


def test_doc_csv_bat_cau_trung(tmp_path):
    p = _viet(tmp_path, "cau,nhan,ghi_chu\nMở cốp xe,open_trunk(),\nmở cốp   XE,open_trunk(),\n")
    rows, loi = doc_csv(p)
    assert len(rows) == 1
    assert "trùng" in loi[0]


def test_thong_ke_dem_dung_3_nhom():
    rows = [
        dong_thanh_row("Mở cốp xe", "open_trunk()"),
        dong_thanh_row("Nóng quá", 'ask: set_ac(state="on")'),
        dong_thanh_row("Kể chuyện cười đi", "irrelevant_function()"),
    ]
    tk = thong_ke(rows)
    assert tk["theo_nhom"] == {"ro_rang": 1, "mo_ho": 1, "ngoai_pham_vi": 1}
    assert "set_fan_speed" in tk["ham_chua_co_cau_test"]


def test_file_mau_trong_repo_hop_le():
    """File mẫu data/test_tay/test_vi.csv phải luôn parse được — nó là hướng dẫn cho người viết."""
    rows, loi = doc_csv(__import__("pathlib").Path("data/test_tay/test_vi.csv"))
    assert loi == [], loi
    assert len(rows) >= 10
