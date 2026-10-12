"""Luật 'hàm CHỈ ĐỌC thì không hỏi lại'.

Hỏi lại là cơ chế chống làm sai. Hàm chỉ đọc số liệu (xem xăng, pin, áp suất lốp) thì
không có gì để làm sai — hỏi thêm chỉ làm phiền người đang lái. "Hình như xe non hơi
một bánh" phải kiểm tra rồi báo luôn, không phải "bạn có muốn tôi kiểm tra không?".
"""

import json

import pytest

from eval.chuyen_test import dong_thanh_row
from gen.don_ask import don_row
from gen.records import SuggestRecord
from gen.teacher_prompts import parse_ask_label_response
from schema.call_format import CallParseError, parse_calls
from schema.loader import CATALOG

CHI_DOC = [
    "get_fuel_level",
    "get_battery_range",
    "get_tire_pressure",
    "get_vehicle_status",
    "get_odometer",
    "get_range_estimate",
    "get_weather",
]


def test_schema_danh_dau_dung_cac_ham_chi_doc():
    assert sorted(f.name for f in CATALOG.functions if f.chi_doc) == sorted(CHI_DOC)


def test_ham_hanh_dong_khong_bi_danh_dau_chi_doc():
    for ten in ("set_ac", "open_trunk", "control_window", "navigate_to", "find_nearby"):
        assert CATALOG.by_name[ten].chi_doc is False


# ── pipeline gán nhãn: ngầm hiểu + chỉ đọc -> làm luôn ──────────────────────
def test_cau_ngam_hieu_ve_ham_chi_doc_thanh_lenh_thuong():
    gia_lap = json.dumps(
        {"loai": "ngam_hieu", "de_xuat": [{"ten_ham": "get_tire_pressure", "tham_so": {}}]}
    )
    lab = parse_ask_label_response(gia_lap, "Hình như xe bị non hơi một bánh")
    assert lab.loai == "ro_rang"  # KHÔNG còn là ngam_hieu
    assert lab.calls == [("get_tire_pressure", {})]


def test_cau_ngam_hieu_ve_ham_hanh_dong_van_giu_hoi_lai():
    de_xuat = [{"ten_ham": "set_ac", "tham_so": {"state": "on"}}]
    gia_lap = json.dumps({"loai": "ngam_hieu", "de_xuat": de_xuat})
    assert parse_ask_label_response(gia_lap, "Nóng quá").loai == "ngam_hieu"


def test_de_xuat_lan_lon_chi_doc_va_hanh_dong_thi_van_hoi_lai():
    """Chỉ chuyển khi TOÀN BỘ đề xuất đều là hàm chỉ đọc."""
    de_xuat = [
        {"ten_ham": "get_fuel_level", "tham_so": {}},
        {"ten_ham": "find_nearby", "tham_so": {"category": "gas_station"}},
    ]
    lab = parse_ask_label_response(json.dumps({"loai": "ngam_hieu", "de_xuat": de_xuat}), "Sắp cạn xăng rồi")
    assert lab.loai == "ngam_hieu"


# ── dọn data cũ: mẫu ask của hàm chỉ đọc -> mẫu call ────────────────────────
def test_don_mau_ask_cu_cua_ham_chi_doc():
    row = SuggestRecord("Không biết pin còn đủ chạy về không", (("get_battery_range", {}),)).to_row()
    moi, doi = don_row(row)
    assert doi is True
    assert moi["meta"]["type"] == "call"
    assert parse_calls(moi["completion"])[0].suggested is False


def test_don_giu_nguyen_mau_ask_cua_ham_hanh_dong():
    row = SuggestRecord("Nóng quá", (("set_ac", {"state": "on"}),)).to_row()
    moi, _ = don_row(row)
    assert moi["meta"]["type"] == "ask"


# ── tập test: chặn người viết gán nhầm ──────────────────────────────────────
def test_tap_test_khong_cho_dung_ask_voi_ham_chi_doc():
    with pytest.raises(CallParseError, match="chỉ đọc"):
        dong_thanh_row("Hình như xe bị non hơi", "ask: get_tire_pressure()")


def test_tap_test_van_cho_ask_voi_ham_hanh_dong():
    row = dong_thanh_row("Nóng quá", 'ask: set_ac(state="on")')
    assert row["meta"]["type"] == "ask"
