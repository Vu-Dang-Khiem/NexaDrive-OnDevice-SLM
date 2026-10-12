"""gen/dung_vong_6.py — sửa nhãn cũ theo schema v6, ghép lượt trước thừa, nhiễu, đổi khối K1–K4 thành mẫu
train."""

import pytest

from gen.dung_vong_6 import _goi, gan_luot_truoc_thua, mau_tu_khoi, sua_nhan_v6, them_mau_nhieu
from gen.records import IM_END
from schema.call_format import render_completion
from schema.loader import CATALOG


def _row(cau, goi, hoi=False, loai=None):
    return {"prompt": CATALOG.render_call_prompt(cau), "completion": render_completion(goi, ask=hoi) + IM_END,
            "meta": {"lang": "vi", "type": loai or ("ask" if hoi else "call"), "fn": goi[0][0]}}


@pytest.mark.parametrize("cau, goi, hoi, ra, ly_do", [
    ("Trong xe lạnh run cả người", [("set_ac", {"state": "on"})], True,
     [("set_ac", {"state": "on", "mode": "heat"})], "them_mode_heat"),
    ("Điều hoà lạnh buốt", [("set_ac", {"state": "off"})], True, [("set_ac", {"state": "off"})], ""),
    ("Bật sưởi lên giúp tôi", [("set_ac", {"state": "on"})], False,
     [("set_ac", {"state": "on", "mode": "heat"})], "them_mode_heat"),
    ("Giảm nhiệt độ máy lạnh", [("set_ac", {"state": "on", "temperature": 18})], False,
     [("set_ac", {"state": "on", "delta": -2})], "nhiet_tuong_doi_thanh_delta"),
    ("Hạ hai kính sau xuống", [("control_window", {"window": "rear_left", "position": "down"}),
                               ("control_window", {"window": "rear_right", "position": "down"})], False,
     [("control_window", {"window": "rear", "position": "down"})], "gop_kinh_rear"),
    ("Hạ kính sau trái, nâng kính sau phải",
     [("control_window", {"window": "rear_left", "position": "down"}),
      ("control_window", {"window": "rear_right", "position": "up"})], False,
     [("control_window", {"window": "rear_left", "position": "down"}),
      ("control_window", {"window": "rear_right", "position": "up"})], ""),  # khác hướng -> giữ hai lời gọi
])
def test_sua_nhan_v6(cau, goi, hoi, ra, ly_do):
    moi, ld = sua_nhan_v6(_row(cau, goi, hoi))
    assert ld == ly_do and _goi(moi) == ra


def test_tu_choi_bat_suoi_bi_loai_con_suoi_ghe_giu():
    assert sua_nhan_v6(_row("Bật sưởi lên", [("irrelevant_function", {})]))[0] is None
    r = _row("Bật sưởi ghế lái", [("irrelevant_function", {})])
    assert sua_nhan_v6(r) == (r, "")


def test_luot_truoc_thua_la_ham_khac_va_nhan_giu_nguyen():
    rows = [_row("Mở cốp sau giúp tôi với", [("open_trunk", {})])] * 5
    kho = [{"fn": "close_trunk", "args": {}}, {"fn": "set_ac", "args": {"state": "on"}}]
    ra = gan_luot_truoc_thua(rows, kho, 3)
    assert len(ra) == 3
    for r in ra:
        assert "Lượt trước: <nexa_6>(state=\"on\")\nMở cốp sau" in r["prompt"]  # không lấy close_trunk (cặp)
        assert r["completion"] == rows[0]["completion"]


def test_nhieu_bo_qua_tham_so_chu_tu_do():
    rows = [_row("Dẫn đường tới Hồ Gươm", [("navigate_to", {"destination": "Hồ Gươm"})]),
            _row("Mở cốp sau giúp tôi", [("open_trunk", {})])]
    ra = them_mau_nhieu(rows * 20, ti_le=1.0)
    assert ra and all(r["meta"]["fn"] == "open_trunk" for r in ra)


def test_nhieu_mau_hai_luot_giu_nguyen_dong_luot_truoc():
    r = {"prompt": CATALOG.render_call_prompt("đóng cốp lại giúp tôi", "<nexa_11>()"),
         "completion": "<nexa_12>()<nexa_end>" + IM_END, "meta": {"type": "call", "fn": "close_trunk"}}
    ra = them_mau_nhieu([r] * 30, ti_le=1.0)
    assert ra
    assert all("Lượt trước: <nexa_11>()\n" in x["prompt"] and "Luot truoc" not in x["prompt"] for x in ra)


def test_mau_tu_khoi_k4_co_luot_truoc():
    ds = [{"khoi": "K4", "giu": True, "query": "đóng đi", "fn": "close_trunk", "args": {},
           "goi_1": "<nexa_11>()", "kieu": "doi_chieu", "fn_1": "open_trunk"},
          {"khoi": "K1", "giu": True, "query": "bật pha", "fn": "set_headlights",
           "args": {"mode": "high_beam"}},
          {"khoi": "K1", "giu": False, "query": "x", "fn": "set_ac"}]
    ra = mau_tu_khoi(ds)
    assert len(ra) == 2
    assert "Lượt trước: <nexa_11>()\nđóng đi" in ra[0]["prompt"]
    assert ra[0]["completion"].startswith("<nexa_12>()")
    assert "Lượt trước" not in ra[1]["prompt"]
