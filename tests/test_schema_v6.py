"""Schema v6 (PLAN.md Phase 7, người dùng duyệt 2026-10-06): `set_ac.mode`, `set_ac.delta`,
kính `front` / `rear`."""

import pytest

from schema.call_format import parse_calls, render_completion
from serve.an_toan import xet
from serve.contracts import AssistantTurn, ClarifyTurn
from serve.kiem_tra import kiem_tra
from serve.mo_hinh import MoHinhGia
from serve.tac_tu import TacTu
from serve.tra_loi import hanh_dong


def _tt(bang: dict[str, tuple[list, bool]]) -> TacTu:
    return TacTu(MoHinhGia(bang={c: render_completion(lg, ask=hoi) for c, (lg, hoi) in bang.items()}))


@pytest.mark.parametrize("cua, bi_doi", [
    ("rear", {"rear_left", "rear_right"}),
    ("front", {"front_left", "front_right"}),
])
def test_hai_kinh_truoc_sau(cua, bi_doi):
    tt = _tt({"hạ hai kính": ([("control_window", {"window": cua, "position": "down"})], False)})
    tl = tt.xu_ly("hạ hai kính")[0]
    assert {k for k, p in tt.xe.s.windows.items() if p == 100} == bi_doi
    assert isinstance(tl, AssistantTurn) and tl.calls[0].arguments["window"] == cua


def test_hai_kinh_sau_van_qua_luat_an_toan():
    s = TacTu(MoHinhGia()).xe.s
    s.khoa_tre_em = True
    ha_sau = {"window": "rear", "position": "down"}
    assert xet("control_window", ha_sau, s, nguon="wake_word").xu_ly == "hoi_lai"


def test_che_do_suoi_noi_la_bat_suoi():
    tt = _tt({"bật sưởi 27 độ": ([("set_ac", {"state": "on", "temperature": 27, "mode": "heat"})], False)})
    tl = tt.xu_ly("bật sưởi 27 độ")[0]
    assert isinstance(tl, AssistantTurn) and tl.text == "Đã bật sưởi 27 độ cho ấm."
    assert tt.xe.s.ac_mode == "heat" and tt.xe.s.ac_temperature == 27


def test_che_do_lam_lanh_khong_noi_la_suoi_du_cau_co_chu_lanh():
    cau = "lạnh quá, bật làm lạnh 20 độ"  # có chữ "lạnh" nhưng model ghi rõ cool -> không đoán "sưởi"
    tt = _tt({cau: ([("set_ac", {"state": "on", "temperature": 20, "mode": "cool"})], False)})
    assert tt.xu_ly(cau)[0].text == "Đã bật làm lạnh 20 độ."


def test_hoi_lai_than_lanh_v6_co_mode_heat():
    tt = _tt({"Rét run cả người": ([("set_ac", {"state": "on", "mode": "heat"})], True)})
    hoi = tt.xu_ly("Rét run cả người")[0]
    assert isinstance(hoi, ClarifyTurn) and "bật sưởi" in hoi.text


def test_che_do_sai_gia_tri_bi_validator_chan():
    assert not kiem_tra("set_ac", {"state": "on", "mode": "fire"}).hop_le
    assert kiem_tra("set_ac", {"state": "on", "mode": "auto"}).hop_le
    assert hanh_dong("set_ac", {"state": "on", "mode": "auto"}) == "chuyển điều hoà sang tự động"


def test_tham_so_vi_tri_cu_van_giu_nghia():
    # `delta`, `mode` đặt CUỐI danh sách -> lời gọi theo vị trí kiểu cũ không đổi nghĩa
    (c,) = parse_calls('<nexa_6>("on", 24, "rear")<nexa_end>')
    assert dict(c.args) == {"state": "on", "temperature": 24, "zone": "rear"}
