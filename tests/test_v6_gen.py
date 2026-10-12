"""gen/v6.py — prompt + luật code của vòng dữ liệu v6 (không cần GPU)."""

import random
from collections import Counter

import pytest

from gen.teacher_prompts import TeacherResponseError
from gen.v6 import (
    CAP_DOI_XUNG,
    HAM_LUOT_1,
    KIEU_NOI_TIEP,
    TU_VUNG,
    ap_quy_uoc_nhiet,
    bo_dau,
    build_da_dang_prompt,
    build_nhan_noi_tiep_prompt,
    dung_do_dai,
    lap_lo_k1,
    loi_go,
    parse_nhan_noi_tiep,
    them_nhieu,
)
from schema.loader import CATALOG


def test_tu_vung_phu_du_24_ham():
    assert set(TU_VUNG) == {f.name for f in CATALOG.functions} - {"irrelevant_function"}
    assert all(tv["goi"] and tv["dong_tu"] for tv in TU_VUNG.values())
    assert "pha" in TU_VUNG["set_headlights"]["goi"] and "kiếng" in TU_VUNG["control_window"]["goi"]


def test_lap_lo_k1_chia_deu_ham_va_cach_goi_va_ti_le_do_dai():
    lo = lap_lo_k1(so_lo_moi_ham=20, cau_moi_lo=5, seed=1)
    assert Counter(x.ham for x in lo) == dict.fromkeys(TU_VUNG, 20)
    goi_den = {x.goi for x in lo if x.ham == "set_headlights"}
    assert goi_den == set(TU_VUNG["set_headlights"]["goi"])  # mọi cách gọi đều được giao
    dd = Counter(x.do_dai for x in lo)
    assert dd["rat_ngan"] / len(lo) == pytest.approx(0.4, abs=0.02)


def test_prompt_k1_ep_dung_cach_goi_va_do_dai():
    lo = next(x for x in lap_lo_k1(10, 5, 2) if x.ham == "set_headlights" and x.goi == "pha")
    p = build_da_dang_prompt(lo)
    assert 'cụm "pha"' in p and "5 câu" in p


@pytest.mark.parametrize("cau, do_dai, dung", [
    ("bật pha", "rat_ngan", True), ("bật đèn pha lên giúp tôi với nhé", "rat_ngan", False),
    ("mở kiếng bên tài coi", "ngan", True), ("tắt", "vua", False),
])
def test_kiem_do_dai(cau, do_dai, dung):
    assert dung_do_dai(cau, do_dai) is dung


@pytest.mark.parametrize("cau, args, ra", [
    ("cho mát hơn chút", {"state": "on", "delta": -3}, {"state": "on", "delta": -2}),  # không số -> ±2
    ("tăng điều hoà thêm 3 độ", {"state": "on", "delta": 3}, {"state": "on", "delta": 3}),
    ("giảm xuống 20 độ", {"state": "on", "delta": -4}, None),  # "xuống 20 độ" mà ghi delta -> loại
    ("ấm hơn tí", {"state": "on", "temperature": 26}, None),  # không số mà có nhiệt độ cụ thể -> bịa
    ("bật điều hoà 22 độ", {"state": "on", "temperature": 22}, {"state": "on", "temperature": 22}),
    ("x", {"state": "on", "temperature": 22, "delta": 2}, None),
    ("giảm một chút", {"state": "on", "delta": -1}, {"state": "on", "delta": -2}),  # "một chút" không phải số
    ("giảm một độ", {"state": "on", "delta": -1}, {"state": "on", "delta": -1}),
    ("bật điều hòa lạnh nhé", {"state": "on", "delta": -2, "mode": "cool"}, {"state": "on", "mode": "cool"}),
    ("bật điều hoà 22 độ", {"state": "on", "temperature": 22, "mode": "heat"},
     {"state": "on", "temperature": 22}),
])
def test_quy_uoc_nhiet(cau, args, ra):
    assert ap_quy_uoc_nhiet(cau, args) == ra


def test_vá_state_va_quy_uoc_den():
    from gen.v6 import ap_quy_uoc, tham_so_v6

    assert tham_so_v6({"delta": -2}, "set_ac") == {"delta": -2, "state": "on"}  # "Giảm nhiệt độ đi mà"
    with pytest.raises(TeacherResponseError):
        tham_so_v6({}, "set_ac")  # không có gì để suy ra -> vẫn thiếu state
    assert ap_quy_uoc("set_headlights", "bật đèn pha lên", {"mode": "low_beam"}) == {"mode": "high_beam"}
    assert ap_quy_uoc("set_headlights", "chuyển sang cốt", {"mode": "high_beam"}) == {"mode": "low_beam"}
    assert ap_quy_uoc("set_headlights", "tắt đèn pha", {"mode": "off"}) == {"mode": "off"}


@pytest.mark.parametrize("cau, args, ra", [
    ("Giảm quạt gió xuống", {"level": 2}, {"delta": -1}),  # không số -> ±1 (người dùng chốt)
    ("Giảm quạt gió xuống một nấc cho đỡ ồn", {"level": 1}, {"delta": -1}),
    ("Cho quạt gió mạnh thêm hai nấc đi", {"level": 4}, {"delta": 2}),
    ("Quạt mạnh lên chút", {"level": 6}, {"delta": 1}),
    ("Tăng quạt gió lên mức 5", {"level": 5}, {"level": 5}),  # mức tuyệt đối giữ nguyên
    ("Bật quạt gió mạnh nhất", {"level": 7}, {"level": 7}),
    ("quạt số 3", {"level": 3}, {"level": 3}),
    ("tắt quạt", {"level": 0}, {"level": 0}),
    ("giảm xuống nấc một", {"level": 1}, {"level": 1}),
    ("quạt lên 3 đi", {"level": 3}, {"level": 3}),  # "lên N" = mức tuyệt đối
    ("mạnh lên chút", {"delta": 2}, {"delta": 1}),  # teacher ghi delta không số -> ±1
])
def test_quy_uoc_quat(cau, args, ra):
    from gen.v6 import ap_quy_uoc

    assert ap_quy_uoc("set_fan_speed", cau, args) == ra


@pytest.mark.parametrize("cau, args, ra", [
    ("to thêm chút nghen", {"delta": 20}, {"delta": 3}),
    ("lên chút đi", {"delta": 5}, {"delta": 3}),
    ("nhỏ hơn chút nữa", {"delta": -20}, {"delta": -3}),
    ("tăng thêm 4 nấc", {"delta": 4}, {"delta": 4}),
    ("đặt mức 15", {"set_value": 15}, {"set_value": 15}),
])
def test_quy_uoc_am_luong(cau, args, ra):
    from gen.v6 import ap_quy_uoc

    assert ap_quy_uoc("set_volume", cau, args) == ra


def test_bo_dau_va_loi_go():
    assert bo_dau("Mở cốp giùm, đóng điều hoà") == "Mo cop gium, dong dieu hoa"
    rng = random.Random(0)
    for _ in range(50):
        moi = loi_go("hạ kính ghế phụ xuống", rng)
        assert moi != "hạ kính ghế phụ xuống" and len(moi.split()) == 5
    kieu = Counter(them_nhieu("mở cốp sau giúp", random.Random(i))[1] for i in range(400))
    assert set(kieu) == {"bo_dau", "loi_go", "bo_dau_loi_go"} and kieu["bo_dau"] > kieu["loi_go"]


def test_nhan_noi_tiep():
    p = build_nhan_noi_tiep_prompt("mở cốp", "open_trunk()", "đóng đi")
    assert "đóng đi" in p and "close_trunk" in p
    assert parse_nhan_noi_tiep('{"ham": "close_trunk", "tham_so": {}}') == ("close_trunk", {})
    with pytest.raises(TeacherResponseError):
        parse_nhan_noi_tiep('{"ham": "dong_cop"}')


def test_kieu_noi_tiep_va_doi_xung_hop_le():
    assert all(set(k) <= set(KIEU_NOI_TIEP) for k in HAM_LUOT_1.values())
    assert all(h in CATALOG.by_name for h in HAM_LUOT_1)
    assert all(h in CATALOG.by_name for h, _, _ in CAP_DOI_XUNG)
