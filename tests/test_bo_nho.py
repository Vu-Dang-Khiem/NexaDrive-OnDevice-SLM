"""Bộ nhớ lượt trước bằng luật (serve/bo_nho.py) — kịch bản người dùng gặp khi test demo 2026-10-06."""

import pytest

from schema.call_format import render_completion
from serve.contracts import AssistantTurn
from serve.mo_hinh import MoHinhGia
from serve.tac_tu import TacTu


def _tt(bang: dict[str, list] | None = None) -> TacTu:
    return TacTu(MoHinhGia(bang={c: render_completion(lg) for c, lg in (bang or {}).items()}))


def _goi(tt: TacTu, cau: str) -> AssistantTurn:
    tl = tt.xu_ly(cau)[0]
    assert isinstance(tl, AssistantTurn)
    return tl


def test_mo_cop_roi_dong_di():
    tt = _tt()
    _goi(tt, "mở cốp")
    assert tt.xe.s.trunk_open
    tl = _goi(tt, "đóng đi")
    assert not tt.xe.s.trunk_open and tl.calls[0].name == "close_trunk"


@pytest.mark.parametrize("cau", ["đóng lại", "Đóng lại đi nhé!", "đóng"])
def test_dong_kinh_vua_ha(cau):
    tt = _tt({"hạ kính ghế phụ": [("control_window", {"window": "front_right", "position": "down"})]})
    _goi(tt, "hạ kính ghế phụ")
    assert tt.xe.s.windows["front_right"] == 100
    _goi(tt, cau)
    assert tt.xe.s.windows["front_right"] == 0


def test_cau_hoi_xen_giua_khong_xoa_bo_nho():
    tt = _tt()
    _goi(tt, "mở cốp")
    _goi(tt, "còn bao nhiêu xăng")  # hàm chỉ đọc -> không ghi đè hành động vừa làm
    _goi(tt, "đóng đi")
    assert not tt.xe.s.trunk_open


def test_giam_nhiet_roi_nua():
    tt = _tt({"giảm nhiệt độ điều hòa": [("set_ac", {"state": "on"})]})
    tt.xe.s.ac_temperature = 24
    _goi(tt, "giảm nhiệt độ điều hòa")  # luật chỉnh nhiệt tương đối -> 22
    tl = _goi(tt, "nữa")
    assert tt.xe.s.ac_temperature == 20 and tl.text == "Đã giảm nhiệt độ điều hoà xuống 20 độ."
    _goi(tt, "thêm chút nữa")
    assert tt.xe.s.ac_temperature == 18
    _goi(tt, "tăng")
    assert tt.xe.s.ac_temperature == 20


def test_tat_dieu_hoa_vua_bat():
    tt = _tt()
    _goi(tt, "bật điều hoà 22 độ")
    _goi(tt, "tắt đi")
    assert not tt.xe.s.ac_on


def test_am_luong_to_nua():
    tt = _tt()
    tt.xe.s.volume = 10
    _goi(tt, "tăng âm lượng")  # model giả: delta +2 -> quy ước "không nói số -> ±3" (serve/quy_uoc.py): 13
    _goi(tt, "to nữa")  # luật bộ nhớ: +3 -> 16
    assert tt.xe.s.volume == 16


@pytest.mark.parametrize("cau", ["đóng cốp", "mở nhạc", "đóng cửa sổ trời lại"])
def test_cau_co_noi_dung_thi_van_de_model(cau):
    tt = _tt()
    _goi(tt, "hạ kính ghế phụ")
    gia = tt.mo_hinh
    goi = []
    goc = gia.goi_ham
    gia.goi_ham = lambda c, truoc=None: (goi.append(c), goc(c))[1]  # type: ignore[method-assign]
    tt.xu_ly(cau)
    assert goi == [cau[:1].upper() + cau[1:]]  # model nhận câu đã viết hoa chữ đầu (serve/chuan_hoa.py)


def test_khong_co_luot_truoc_thi_de_model():
    tt = _tt()
    assert tt.xu_ly("đóng đi")[0].loai == "tu_choi"


def test_dong_sau_mo_khoa_khong_suy_ra_khoa_cua():
    # "đóng" sau "mở khoá" là đóng cánh cửa, không phải khoá -> không suy (để model)
    tt = _tt()
    _goi(tt, "mở khoá cửa")
    tt.xu_ly("đóng đi")
    assert tt.xe.s.doors_locked is False


def test_model_v6_nho_luot_truoc_thi_khong_dung_luat_ma_dua_vao_prompt():
    tt = _tt()
    tt.mo_hinh.nho_luot_truoc = True
    nhan: list = []
    goc = tt.mo_hinh.goi_ham
    tt.mo_hinh.goi_ham = lambda c, truoc=None: (nhan.append((c, truoc)), goc(c))[1]  # type: ignore[method-assign]
    _goi(tt, "mở cốp")
    tt.xu_ly("đóng đi")
    # hoa chữ đầu (serve/chuan_hoa.py); model giả không đọc lượt trước nên TỪ CHỐI "đóng đi" -> router hỏi lại
    # không kèm lượt trước (ngữ cảnh chỉ để giúp, không để làm model từ chối thêm)
    assert nhan == [("Mở cốp", None), ("Đóng đi", "<nexa_11>()"), ("Đóng đi", None)]


def test_prompt_co_luot_truoc_va_khong_co_thi_y_het_v5():
    from schema.loader import CATALOG

    assert CATALOG.render_call_prompt("mở cốp") == CATALOG.render_call_prompt("mở cốp", None)
    p = CATALOG.render_call_prompt("đóng đi", "<nexa_11>()")
    assert "Lượt trước: <nexa_11>()\nđóng đi<|im_end|>" in p


def test_mo_cop_suy_ra_van_qua_luat_an_toan():
    tt = _tt()
    _goi(tt, "mở cốp")
    _goi(tt, "đóng đi")
    tt.xe.s.dang_chay = True
    tl = _goi(tt, "mở ra")
    assert not tt.xe.s.trunk_open and "đang chạy" in tl.text
