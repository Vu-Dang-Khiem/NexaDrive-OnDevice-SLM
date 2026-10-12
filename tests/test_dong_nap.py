"""Đóng nắp xăng / nắp sạc bằng luật runtime (serve/dong_nap.py — "cách A", người dùng chọn 2026-10-12)."""

import pytest

from serve.contracts import AssistantTurn, StateEvent
from serve.dong_nap import nap_can_dong
from serve.mo_hinh import MoHinhGia
from serve.tac_tu import TacTu
from serve.xe_mo_phong import TrangThaiXe, XeMoPhong


@pytest.mark.parametrize(
    "cau,ra",
    [
        ("Đóng nắp xăng", "xang"),
        ("đóng nắp bình xăng lại giúp mình", "xang"),
        ("Khép giúp mình cái nắp xăng", "xang"),
        ("Đóng nắp nhiên liệu", "xang"),
        ("Đóng nắp sạc", "sac"),
        ("Đậy cổng sạc lại", "sac"),
        ("Đóng ổ sạc đi", "sac"),
        # KHÔNG phải lệnh đóng nắp
        ("Mở nắp xăng", None),
        ("Đóng cốp", None),
        ("Xăng còn bao nhiêu", None),
        ("đóng đi", None),  # không có lượt trước mở nắp
    ],
)
def test_nhan_cau_dong_nap(cau, ra):
    assert nap_can_dong(cau) == ra


def test_dong_di_sau_khi_mo_nap():
    assert nap_can_dong("đóng đi", "open_fuel_flap") == "xang"
    assert nap_can_dong("Đóng lại giúp mình nhé", "open_charge_port") == "sac"
    assert nap_can_dong("đóng đi", "open_trunk") is None  # việc của model / bộ nhớ, không phải luật này


class _DemModel(MoHinhGia):
    def __init__(self) -> None:
        super().__init__()
        self.so_lan = 0

    def goi_ham(self, cau, luot_truoc=None):
        self.so_lan += 1
        return super().goi_ham(cau, luot_truoc)


def _luot(sk) -> AssistantTurn:
    return next(s for s in sk if isinstance(s, AssistantTurn))


def test_dong_nap_xang_khong_hoi_model():
    m = _DemModel()
    a = TacTu(m, XeMoPhong(TrangThaiXe(fuel_flap_open=True)))
    sk = a.xu_ly("Đóng nắp xăng")
    assert not a.xe.s.fuel_flap_open and m.so_lan == 0
    luot = _luot(sk)
    assert luot.text == "Đã đóng nắp bình xăng." and luot.calls[0].name == "close_fuel_flap"
    assert any(isinstance(s, StateEvent) for s in sk)


def test_mo_roi_dong_di():
    a = TacTu(MoHinhGia(), XeMoPhong())
    a.xu_ly("Mở nắp sạc")
    assert a.xe.s.charge_port_open
    sk = a.xu_ly("đóng đi")
    assert not a.xe.s.charge_port_open and _luot(sk).calls[0].name == "close_charge_port"


def test_nap_da_dong_san():
    a = TacTu(MoHinhGia(), XeMoPhong())
    assert _luot(a.xu_ly("Đóng nắp xăng")).text == "Nắp bình xăng đang đóng rồi."


def test_dong_nap_khi_xe_dang_chay_van_duoc():
    a = TacTu(MoHinhGia(), XeMoPhong(TrangThaiXe(dang_chay=True, fuel_flap_open=True)))
    a.xu_ly("Đóng nắp xăng")
    assert not a.xe.s.fuel_flap_open  # đóng nắp không có rủi ro -> không chặn, không hỏi lại


def test_cau_nhieu_lenh_co_dong_nap():
    a = TacTu(MoHinhGia(), XeMoPhong(TrangThaiXe(fuel_flap_open=True)))
    a.xu_ly("Đóng nắp xăng xong mở cốp giúp tôi")
    assert not a.xe.s.fuel_flap_open and a.xe.s.trunk_open
