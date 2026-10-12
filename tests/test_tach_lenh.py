"""Câu NHIỀU LỆNH (serve/tach_lenh.py + TacTu._nhieu_lenh) — người dùng chọn hướng A 2026-10-08."""

import pytest

from serve.contracts import AssistantTurn, ClarifyTurn, StateEvent
from serve.mo_hinh import MoHinhGia
from serve.tac_tu import TacTu
from serve.tach_lenh import tach_lenh
from serve.xe_mo_phong import TrangThaiXe, XeMoPhong


@pytest.mark.parametrize(
    "cau,ra",
    [
        (
            "Bật cái đèn pha lên xong kiểm tra tình trạng xăng xe thế nào xong đó mở cốp lên",
            ["Bật cái đèn pha lên", "kiểm tra tình trạng xăng xe thế nào", "mở cốp lên"],
        ),
        ("Bật điều hoà 22 độ sau đó mở cốp giúp tôi", ["Bật điều hoà 22 độ", "mở cốp giúp tôi"]),
        ("Mở cốp giúp tôi, rồi bật điều hoà 22 độ", ["Mở cốp giúp tôi", "bật điều hoà 22 độ"]),
        ("Hạ kính ghế phụ với cả tắt nhạc đi", ["Hạ kính ghế phụ", "tắt nhạc đi"]),
        # KHÔNG tách
        ("bật điều hoà và quạt", ["bật điều hoà và quạt"]),  # "và" là một ý
        ("hết xăng rồi", ["hết xăng rồi"]),  # "rồi" đứng một mình
        ("Sạc xong chưa", ["Sạc xong chưa"]),  # đoạn quá ngắn
        ("mở cốp", ["mở cốp"]),
    ],
)
def test_tach_lenh(cau, ra):
    assert tach_lenh(cau) == ra


def _tt(**co_xe) -> TacTu:
    return TacTu(MoHinhGia(), XeMoPhong(TrangThaiXe(**co_xe)))


def _luot(sk) -> AssistantTurn:
    return next(s for s in sk if isinstance(s, AssistantTurn))


def test_lam_lan_luot_dung_thu_tu_va_gop_mot_luot():
    a = _tt()
    sk = a.xu_ly("Bật điều hoà 22 độ xong mở cốp giúp tôi")
    assert a.xe.s.ac_on and a.xe.s.ac_temperature == 22 and a.xe.s.trunk_open
    luot = _luot(sk)
    assert [c.name for c in luot.calls] == ["set_ac", "open_trunk"] and luot.loai == "thuc_thi"
    assert sum(isinstance(s, StateEvent) for s in sk) == 1  # chỉ trạng thái SAU CÙNG


def test_thu_tu_theo_loi_nguoi_lai():
    sk = _tt().xu_ly("Mở cốp giúp tôi xong xăng còn bao nhiêu")
    assert [c.name for c in _luot(sk).calls] == ["open_trunk", "get_fuel_level"]
    assert "62%" in _luot(sk).text


def test_gap_doan_can_hoi_lai_thi_dung_viec_sau_chua_lam():
    a = _tt()
    sk = a.xu_ly("Nóng quá xong mở cốp giúp tôi")
    hoi = next(s for s in sk if isinstance(s, ClarifyTurn))
    assert not a.xe.s.trunk_open  # việc phía sau CHƯA làm
    assert "Mình chưa làm tiếp" in hoi.text and "«mở cốp giúp tôi»" in hoi.text
    assert a.dang_cho is not None  # "có" sẽ trả lời câu hỏi lại của đoạn đầu


def test_luat_an_toan_van_ap_cho_tung_doan():
    a = _tt(dang_chay=True)
    sk = a.xu_ly("Bật điều hoà 22 độ xong mở cốp giúp tôi")
    assert a.xe.s.ac_on and not a.xe.s.trunk_open
    luot = _luot(sk)
    assert [c.name for c in luot.calls] == ["set_ac"] and "đang chạy" in luot.text


def test_dang_cho_xac_nhan_thi_khong_tach():
    a = _tt()
    a.xu_ly("Nóng quá")
    assert a.dang_cho is not None
    a.xu_ly("có")  # trả lời bình thường, không bị coi là nhiều lệnh
    assert a.dang_cho is None
