"""Sửa từ khoá xe nghe nhầm (l/n, s/x, ch/tr, d/gi/r) — chỉ cụm ≥ 2 âm tiết lấy từ gen.v6.TU_VUNG."""

import pytest

from serve.sua_tu_khoa import BANG, sua


@pytest.mark.parametrize(
    "vao,ra",
    [
        ("mở lắp sang giúp tôi", "mở nắp xăng giúp tôi"),          # l/n + s/x + dấu
        ("Lắp bình sang mở hộ", "Nắp bình xăng mở hộ"),            # cụm 3 âm tiết, giữ chữ hoa đầu câu
        ("mở cổng sắc đi", "mở cổng sạc đi"),                       # chỉ sai dấu thanh
        ("đóng của sổ trời lại", "đóng cửa sổ trời lại"),
        ("bật đèn khẩn cấp, nhanh lên", "bật đèn khẩn cấp, nhanh lên"),  # viết đúng -> giữ nguyên
    ],
)
def test_sua_cum_nghe_nham(vao, ra):
    assert sua(vao) == ra


@pytest.mark.parametrize(
    "cau",
    [
        "chuyển sang chế độ sưởi",   # "sang" là từ thật — từ đơn không bao giờ bị sửa
        "bật điều hòa 22 độ",        # chỉ khác chỗ đặt dấu (hòa / hoà) -> không phải nghe nhầm
        "bật điều hoà 22 độ",
        "mở cốp",
        "kể chuyện cười đi",
        "",
    ],
)
def test_khong_dung_cau_dung(cau):
    assert sua(cau) == cau


def test_dau_cau_chen_giua_thi_khong_ghep_cum():
    assert sua("mở lắp, sang trái") == "mở lắp, sang trái"


def test_bang_chi_co_cum_nhieu_am_tiet():
    assert BANG and all(len(k) >= 2 for k in BANG)
