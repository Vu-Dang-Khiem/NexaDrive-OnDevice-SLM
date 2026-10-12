"""Luật chấm C + đóng băng tập test.

Luật C (PLAN.md, chốt 2026-09-27): (1) bỏ tham số bằng mặc định ở cả nhãn lẫn output; (2) câu mơ hồ
chỉ chấm đúng hàm + đúng hướng; nhãn nhiều hàm = nhiều lựa chọn chấp nhận được.
"""

import hashlib
from pathlib import Path

import pytest

from eval.metrics import DUNG, SAI_HAM, SAI_THAM_SO, bo_mac_dinh, cham_mot_cau


def _c(out: str, gold: str, luat: str = "C") -> str:
    return cham_mot_cau(out, gold, luat)[0]


# ── (1) tham số bằng mặc định ───────────────────────────────────────────────
@pytest.mark.parametrize(
    "out, gold",
    [
        ('<nexa_4>(unit="km")<nexa_end>', "<nexa_4>()<nexa_end>"),  # tranh cãi thật trong tập test
        ("<nexa_4>()<nexa_end>", '<nexa_4>(unit="km")<nexa_end>'),  # chiều ngược lại
        ('<nexa_10>(door="all")<nexa_end>', "<nexa_10>()<nexa_end>"),
        ("<nexa_23>(day=0)<nexa_end>", "<nexa_23>()<nexa_end>"),
        ('<nexa_15>(window="all", position="down")<nexa_end>',
         '<nexa_15>(window="all", position="down", amount=100)<nexa_end>'),
    ],
)
def test_luat_c_coi_tham_so_bang_mac_dinh_la_tuong_duong(out, gold):
    assert _c(out, gold) == DUNG


def test_luat_cu_van_so_khit():
    """Giữ luật cũ để báo cáo cả hai con số."""
    assert _c('<nexa_4>(unit="km")<nexa_end>', "<nexa_4>()<nexa_end>", "cu") == SAI_THAM_SO


def test_luat_c_khong_cuu_khac_biet_hanh_vi():
    """Nhóm rõ ràng: delta=1 và delta=2 là hai hành vi khác nhau -> vẫn sai."""
    assert _c("<nexa_19>(delta=1)<nexa_end>", "<nexa_19>(delta=2)<nexa_end>") == SAI_THAM_SO
    assert _c('<nexa_4>(unit="mi")<nexa_end>', "<nexa_4>()<nexa_end>") == SAI_THAM_SO


def test_bo_mac_dinh_giu_tham_so_khac_mac_dinh():
    assert bo_mac_dinh("get_odometer", {"unit": "mi"}) == {"unit": "mi"}
    assert bo_mac_dinh("get_odometer", {"unit": "km"}) == {}


# ── (2) nhóm mơ hồ: đúng hàm + đúng hướng ───────────────────────────────────
def test_mo_ho_muc_quat_khac_so_nhung_cung_huong_la_dung():
    """'Quạt kêu to quá': giảm về 2 hay 3 không có đáp án nào đúng hơn."""
    assert _c("<nexa_ask><nexa_7>(level=3)<nexa_end>", "<nexa_ask><nexa_7>(level=2)<nexa_end>") == DUNG


def test_mo_ho_sai_huong_la_sai():
    assert _c("<nexa_ask><nexa_7>(level=6)<nexa_end>", "<nexa_ask><nexa_7>(level=2)<nexa_end>") == SAI_THAM_SO
    assert _c('<nexa_ask><nexa_15>(window="all", position="down")<nexa_end>',
              '<nexa_ask><nexa_15>(window="all", position="up")<nexa_end>') == SAI_THAM_SO


def test_mo_ho_khong_cham_cua_nao():
    """Câu 'Con bé say xe...' không nói bên nào -> chỉ chấm hạ hay nâng."""
    assert _c('<nexa_ask><nexa_15>(window="rear_left", position="down")<nexa_end>',
              '<nexa_ask><nexa_15>(window="all", position="down")<nexa_end>') == DUNG


def test_mo_ho_hai_lua_chon_model_chon_mot_la_dung():
    gold = '<nexa_ask><nexa_21>(destination="bệnh viện");<nexa_22>(category="hospital")<nexa_end>'
    assert _c('<nexa_ask><nexa_22>(category="hospital")<nexa_end>', gold) == DUNG
    assert _c('<nexa_ask><nexa_21>(destination="bệnh viện Bạch Mai")<nexa_end>', gold) == DUNG


def test_mo_ho_ham_ngoai_lua_chon_la_sai_ham():
    gold = '<nexa_ask><nexa_6>(state="on");<nexa_15>(window="all", position="down")<nexa_end>'
    assert _c('<nexa_ask><nexa_16>(action="open")<nexa_end>', gold) == SAI_HAM


def test_mo_ho_he_cua_so_troi_cung_huong_mo():
    assert _c('<nexa_ask><nexa_16>(action="tilt")<nexa_end>',
              '<nexa_ask><nexa_16>(action="open")<nexa_end>') == DUNG
    assert _c('<nexa_ask><nexa_16>(action="close")<nexa_end>',
              '<nexa_ask><nexa_16>(action="open")<nexa_end>') == SAI_THAM_SO


def test_mo_ho_den_pha_xa_van_tach_rieng():
    """Gợi ý pha xa khi trời tối là lỗi an toàn — không được gộp vào 'bật đèn'."""
    gold = '<nexa_ask><nexa_17>(mode="low_beam")<nexa_end>'
    assert _c('<nexa_ask><nexa_17>(mode="auto")<nexa_end>', gold) == DUNG
    assert _c('<nexa_ask><nexa_17>(mode="high_beam")<nexa_end>', gold) == SAI_THAM_SO


def test_mo_ho_nhung_model_lam_luon_van_la_thieu_hoi_lai():
    assert _c('<nexa_7>(level=2)<nexa_end>', "<nexa_ask><nexa_7>(level=2)<nexa_end>") == "thieu_hoi_lai"


# ── đóng băng tập test ──────────────────────────────────────────────────────
# Sửa tập test sau khi đã thấy điểm là tự ra đề rồi tự chấm. File nào đổi dù một ký tự -> đỏ.
# Chỉ được cập nhật mã băm ở đây khi có quyết định ghi trong PLAN.md.
_F = "data/processed/final/"
DONG_BANG = {
    _F + "test_vi.jsonl": "22254e2ad83c8f298828755a429858bad81496d3366848331cc666127846abee",
    _F + "test_vi_sach_v2.jsonl": "58312634432921e65e46a8a86418f3178d80c3d98bfacd9163fc7f2886fc5c2b",
    _F + "test_vi_bo_sung.jsonl": "dfe426aece9fc734f084a8d6beec74cfd006f9a46e4fffe0e264f1aef3f65b34",
    "data/test_tay/test_vi_bo_sung.csv": "d71af6b2c6c7b63db169e7f3ba09adf00c816eb2c409daba6024fa02efb6714f",
    # Tập DEV người viết (PLAN.md mục N) — dùng để thử/chỉnh; đóng băng như test để không chỉnh nhãn theo điểm
    "data/dev_tay_2/viet_cau_dev_2_da_viet.csv": (
        "608a679b1467750e3afe0f1b86da87ce5a2361686aa4aafd125b2b3adb3a2b53"
    ),
    "data/dev_tay_2/dev_vi_2.csv": "9c36bd8a9d15b40a2d13736d0f3738edcfc7d8d1450b1bf2556f38200e65fee6",
    "data/dev_tay_2/dev_vi_2.jsonl": "9a701efdb8ea5e87375f0691e5954237117401f1a6636d19a55a3adc1ac1d265",
    # Bộ mơ hồ thứ 3 (PLAN.md mục T) — đóng băng 2026-10-03, TRƯỚC khi chạy model nào trên nó
    _F + "test_vi_mo_ho_3.jsonl": "e6d3f8a2337a671a778724bce2ca7d5b5d406b9d036b82fe18fa5e8298995caf",
    "data/mo_ho_3/2_gan_nhan_da_duyet.csv": (
        "101c3d162b07b6c57de181c583f6f2f8a454f754bb944aeb6274bee10ba7db86"
    ),
    # Bộ mơ hồ thứ 4 (PLAN.md mục U — đánh giá vòng 5) — đóng băng 2026-10-04, TRƯỚC khi chạy model nào
    _F + "test_vi_mo_ho_4.jsonl": "dd792c70d8a27044342cb9f37200768d1f45a81dd119060b5563e67a7875b4bb",
    "data/mo_ho_4/2_gan_nhan_da_duyet.csv": (
        "6f59765baf7919665da95f2c0427795779d4b0a33d3bdbffb92272a21f93ec18"
    ),
}


@pytest.mark.parametrize("duong_dan, bam", sorted(DONG_BANG.items()))
def test_tap_test_khong_bi_sua(duong_dan, bam):
    p = Path(duong_dan)
    if not p.exists():
        pytest.skip("không có tập test ở máy này")
    assert hashlib.sha256(p.read_bytes()).hexdigest() == bam, f"{duong_dan} đã bị sửa sau khi đóng băng"
