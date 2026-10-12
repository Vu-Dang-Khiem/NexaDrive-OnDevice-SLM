"""Test phần từ đồng nghĩa — chống tái diễn lỗi đã làm control_window chỉ đúng 8,3%.

Lỗi gốc: teacher sinh 267/267 câu cho `control_window` dùng "cửa sổ", 0 câu dùng
"kính", trong khi `set_defrost` chiếm 100% chữ "kính". Model học được luật sạch không
một phản ví dụ nào: "kính" ⟹ sấy kính. Đến khi người thật nói "hạ kính bên tài" thì
nó gọi nhầm hàm.

Mấy test này canh ba thứ:
  1. Từ đồng nghĩa có thật sự đi vào prompt gửi teacher không (khai trong YAML mà
     prompt không nhắc thì vô nghĩa);
  2. Các hàm rủi ro cao có đủ từ đang thiếu không;
  3. Hai hàm đèn có danh sách TÁCH BIỆT không — chúng tranh nhau chữ "đèn".
"""

import pytest

from gen.teacher_prompts import build_query_gen_prompt
from schema.loader import CATALOG

# 15 hàm bị khoá từ vựng, phát hiện bằng gen/ra_soat_tu_vung.py trên dữ liệu vòng 1.
HAM_PHAI_CO = [
    "get_fuel_level", "set_ac", "set_fan_speed", "set_defrost", "lock_doors",
    "unlock_doors", "open_trunk", "close_trunk", "open_charge_port", "open_fuel_flap",
    "control_window", "operate_sunroof", "set_headlights", "set_hazard_lights",
    "set_volume",
]


def _fn(ten: str):
    return CATALOG.by_name[ten]


# ── khai báo trong schema ───────────────────────────────────────────────────
@pytest.mark.parametrize("ten", HAM_PHAI_CO)
def test_ham_bi_khoa_deu_co_tu_dong_nghia(ten):
    assert len(_fn(ten).tu_dong_nghia) >= 3, f"{ten} cần ít nhất 3 cách gọi"


@pytest.mark.parametrize("ten", HAM_PHAI_CO)
def test_khong_co_tu_trung_nhau_trong_cung_mot_ham(ten):
    ds = _fn(ten).tu_dong_nghia
    assert len(set(ds)) == len(ds)


def test_control_window_phai_co_chu_kinh():
    """Đây là từ đã gây ra lỗi — thiếu nó là tái diễn y nguyên."""
    assert any("kính" in t for t in _fn("control_window").tu_dong_nghia)


def test_control_window_van_giu_chu_cua_so():
    """Thêm từ mới không được làm mất từ cũ — câu "hạ cửa sổ" vẫn phải hiểu được."""
    assert "cửa sổ" in _fn("control_window").tu_dong_nghia


def test_set_defrost_phan_biet_bang_dong_tu():
    """set_defrost giữ chữ "kính" nhưng phải gắn động từ sấy/sưởi để tách khỏi hạ kính."""
    ds = _fn("set_defrost").tu_dong_nghia
    assert all(t != "kính" for t in ds), "để 'kính' trơ trọi thì lại tranh với control_window"
    assert any("sấy" in t or "sưởi" in t for t in ds)


def test_hai_ham_den_khong_dung_chung_tu_nao():
    """set_headlights và set_hazard_lights đều phủ 100% chữ "đèn" -> phải tách hẳn."""
    pha = set(_fn("set_headlights").tu_dong_nghia)
    khan_cap = set(_fn("set_hazard_lights").tu_dong_nghia)
    assert not (pha & khan_cap)


def test_cua_so_troi_khong_lan_sang_cua_so():
    """"cửa sổ trời" chứa sẵn chuỗi "cửa sổ" -> sunroof cần cách gọi KHÔNG dính chữ đó."""
    ds = _fn("operate_sunroof").tu_dong_nghia
    assert any("cửa sổ" not in t for t in ds), "cần ít nhất một cách gọi như 'nóc xe'"


# ── có đi vào prompt không ──────────────────────────────────────────────────
@pytest.mark.parametrize("ten", HAM_PHAI_CO)
def test_moi_tu_dong_nghia_deu_co_mat_trong_prompt(ten):
    """Khai trong YAML mà prompt không nhắc thì teacher không biết -> vô nghĩa."""
    p = build_query_gen_prompt(_fn(ten), 40)
    for t in _fn(ten).tu_dong_nghia:
        assert t in p, f"{ten}: prompt thiếu cách gọi {t!r}"


def test_prompt_yeu_cau_chia_deu_chu_khong_chi_liet_ke():
    """Chỉ liệt kê thì teacher vẫn dồn vào một từ — phải nói rõ số câu mỗi cách."""
    p = build_query_gen_prompt(_fn("control_window"), 40)
    assert "DÙNG ĐỀU" in p
    assert "8 câu" in p  # 40 câu / 5 cách gọi


def test_ham_khong_khai_tu_dong_nghia_thi_prompt_giu_nguyen():
    fn = _fn("navigate_to")
    assert not fn.tu_dong_nghia
    assert "DÙNG ĐỀU" not in build_query_gen_prompt(fn, 40)


def test_prompt_van_giu_du_phan_cu():
    p = build_query_gen_prompt(_fn("control_window"), 40)
    assert "Chỉ trả về đúng JSON" in p
    assert "queries" in p
    assert "def control_window" in p  # teacher_doc vẫn còn
