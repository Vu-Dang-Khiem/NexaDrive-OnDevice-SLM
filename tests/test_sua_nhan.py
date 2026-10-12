"""Luật sửa nhãn vòng 3 — mỗi lỗi hệ thống người kiểm tìm ra có test ở cả hai chiều
(câu phải sửa VÀ câu gần giống nhưng phải giữ nguyên)."""

import pytest

from gen.records import CallRecord, ReplyRecord, negative_record
from gen.sua_nhan import BO, GIU, SUA, sua_loi_goi, sua_rows


# ── xăng / tầm xa ──────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "cau,km",
    [("Xăng còn đủ để đi 950km nữa không?", 950), ("Xăng còn đủ cho 200 km nữa không?", 200),
     ("Còn xăng chạy thêm 350 cây số không?", 350), ("Xăng đủ đi 1.200 km không?", 1200)],
)
def test_xang_hoi_km_chuyen_di_sang_tam_xa(cau, km):
    assert sua_loi_goi(cau, "get_fuel_level", {}) == (SUA, "get_range_estimate", {"trip_km": km},
                                                      "xang_hoi_km_chuyen_di")


def test_km_vo_ly_bi_bo():
    assert sua_loi_goi("Xăng còn đủ để đi thêm 5000000 km không?", "get_fuel_level", {})[0] == BO
    assert sua_loi_goi("Đi 9000km có tới không?", "get_range_estimate", {"trip_km": 9000})[0] == BO


def test_hoi_tam_xa_khong_noi_xang_la_range_estimate():
    kq = sua_loi_goi("Xe còn đi được bao xa nữa?", "get_fuel_level", {})
    assert kq[:3] == (SUA, "get_range_estimate", {})


@pytest.mark.parametrize(
    "cau",
    ["Xăng còn bao nhiêu phần trăm?", "Xăng còn đủ để đi bao xa?", "Bình xăng còn đủ đi bao xa nữa?",
     "Đã đến mức phải đổ xăng chưa?"],
)
def test_cau_hoi_xang_giu_nguyen(cau):
    assert sua_loi_goi(cau, "get_fuel_level", {})[0] == GIU


def test_range_estimate_hop_ly_giu_nguyen():
    kq = sua_loi_goi("Chuyến 350 cây nữa có cần dừng không?", "get_range_estimate", {"trip_km": 350})
    assert kq[0] == GIU


# ── âm lượng ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "cau,cu,moi",
    [("Tăng âm lượng lên 18", {"delta": 18}, {"set_value": 18}),
     ("Giảm âm lượng xuống 10", {"delta": -10}, {"set_value": 10}),
     ("Vặn nhỏ tiếng xuống 5", {"delta": -5}, {"set_value": 5}),
     ("Chỉnh âm lượng về mức 12", {"delta": 12}, {"set_value": 12}),
     ("Tăng âm lượng lên 5 nấc", {"set_value": 5}, {"delta": 5}),
     ("Giảm âm lượng đi 3 mức", {"set_value": 3}, {"delta": -3}),
     ("Bớt 4 nấc âm lượng", {"set_value": 4}, {"delta": -4})],
)
def test_quy_uoc_am_luong(cau, cu, moi):
    kq = sua_loi_goi(cau, "set_volume", cu)
    assert kq[:3] == (SUA, "set_volume", moi)


@pytest.mark.parametrize(
    "cau,args",
    [("Tăng âm lượng lên 5 nấc", {"delta": 5}), ("Đặt âm lượng mức 20", {"set_value": 20}),
     ("Tăng âm lượng thêm 8", {"delta": 8}), ("Giảm âm lượng 5", {"delta": -5}),
     ("Vặn nhỏ tiếng xuống", {"delta": -3})],
)
def test_am_luong_da_dung_hoac_khong_du_can_cu_thi_giu(cau, args):
    assert sua_loi_goi(cau, "set_volume", args)[0] == GIU


# ── kính ───────────────────────────────────────────────────────────────────
def test_dong_cua_so_la_nang_kinh():
    kq = sua_loi_goi("Đóng hết cửa sổ lại", "control_window", {"window": "all", "position": "down"})
    assert kq[:3] == (SUA, "control_window", {"window": "all", "position": "up"})


def test_mo_cua_so_bi_gan_up_thi_sua():
    kq = sua_loi_goi("Mở cửa sổ ghế lái", "control_window", {"window": "front_left", "position": "up"})
    assert kq[2]["position"] == "down"


def test_cau_tu_mau_thuan_bi_bo():
    kq = sua_loi_goi("Đóng cửa sổ ghế lái xuống 50%", "control_window",
                     {"window": "front_left", "position": "down", "amount": 50})
    assert kq[0] == BO


@pytest.mark.parametrize(
    "cau,pos",
    [("Hạ kính ghế phụ xuống", "down"), ("Nâng kính lên hết", "up"), ("Mở cửa sổ ghế sau", "down"),
     ("Đóng kính lại", "up")],
)
def test_kinh_dung_chieu_giu_nguyen(cau, pos):
    assert sua_loi_goi(cau, "control_window", {"window": "all", "position": pos})[0] == GIU


# ── từ chối nhầm ───────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "cau,nhom",
    [("Bạn có thể tìm cho tôi một cây ATM gần đây không?", "atm"),
     ("Gợi ý cho tôi một nhà hàng ngon gần đây", "restaurant")],
)
def test_tu_choi_nham_cau_find_nearby_lam_duoc(cau, nhom):
    assert sua_loi_goi(cau, "irrelevant_function", {})[:3] == (SUA, "find_nearby", {"category": nhom})


@pytest.mark.parametrize(
    "cau",
    ["Tự lái xe đến bệnh viện gần nhất", "Đặt một bàn tại nhà hàng gần đây", "Lịch chiếu phim ở rạp gần nhất",
     "Số điện thoại của garage gần nhất", "Thời gian hoạt động của quán cà phê gần nhất",
     "Tìm một bác sĩ gần đây", "Kể chuyện cười đi"],
)
def test_tu_choi_dung_giu_nguyen(cau):
    assert sua_loi_goi(cau, "irrelevant_function", {})[0] == GIU


# ── cả kho ─────────────────────────────────────────────────────────────────
def test_sua_rows_doi_nhan_va_danh_dau():
    r = CallRecord("Xăng còn đủ để đi 300km không?", "get_fuel_level").to_row()
    r["meta"]["uu_tien"] = True
    ra, bc = sua_rows([r])
    assert ra[0]["completion"].startswith("<nexa_5>(trip_km=300)<nexa_end>")
    assert ra[0]["meta"] == {**r["meta"], "fn": "get_range_estimate", "sua_nhan": "xang_hoi_km_chuyen_di"}
    assert bc["dem"] == {"call:sua:xang_hoi_km_chuyen_di": 1}


def test_reply_dung_tren_loi_goi_sai_bi_bo():
    args = {"window": "all", "position": "down"}
    kq = {**args, "opening_percent": 100}
    sai = ReplyRecord("Đóng hết cửa sổ lại", "control_window", args, kq, "Đã hạ hết kính.").to_row()
    dung = ReplyRecord("Hạ hết cửa sổ", "control_window", args, kq, "Đã hạ hết kính.").to_row()
    ra, bc = sua_rows([sai, dung])
    assert ra == [dung]
    assert bc["dem"] == {"reply:bo:kinh_nguoc_chieu": 1}


def test_cau_trung_khac_nhan_bi_bo_ca_hai_ban():
    a = CallRecord("Pin còn bao nhiêu phần trăm?", "get_battery_range").to_row()
    b = CallRecord("Pin còn bao nhiêu phần trăm?", "get_vehicle_status").to_row()
    c = CallRecord("Pin còn bao nhiêu phần trăm?", "get_battery_range").to_row()
    khac = CallRecord("Xăng còn bao nhiêu?", "get_fuel_level").to_row()
    ra, bc = sua_rows([a, b, c, khac, dict(khac)])
    assert ra == [khac, khac]  # trùng câu CÙNG nhãn thì giữ
    assert bc["dem"] == {"call:bo:trung_cau_khac_nhan": 3}


def test_mau_hoi_lai_va_mau_khong_lien_quan_khong_bi_dung():
    ask = {"prompt": negative_record("Đóng cửa mãi bí quá").to_row()["prompt"],
           "completion": "<nexa_ask><nexa_15>(window=\"all\", position=\"down\")<nexa_end><|im_end|>",
           "meta": {"lang": "vi", "type": "ask", "fn": "control_window"}}
    khac = CallRecord("Bật điều hoà 22 độ", "set_ac", {"state": "on", "temperature": 22}).to_row()
    ra, bc = sua_rows([ask, khac])
    assert ra == [ask, khac] and bc["dem"] == {}
