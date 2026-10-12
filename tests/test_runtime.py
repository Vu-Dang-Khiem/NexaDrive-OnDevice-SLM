"""Phase 5 — từng mảnh của luồng chạy thật: tool (xe mô phỏng), validator, luật an toàn, schema."""

import pytest

from gen.cham_ask import HAM_NHAY_CAM
from schema.loader import CATALOG
from serve import an_toan
from serve.kiem_tra import THIEU_MUC_AM_LUONG, kiem_tra
from serve.mo_hinh import MoHinhGia
from serve.tra_loi import bao_ket_qua
from serve.xe_mo_phong import TrangThaiXe, XeMoPhong

# Một lời gọi HỢP LỆ cho mỗi hàm — để chạy thử cả 24 tool
MAU = {
    "get_fuel_level": {},
    "get_battery_range": {},
    "get_tire_pressure": {},
    "get_vehicle_status": {},
    "get_odometer": {},
    "get_range_estimate": {"trip_km": 200},
    "set_ac": {"state": "on", "temperature": 22},
    "set_fan_speed": {"level": 4},
    "set_defrost": {"state": "on"},
    "lock_doors": {},
    "unlock_doors": {},
    "open_trunk": {},
    "close_trunk": {},
    "open_charge_port": {},
    "open_fuel_flap": {},
    "control_window": {"window": "front_right", "position": "down", "amount": 50},
    "operate_sunroof": {"action": "tilt"},
    "set_headlights": {"mode": "low_beam"},
    "set_hazard_lights": {"state": "on"},
    "set_volume": {"delta": 3},
    "play_media": {"query": "Sơn Tùng"},
    "navigate_to": {"destination": "sân bay Nội Bài"},
    "find_nearby": {"category": "coffee"},
    "get_weather": {"day": 1},
    "irrelevant_function": {},
}


# ── schema ──────────────────────────────────────────────────────────────────
def test_moi_ham_co_mau_thu():
    assert set(MAU) == {f.name for f in CATALOG.functions}


def test_ham_nhay_cam_trong_schema_khop_day_chuyen_du_lieu():
    """Runtime và dây chuyền dữ liệu phải cùng một danh sách hàm nhạy cảm."""
    assert {f.name for f in CATALOG.functions if f.nhay_cam} == set(HAM_NHAY_CAM)


# ── tool: xe mô phỏng ──────────────────────────────────────────────────────
@pytest.mark.parametrize("ten", sorted(MAU))
def test_ket_qua_tool_dung_ten_truong_nhu_luc_train(ten):
    """Lượt 2 của model được train trên kết quả có đúng các trường `returns` — xe thật phải trả y như vậy."""
    kq = XeMoPhong().thuc_thi(ten, MAU[ten])
    assert set(kq) == set(CATALOG.by_name[ten].returns)


@pytest.mark.parametrize("ten", sorted(MAU))
def test_moi_ket_qua_co_cau_mau(ten):
    cau = bao_ket_qua(ten, MAU[ten], XeMoPhong().thuc_thi(ten, MAU[ten]))
    assert cau and "{" not in cau


def test_xe_nho_trang_thai():
    xe = XeMoPhong()
    xe.thuc_thi("open_trunk", {})
    assert xe.thuc_thi("get_vehicle_status", {})["trunk_open"] is True
    xe.thuc_thi("control_window", {"window": "all", "position": "down"})
    assert set(xe.s.windows.values()) == {100}
    xe.thuc_thi("control_window", {"window": "rear_left", "position": "up", "amount": 70})
    assert xe.s.windows["rear_left"] == 30


def test_am_luong_khong_vuot_gioi_han():
    xe = XeMoPhong()
    assert xe.thuc_thi("set_volume", {"delta": 40})["volume"] == 40
    assert xe.thuc_thi("set_volume", {"set_value": 5})["volume"] == 5


def test_dich_vu_ngoai_tai_lap_duoc():
    a = XeMoPhong().thuc_thi("navigate_to", {"destination": "Hồ Gươm"})
    b = XeMoPhong().thuc_thi("navigate_to", {"destination": "Hồ Gươm"})
    assert a == b


# ── guardrail 1: validator ─────────────────────────────────────────────────
def test_gia_tri_ngoai_danh_sach_bi_chan():
    """Đã thấy thật trên tập test: model sinh operate_sunroof(action="fly")."""
    kq = kiem_tra("operate_sunroof", {"action": "fly"})
    assert not kq.hop_le and kq.loi and not kq.thieu


def test_thieu_tham_so_bat_buoc_thi_hoi_lai():
    kq = kiem_tra("control_window", {"position": "down"})
    assert kq.thieu == ["window"] and not kq.loi


def test_tham_so_bat_buoc_rong_tinh_la_thieu():
    """Dữ liệu hỏi lại có `navigate_to(destination="")` ("sợ lạc quá" — chưa biết đi đâu). Người dùng
    đồng ý thì phải hỏi "đi đâu?", không dẫn đường tới chuỗi rỗng (người dùng bắt lỗi khi kiểm tay)."""
    for rong in ("", "   "):
        kq = kiem_tra("navigate_to", {"destination": rong})
        assert kq.thieu == ["destination"] and not kq.loi and not kq.hop_le


def test_tham_so_la_bi_bo_qua_va_con_thieu_thi_hoi_lai():
    """Đã thấy thật: set_hazard_lights(level=3) — bỏ `level`, thiếu `state` -> hỏi lại."""
    kq = kiem_tra("set_hazard_lights", {"level": 3})
    assert kq.bo_qua == ["level"] and kq.thieu == ["state"]


def test_ep_kieu_va_kiem_khoang():
    assert kiem_tra("set_ac", {"state": "on", "temperature": "22"}).args["temperature"] == 22
    assert not kiem_tra("set_ac", {"state": "on", "temperature": 45}).hop_le


def test_am_luong_can_dung_mot_trong_hai_tham_so():
    assert not kiem_tra("set_volume", {"set_value": 10, "delta": 2}).hop_le
    assert kiem_tra("set_volume", {}).thieu == [THIEU_MUC_AM_LUONG]


# ── guardrail 2: luật an toàn ──────────────────────────────────────────────
def test_mo_cop_khi_do_thi_duoc_khi_chay_thi_tu_choi():
    assert an_toan.xet("open_trunk", {}, TrangThaiXe()).xu_ly == "cho_phep"
    assert an_toan.xet("open_trunk", {}, TrangThaiXe(dang_chay=True)).xu_ly == "tu_choi"


def test_mo_khoa_khi_chay_thi_hoi_lai():
    assert an_toan.xet("unlock_doors", {}, TrangThaiXe(dang_chay=True)).xu_ly == "hoi_lai"


@pytest.mark.parametrize("ten", sorted(HAM_NHAY_CAM))
def test_ham_nhay_cam_tu_wake_word_phai_xac_nhan(ten):
    assert an_toan.xet(ten, {}, TrangThaiXe(), nguon="wake_word").xu_ly == "hoi_lai"
    assert an_toan.xet(ten, {}, TrangThaiXe(), nguon="nut").xu_ly == "cho_phep"


def test_khoa_tre_em_chi_xac_nhan_lenh_wake_word_khong_chan_tai_xe():
    """Người dùng chốt 2026-10-07: khoá trẻ em như nút khoá kính xe thật — chặn ghế sau, không chặn tài xế."""
    xe = TrangThaiXe(khoa_tre_em=True)
    for kinh in ("rear_left", "rear", "all"):
        ha = {"window": kinh, "position": "down"}
        assert an_toan.xet("control_window", ha, xe, nguon="nut").xu_ly == "cho_phep"  # bấm mic = tài xế
        assert an_toan.xet("control_window", ha, xe, nguon="wake_word").xu_ly == "hoi_lai"
    for args in ({"window": "front_left", "position": "down"}, {"window": "rear_left", "position": "up"}):
        assert an_toan.xet("control_window", args, xe, nguon="wake_word").xu_ly == "cho_phep"
    # khoá trẻ em TẮT thì wake-word hạ kính sau làm luôn (kính không phải hàm nhạy cảm)
    tat = {"window": "rear", "position": "down"}
    assert an_toan.xet("control_window", tat, TrangThaiXe(), nguon="wake_word").xu_ly == "cho_phep"


def test_khoa_tre_em_mo_cua_so_troi_tu_wake_word_phai_xac_nhan():
    xe = TrangThaiXe(khoa_tre_em=True)
    assert an_toan.xet("operate_sunroof", {"action": "open"}, xe, nguon="nut").xu_ly == "cho_phep"
    assert an_toan.xet("operate_sunroof", {"action": "tilt"}, xe, nguon="wake_word").xu_ly == "hoi_lai"
    assert an_toan.xet("operate_sunroof", {"action": "close"}, xe, nguon="wake_word").xu_ly == "cho_phep"


def test_xe_chay_ha_kinh_khong_gioi_han_phan_tram():
    """Người dùng bỏ luật 30% (2026-10-07) — xe thật không giới hạn; trước đó "hạ hết kính" còn lọt luật."""
    xe = TrangThaiXe(dang_chay=True)
    for kinh in ("rear_right", "rear", "all"):
        qd = an_toan.xet("control_window", {"window": kinh, "position": "down"}, xe)
        assert qd.xu_ly == "cho_phep" and "amount" not in qd.args


# ── model giả ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "cau,dau",
    [
        ("Mở cốp giúp tôi", "<nexa_11>()"),
        ("Nóng quá", "<nexa_ask><nexa_6>"),
        ("Kể chuyện cười đi", "<nexa_24>()"),
        ("Bật điều hoà 22 độ", '<nexa_6>(state="on", temperature=22)'),
        ("Hạ kính sau", '<nexa_15>(window="rear"'),  # schema v6: cả hai kính sau = "rear"
    ],
)
def test_model_gia_sinh_dung_dinh_dang(cau, dau):
    assert MoHinhGia().goi_ham(cau).startswith(dau)
