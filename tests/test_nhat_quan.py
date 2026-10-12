"""Test 2 lỗi do thẩm định chéo (giám khảo Llama-3.3-70B) phát hiện."""

from gen.loc import ket_qua_nhat_quan, loc_rows
from gen.records import ReplyRecord
from gen.simulate import simulate_result


# ── lỗi 1: simulate_result bỏ qua radius_km ────────────────────────────────
def test_find_nearby_ton_trong_ban_kinh():
    """Trước khi sửa: xin 1 km vẫn trả về địa điểm 4.9 km."""
    for seed in range(30):
        kq = simulate_result("find_nearby", {"category": "coffee", "radius_km": 1}, seed=seed)
        km = [float(x.split("—")[1].replace("km", "").strip()) for x in kq["results"]]
        assert all(x <= 1.0 for x in km), f"seed {seed}: {km} vượt 1 km"


def test_find_nearby_khong_noi_ban_kinh_thi_dung_mac_dinh():
    kq = simulate_result("find_nearby", {"category": "parking"}, seed=1)
    assert kq["results"] and kq["category"] == "parking"


# ── lỗi 2: navigate_to bỏ qua avoid ────────────────────────────────────────
def test_navigate_phan_anh_yeu_cau_tranh_duong():
    assert "tránh trạm thu phí" in simulate_result(
        "navigate_to", {"destination": "Nội Bài", "avoid": "tolls"}, seed=1
    )["route_summary"]
    assert "tránh cao tốc" in simulate_result(
        "navigate_to", {"destination": "Nội Bài", "avoid": "highways"}, seed=1
    )["route_summary"]


def test_navigate_khong_tranh_gi_thi_khong_them_chu():
    tom_tat = simulate_result("navigate_to", {"destination": "Nội Bài"}, seed=1)["route_summary"]
    assert "tránh" not in tom_tat


# ── dọn mẫu cũ đã sinh bằng simulator lỗi ──────────────────────────────────
def _reply_find_nearby(ban_kinh: float, khoang_cach: list[float]) -> dict:
    ds = [f"coffee_{i + 1} — {d} km" for i, d in enumerate(khoang_cach)]
    ket_qua = {"category": "coffee", "results": ds}
    return ReplyRecord(
        "Tìm quán cà phê trong bán kính 1km",
        "find_nearby",
        {"category": "coffee", "radius_km": ban_kinh},
        ket_qua,
        "Có 2 quán gần đây.",
    ).to_row()


def test_bat_duoc_ket_qua_vuot_ban_kinh():
    ok, ly_do = ket_qua_nhat_quan(_reply_find_nearby(1.0, [3.5, 0.4]))
    assert ok is False
    assert "vượt bán kính" in ly_do


def test_ket_qua_trong_ban_kinh_thi_giu():
    assert ket_qua_nhat_quan(_reply_find_nearby(5.0, [3.5, 0.4]))[0] is True


def test_mau_khong_phai_find_nearby_thi_bo_qua():
    row = ReplyRecord("còn xăng?", "get_fuel_level", {}, {"fuel_percent": 42}, "Còn 42%.").to_row()
    assert ket_qua_nhat_quan(row)[0] is True


def test_loc_rows_loai_mau_mau_thuan():
    sach, bo = loc_rows([_reply_find_nearby(1.0, [4.9]), _reply_find_nearby(5.0, [1.2])])
    assert len(sach) == 1
    assert len(bo) == 1 and "vượt bán kính" in bo[0]["ly_do_loai"]
