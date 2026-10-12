"""Nhóm A (2026-10-07, PLAN.md Phase 7) — lỗi người dùng gặp trên demo, sửa bằng code (không train lại)."""

import pytest

from schema.call_format import render_completion
from serve.chuan_hoa import chuan_hoa_chu
from serve.contracts import AssistantTurn, ClarifyTurn
from serve.mo_hinh import MoHinhGia
from serve.quy_uoc import ap_quy_uoc_chay
from serve.tac_tu import TacTu
from serve.xe_mo_phong import TrangThaiXe, XeMoPhong


def luot(sk) -> AssistantTurn:
    return next(s for s in sk if isinstance(s, AssistantTurn))


# ── 1. ngữ cảnh làm model từ chối -> hỏi lại không kèm lượt trước ─────────────────────────────────
class _MoHinhNgCanh(MoHinhGia):
    """Có lượt trước thì từ chối, không có thì đề xuất mở cửa sổ trời (đúng hiện tượng đo trên v6.1)."""

    nho_luot_truoc = True

    def goi_ham(self, cau, luot_truoc=None):
        if cau == "Ngoài trời đẹp thật":
            if luot_truoc:
                return render_completion([("irrelevant_function", {})])
            return '<nexa_ask><nexa_16>(action="open")<nexa_end>'
        return super().goi_ham(cau, luot_truoc)


def test_ngu_canh_lam_tu_choi_thi_dung_ban_khong_ngu_canh():
    tt = TacTu(_MoHinhNgCanh(), XeMoPhong())
    tt.xu_ly("Hạ kính ghế phụ")
    sk = tt.xu_ly("ngoài trời đẹp thật")
    assert any(isinstance(s, ClarifyTurn) and "cửa sổ trời" in s.text for s in sk)


def test_khong_ngu_canh_cung_tu_choi_thi_van_tu_choi():
    tu_choi = render_completion([("irrelevant_function", {})])
    tt = TacTu(MoHinhGia(bang={"Kể chuyện cười đi": tu_choi}), XeMoPhong())
    tt.mo_hinh.nho_luot_truoc = True
    tt.xu_ly("Mở cốp")
    assert luot(tt.xu_ly("kể chuyện cười đi")).loai == "tu_choi"


# ── 3. quy ước gán nhãn áp lại lúc chạy ─────────────────────────────────────────────────────────
# "nữa" -> ±2 · có số thì giữ · quạt ±1 · âm lượng ±3 · "2 kính" = số kính · có mức ("một nửa") thì giữ amount
@pytest.mark.parametrize("ten, args, cau, ra", [
    ("set_ac", {"state": "on", "delta": -1}, "giảm nữa", {"state": "on", "delta": -2}),
    ("set_ac", {"state": "on", "delta": -3}, "giảm 3 độ", {"state": "on", "delta": -3}),
    ("set_fan_speed", {"delta": -2}, "giảm quạt", {"delta": -1}),
    ("set_volume", {"delta": 2}, "to lên", {"delta": 3}),
    ("control_window", {"window": "rear", "position": "down", "amount": 2}, "hạ 2 kính ghế sau xuống",
     {"window": "rear", "position": "down"}),
    ("control_window", {"window": "rear", "position": "down", "amount": 50}, "hạ 2 kính sau một nửa",
     {"window": "rear", "position": "down", "amount": 50}),
])
def test_quy_uoc_luc_chay(ten, args, cau, ra):
    assert ap_quy_uoc_chay(ten, args, cau) == ra


def test_de_xuat_den_cho_cau_than_phien_khong_bi_doi_theo_chu_xa():
    # "rọi xa quá làm chói" -> đề xuất chiếu gần ĐÚNG; luật chữ "xa" -> chiếu xa chỉ dành cho LỆNH
    cau = "Mấy xe ngược chiều cứ nháy đèn, chắc đèn mình rọi xa quá làm người ta chói mắt"
    assert ap_quy_uoc_chay("set_headlights", {"mode": "low_beam"}, cau, de_xuat=True) == {"mode": "low_beam"}
    assert ap_quy_uoc_chay("set_headlights", {"mode": "low_beam"}, "bật đèn pha", de_xuat=False) == {
        "mode": "high_beam"}


@pytest.mark.parametrize("vao, ra", [
    ("NÓNG QUÁ", "Nóng quá"),                                   # đo: model đề xuất âm lượng khi để in hoa
    ("MỞ CỐP ĐI", "Mở cốp đi"),
    ("Dẫn đường tới sân bay Nội Bài", "Dẫn đường tới sân bay Nội Bài"),  # câu thường: tên riêng giữ nguyên
    ("OK", "OK"),                                                # quá ngắn: không đổi
    ("bật USB", "Bật USB"),
])
def test_cau_toan_in_hoa(vao, ra):
    assert chuan_hoa_chu(vao) == ra


def test_go_lap_chu():
    assert chuan_hoa_chu("mở cốpp") == "Mở cốp"
    assert chuan_hoa_chu("alooo bật nhạc") == "Alo bật nhạc"
    assert chuan_hoa_chu("đi thêm 1000 km") == "Đi thêm 1000 km"  # chữ số không đụng


@pytest.mark.parametrize("co_xe, cau, loi_goi, chua", [
    ({"fan_level": 0}, "giảm quạt một nấc", ("set_fan_speed", {"delta": -1}), "Quạt gió đang tắt rồi."),
    ({"fan_level": 7}, "tăng quạt", ("set_fan_speed", {"delta": 1}), "mạnh nhất (7) rồi"),
    ({"volume": 0}, "nhỏ tiếng xuống", ("set_volume", {"delta": -3}), "Âm lượng đã nhỏ nhất rồi."),
    ({"ac_on": True, "ac_temperature": 16}, "giảm nhiệt độ", ("set_ac", {"state": "on", "delta": -2}),
     "thấp nhất 16 độ"),
])
def test_cham_gioi_han_thi_noi_ro(co_xe, cau, loi_goi, chua):
    tt = TacTu(MoHinhGia(bang={cau: render_completion([loi_goi])}), XeMoPhong(TrangThaiXe(**co_xe)))
    assert chua in luot(tt.xu_ly(cau)).text


def test_giam_quat_tu_1_xuong_0_van_bao_da_tat():
    tt = TacTu(MoHinhGia(bang={"giảm quạt": render_completion([("set_fan_speed", {"delta": -1})])}),
               XeMoPhong(TrangThaiXe(fan_level=1)))
    assert luot(tt.xu_ly("giảm quạt")).text == "Đã tắt quạt gió."


# ── 4. trợ lý xưng "mình" ──────────────────────────────────────────────────────────────────────────
def test_tro_ly_xung_minh():
    from serve import tra_loi

    for cau in (tra_loi.TU_CHOI, tra_loi.CHUA_HIEU, tra_loi.CHUA_RO, tra_loi.HUY):
        assert "tôi" not in cau.lower().split() and "mình" in cau.lower()
    tt = TacTu(MoHinhGia(), XeMoPhong(TrangThaiXe(dang_chay=True)))
    assert "mình không mở cốp" in luot(tt.xu_ly("Mở cốp giúp tôi")).text
