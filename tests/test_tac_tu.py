"""Phase 5 — kịch bản cả luồng, chấm KIỂU τ-bench: nói xong thì kiểm TRẠNG THÁI XE CUỐI, không so câu chữ.

Kịch bản lấy từ tiêu chí "Xong khi" của Phase 5 trong PLAN.md. Dùng model giả (luật từ khoá) nên kết quả
xác định — test đúng LUỒNG (router, guardrail, xe), không test độ giỏi của model.
"""

import pytest

from serve.contracts import AssistantTurn, ClarifyTurn, StateEvent
from serve.mo_hinh import MoHinhGia
from serve.tac_tu import TacTu
from serve.tra_loi import CHUA_HIEU, HUY, TU_CHOI
from serve.xe_mo_phong import TrangThaiXe, XeMoPhong


def tac_tu(**co_xe) -> TacTu:
    return TacTu(MoHinhGia(), XeMoPhong(TrangThaiXe(**co_xe)))


def luot(su_kien) -> AssistantTurn:
    return next(s for s in su_kien if isinstance(s, AssistantTurn))


def hoi(su_kien) -> ClarifyTurn:
    return next(s for s in su_kien if isinstance(s, ClarifyTurn))


# ── làm luôn ───────────────────────────────────────────────────────────────
def test_mo_cop_khi_do():
    a = tac_tu()
    sk = a.xu_ly("Mở cốp giúp tôi")
    assert a.xe.s.trunk_open is True
    assert luot(sk).loai == "thuc_thi" and luot(sk).calls[0].name == "open_trunk"
    assert luot(sk).ui_actions[0].animation == "trunk_open"
    assert any(isinstance(s, StateEvent) for s in sk)  # frontend nhận trạng thái mới


def test_hoi_thong_tin_khong_doi_trang_thai_xe():
    a = tac_tu()
    sk = a.xu_ly("Xăng còn bao nhiêu?")
    assert "62%" in luot(sk).text
    assert not any(isinstance(s, StateEvent) for s in sk)


# ── luật an toàn ───────────────────────────────────────────────────────────
def test_mo_cop_khi_xe_chay_bi_tu_choi():
    a = tac_tu(dang_chay=True)
    sk = a.xu_ly("Mở cốp giúp tôi")
    assert a.xe.s.trunk_open is False
    assert luot(sk).loai == "tu_choi" and "đang chạy" in luot(sk).text


def test_khoa_tre_em_tai_xe_bam_mic_thi_ha_kinh_sau_luon():
    a = tac_tu(khoa_tre_em=True)
    a.xu_ly("Hạ kính sau bên trái")  # nguồn mặc định = nút mic = tài xế
    assert a.xe.s.windows["rear_left"] == 100


def test_khoa_tre_em_wake_word_ha_kinh_sau_phai_xac_nhan():
    a = tac_tu(khoa_tre_em=True)
    h = hoi(a.xu_ly("Hạ kính sau bên trái", nguon="wake_word"))
    assert "Khoá trẻ em" in h.text and a.xe.s.windows["rear_left"] == 0
    a.xu_ly("có")
    assert a.xe.s.windows["rear_left"] == 100


def test_xe_chay_ha_kinh_sau_mo_het():
    a = tac_tu(dang_chay=True)
    a.xu_ly("Hạ kính sau")
    assert a.xe.s.windows["rear_left"] == 100 and a.xe.s.windows["rear_right"] == 100


def test_kinh_xe_sau_chi_mo_hai_kinh_sau():
    """Người dùng thử thật (2026-09-30): "mở kính xe sau" từng mở CẢ 4 kính vì chữ "xe" chen giữa."""
    a = tac_tu()
    a.xu_ly("Mở kính xe sau")
    assert a.xe.s.windows == {"front_left": 0, "front_right": 0, "rear_left": 100, "rear_right": 100}


def test_khong_ro_kinh_nao_thi_hoi_lai_khong_mo_tat_ca():
    a = tac_tu()
    h = hoi(a.xu_ly("Hạ kính xuống"))
    assert h.ly_do == "thieu_tham_so" and "cửa nào" in h.text
    assert set(a.xe.s.windows.values()) == {0}


def test_noi_ro_tat_ca_thi_mo_het():
    a = tac_tu()
    a.xu_ly("Hạ hết kính xuống")
    assert set(a.xe.s.windows.values()) == {100}


def test_mo_khoa_khi_chay_hoi_lai_roi_xac_nhan():
    a = tac_tu(dang_chay=True)
    assert hoi(a.xu_ly("Mở khoá cửa")).ly_do == "an_toan"
    assert a.xe.s.doors_locked is True  # chưa làm khi chưa xác nhận
    a.xu_ly("có")
    assert a.xe.s.doors_locked is False


def test_lenh_nhay_cam_tu_wake_word_phai_xac_nhan():
    a = tac_tu()
    assert hoi(a.xu_ly("Mở cốp", nguon="wake_word")).ly_do == "an_toan"
    assert a.xe.s.trunk_open is False
    a.xu_ly("được")
    assert a.xe.s.trunk_open is True


# ── hỏi lại khi câu mơ hồ ──────────────────────────────────────────────────
def test_mo_ho_roi_dong_y():
    a = tac_tu()
    assert hoi(a.xu_ly("Nóng quá")).ly_do == "de_xuat"
    assert a.xe.s.ac_on is False
    a.xu_ly("ừ bật đi")
    assert a.xe.s.ac_on is True


def test_mo_ho_roi_tu_choi():
    a = tac_tu()
    a.xu_ly("Nóng quá")
    sk = a.xu_ly("thôi khỏi")
    assert luot(sk).text == HUY and a.xe.s.ac_on is False and a.dang_cho is None


def test_mo_ho_roi_noi_viec_khac_thi_huy_de_xuat_va_lam_viec_moi():
    a = tac_tu()
    a.xu_ly("Nóng quá")
    a.xu_ly("dạ, mở cốp đi")  # "dạ" đứng đầu nhưng là LỆNH MỚI, không phải đồng ý bật điều hoà
    assert a.xe.s.trunk_open is True and a.xe.s.ac_on is False and a.dang_cho is None


def test_de_xuat_hai_lua_chon_noi_co_thi_hoi_chon():
    a = tac_tu()
    a.xu_ly("Trong xe bí quá")
    assert hoi(a.xu_ly("có")).ly_do == "chon"  # không tự chọn thay người lái
    a.xu_ly("cái thứ hai")
    assert set(a.xe.s.windows.values()) == {100} and a.xe.s.ac_on is False


def test_khong_bao_gio_tu_de_xuat_ham_nhay_cam():
    a = TacTu(MoHinhGia(bang={"Có người đợi ngoài kia": "<nexa_ask><nexa_10>()<nexa_end>"}), XeMoPhong())
    sk = a.xu_ly("Có người đợi ngoài kia")
    assert not any(isinstance(s, ClarifyTurn) for s in sk) and a.dang_cho is None
    assert a.xe.s.doors_locked is True


# ── validator ──────────────────────────────────────────────────────────────
def test_model_sinh_gia_tri_sai_thi_khong_dung_vao_xe():
    a = TacTu(MoHinhGia(bang={"Phun cánh ra để bay": '<nexa_16>(action="fly")<nexa_end>'}), XeMoPhong())
    sk = a.xu_ly("Phun cánh ra để bay")
    assert luot(sk).loai == "chua_hieu" and luot(sk).text == CHUA_HIEU
    assert a.xe.s.sunroof_state == "close"


def test_thieu_tham_so_thi_hoi_lai():
    a = TacTu(MoHinhGia(bang={"Hạ kính xuống": '<nexa_15>(position="down")<nexa_end>'}), XeMoPhong())
    h = hoi(a.xu_ly("Hạ kính xuống"))
    assert h.ly_do == "thieu_tham_so" and h.missing == ["window"] and "cửa nào" in h.text


def test_output_hong_thi_bao_chua_hieu():
    a = TacTu(MoHinhGia(bang={"abc": "chuỗi rác không phải lời gọi"}), XeMoPhong())
    assert luot(a.xu_ly("abc")).loai == "chua_hieu"


# ── từ chối + lưới an toàn lượt 2 ──────────────────────────────────────────
def test_ngoai_pham_vi_tu_choi_va_xe_khong_doi():
    a = tac_tu()
    truoc = a.xe.anh_chup()
    sk = a.xu_ly("Kể chuyện cười đi")
    assert luot(sk).loai == "tu_choi" and luot(sk).text == TU_CHOI
    assert a.xe.anh_chup() == truoc


def test_xin_chao_dap_ten_tro_ly_khong_qua_model():
    a = TacTu(MoHinhGia(bang={"Xin chào": "<nexa_ask><nexa_0>()<nexa_end>"}), XeMoPhong())  # model hỏi thừa
    tl = luot(a.xu_ly("Xin chào"))
    assert tl.loai == "tro_chuyen" and tl.text == "Chào bạn, mình là NexaAI, mình có thể giúp gì cho bạn nhỉ?"


@pytest.mark.parametrize(
    "cau, chua",
    [("cảm ơn nhé", "Không có gì"), ("tạm biệt", "Tạm biệt"), ("bạn là ai", "Mình là NexaAI"),
     ("bạn làm được những gì", "Mình giúp được"), ("thông minh quá", "khen")],
)
def test_cau_xa_giao_dap_theo_loai(cau, chua):
    tl = luot(tac_tu().xu_ly(cau))
    assert tl.loai == "tro_chuyen" and chua in tl.text and not tl.calls


def test_chao_kem_lenh_van_lam_lenh():
    a = tac_tu()
    tl = luot(a.xu_ly("Chào bạn, mở cốp giúp tôi"))
    assert tl.loai == "thuc_thi" and a.xe.s.trunk_open is True


@pytest.mark.parametrize(
    "cau", ["hỏng xe giữa đường rồi", "Hôm nay trời đẹp ghê", "Đặt vé xem phim giúp tôi"]
)
def test_model_tu_choi_cau_khong_phai_xa_giao_thi_xin_loi_trung_tinh(cau):
    # người dùng gặp 2026-10-07: model từ chối nhầm "hỏng xe giữa đường rồi" -> từng đáp vui "Đúng vậy nhỉ!"
    a = TacTu(MoHinhGia(bang={cau: "<nexa_24>()<nexa_end>"}), XeMoPhong())
    tl = luot(a.xu_ly(cau))
    assert tl.loai == "tu_choi" and tl.text == TU_CHOI


def test_cau_go_chu_thuong_dua_vao_model_voi_chu_dau_viet_hoa():
    # người dùng gặp 2026-10-07: "hỏng xe giữa đường rồi" (thường) bị từ chối, viết hoa thì hỏi đèn cảnh báo
    a = tac_tu()
    nhan: list[str] = []
    goc = a.mo_hinh.goi_ham
    a.mo_hinh.goi_ham = lambda c, truoc=None: (nhan.append(c), goc(c))[1]  # type: ignore[method-assign]
    a.xu_ly("hỏng   xe giữa đường rồi")
    assert nhan == ["Hỏng xe giữa đường rồi"]


def test_cau_dai_co_chao_van_qua_model():
    # "chào" nhưng là than lạnh -> không đi lối tắt xã giao; model giả đề xuất điều hoà -> hỏi lại
    cau = "Chào buổi sáng, trong xe lạnh quá"
    a = TacTu(MoHinhGia(bang={cau: '<nexa_ask><nexa_6>(state="on")<nexa_end>'}), XeMoPhong())
    assert any(isinstance(s, ClarifyTurn) for s in a.xu_ly(cau))


def test_mac_dinh_luot_2_dung_cau_mau_ke_ca_khi_model_viet_dung():
    # người dùng chốt 2026-10-05: câu mẫu mặc định (đúng số, tức thì); model lượt 2 chỉ khi luot_2="model"
    a = TacTu(MoHinhGia(cau_tra_loi={"*": "Xăng còn 62%, bạn yên tâm."}), XeMoPhong())
    assert luot(a.xu_ly("Xăng còn bao nhiêu?")).text.startswith("Xăng còn 62%, đi được khoảng")
    assert luot(a.xu_ly("Kể chuyện cười đi")).text == TU_CHOI


def _model_viet(cau_tra_loi):
    return TacTu(MoHinhGia(cau_tra_loi=cau_tra_loi), XeMoPhong(), luot_2="model")


def test_cau_tra_loi_cua_model_duoc_dung_khi_dung_so():
    a = _model_viet({"get_fuel_level": "Xăng còn 62%, bạn yên tâm."})
    assert luot(a.xu_ly("Xăng còn bao nhiêu?")).text == "Xăng còn 62%, bạn yên tâm."


def test_cau_tra_loi_bia_so_thi_dung_cau_mau():
    a = _model_viet({"get_fuel_level": "Xăng còn 95%, đi thêm 900 km."})
    text = luot(a.xu_ly("Xăng còn bao nhiêu?")).text
    assert "95%" not in text and "62%" in text


def test_cau_tra_loi_sinh_lai_loi_goi_thi_dung_cau_mau():
    a = _model_viet({"open_trunk": "<nexa_11>()<nexa_end>"})
    assert luot(a.xu_ly("Mở cốp")).text == "Đã mở cốp sau."


def test_do_tre_duoc_ghi():
    assert luot(tac_tu().xu_ly("Mở cốp")).latency_ms is not None
