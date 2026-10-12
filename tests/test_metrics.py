"""Test chấm điểm (eval/metrics.py) — dữ liệu hardcode, không cần model."""

import pytest

from eval.metrics import (
    BIA_SO,
    DUNG,
    GOI_BUA,
    HOI_LAI_THUA,
    KHONG_PARSE,
    SAI_HAM,
    SAI_THAM_SO,
    THIEU_HOI_LAI,
    TRA_LOI_RONG,
    TU_CHOI_NHAM,
    cau_sai,
    cham_cau_tra_loi,
    cham_mot_cau,
    cham_toan_bo,
    doc_output,
    khong_bia_so,
)
from gen.records import CallRecord, ReplyRecord, SuggestRecord, negative_record

MO_COP = CallRecord("Mở cốp xe", "open_trunk", {}).to_row()
DIEU_HOA = CallRecord("Bật điều hòa 26 độ", "set_ac", {"state": "on", "temperature": 26}).to_row()
NONG_QUA = SuggestRecord("Nóng quá", (("set_ac", {"state": "on"}),)).to_row()
TAN_GAU = negative_record("Kể chuyện cười đi").to_row()


def _cham(row, output):
    return cham_mot_cau(output, row["completion"])[0]


# ── từng kiểu kết cục ───────────────────────────────────────────────────────
def test_dung_hoan_toan():
    assert _cham(MO_COP, MO_COP["completion"]) == DUNG


def test_sai_ham():
    assert _cham(MO_COP, "<nexa_12>()<nexa_end>") == SAI_HAM  # close_trunk


def test_sai_tham_so():
    assert _cham(DIEU_HOA, '<nexa_6>(state="on", temperature=22)<nexa_end>') == SAI_THAM_SO


def test_thieu_tham_so_cung_tinh_la_sai_tham_so():
    assert _cham(DIEU_HOA, '<nexa_6>(state="on")<nexa_end>') == SAI_THAM_SO


def test_tu_choi_nham_cau_hop_le():
    assert _cham(MO_COP, "<nexa_24>()<nexa_end>") == TU_CHOI_NHAM


def test_goi_bua_voi_cau_ngoai_pham_vi():
    assert _cham(TAN_GAU, "<nexa_20>()<nexa_end>") == GOI_BUA


def test_tu_choi_dung_cau_ngoai_pham_vi():
    assert _cham(TAN_GAU, "<nexa_24>()<nexa_end>") == DUNG


def test_hoi_lai_thua_voi_cau_ro_rang():
    assert _cham(MO_COP, "<nexa_ask><nexa_11>()<nexa_end>") == HOI_LAI_THUA


def test_thieu_hoi_lai_voi_cau_mo_ho():
    assert _cham(NONG_QUA, '<nexa_6>(state="on")<nexa_end>') == THIEU_HOI_LAI


def test_hoi_lai_dung_thi_tinh_dung():
    assert _cham(NONG_QUA, NONG_QUA["completion"]) == DUNG


def test_output_rac_khong_parse_duoc():
    assert _cham(MO_COP, "Tôi nghĩ bạn nên mở cốp xe") == KHONG_PARSE


def test_thieu_token_ket_thuc_van_chap_nhan():
    """Chấm nới lỏng giống paper: thiếu <nexa_end> vẫn tính đúng nếu lời gọi đúng."""
    assert _cham(MO_COP, "<nexa_11>()") == DUNG


def test_doc_output_bat_duoc_co_hoi_lai():
    assert doc_output('<nexa_ask><nexa_6>(state="on")<nexa_end>').hoi_lai is True
    assert doc_output('<nexa_6>(state="on")<nexa_end>').hoi_lai is False


# ── kiểm câu trả lời không bịa số ───────────────────────────────────────────
def test_khong_bia_so_hop_le():
    assert khong_bia_so("Xăng còn 42%, đi được khoảng 380 km.", {"fuel_percent": 42, "range_km": 380})


def test_bat_duoc_so_bia():
    assert not khong_bia_so("Xăng còn 42%, đi được 500 km.", {"fuel_percent": 42, "range_km": 380})


def test_so_lam_tron_van_chap_nhan():
    assert khong_bia_so("Đi được 380 km nữa.", {"range_km": 380.0})


# ── 4 kiểu diễn đạt hợp lệ, trước đây bị bắt oan (83/2122 mẫu data thật) ────
def test_lam_tron_phut_duoc_chap_nhan():
    assert khong_bia_so("Mất khoảng 75 phút.", {"eta_min": 75.8})


def test_lam_tron_qua_xa_thi_van_bat():
    assert not khong_bia_so("Mất khoảng 60 phút.", {"eta_min": 75.8})


def test_dau_ngan_nghin_kieu_viet():
    assert khong_bia_so("Công tơ mét hiện tại là 42.750 km.", {"odometer": 42750, "unit": "km"})


def test_dau_ngan_nghin_kieu_anh():
    assert khong_bia_so("Xe đã chạy 40,181.6 dặm.", {"odometer": 40181.6, "unit": "mi"})


def test_so_sai_du_co_dau_ngan_nghin_van_bat():
    assert not khong_bia_so("Công tơ mét là 52.750 km.", {"odometer": 42750})


def test_bo_dau_am_khi_dien_dat_bang_chu():
    assert khong_bia_so(
        "Không đủ, còn thiếu khoảng 218.6 km.", {"total_range_km": 231.4, "margin_km": -218.6}
    )


def test_so_lay_tu_cau_hoi_khong_tinh_la_bia():
    assert khong_bia_so(
        "Xe chỉ đi được 231.4 km nên 450 km là không đủ.",
        {"total_range_km": 231.4, "reachable": False},
        query="Đi 450km có đủ xăng không?",
    )


def test_so_khong_co_o_dau_ca_van_bi_bat():
    assert not khong_bia_so(
        "Xe đi được 231.4 km, cần đổ xăng sau 900 km.",
        {"total_range_km": 231.4},
        query="Đi 450km có đủ xăng không?",
    )


def test_cau_tra_loi_khong_co_so_thi_luon_dat():
    assert khong_bia_so("Đã mở cốp cho bạn rồi nhé.", {"trunk_open": True})


# ── cộng dồn cả tập ─────────────────────────────────────────────────────────
def _tap_test():
    return [MO_COP, DIEU_HOA, NONG_QUA, TAN_GAU]


def test_cham_toan_bo_tach_dung_3_nhom():
    du_doan = {r["prompt"]: r["completion"] for r in _tap_test()}  # model hoàn hảo
    bc = cham_toan_bo(_tap_test(), du_doan)
    assert bc["theo_nhom"]["ro_rang"] == {"so_cau": 2, "dung_%": 100.0}
    assert bc["theo_nhom"]["mo_ho"] == {"so_cau": 1, "dung_%": 100.0}
    assert bc["theo_nhom"]["ngoai_pham_vi"] == {"so_cau": 1, "dung_%": 100.0}


def test_model_tu_choi_tat_ca_bi_lo_ngay():
    """Model 'lười' phải lộ ra: nhóm từ chối 100% nhưng 2 nhóm kia 0%."""
    du_doan = {r["prompt"]: "<nexa_24>()<nexa_end>" for r in _tap_test()}
    bc = cham_toan_bo(_tap_test(), du_doan)
    assert bc["theo_nhom"]["ngoai_pham_vi"]["dung_%"] == 100.0
    assert bc["theo_nhom"]["ro_rang"]["dung_%"] == 0.0
    assert bc["ty_le_loi_%"]["tu_choi_nham"] == 75.0


def test_thong_ke_f1_tham_so():
    du_doan = {DIEU_HOA["prompt"]: '<nexa_6>(state="on", temperature=22)<nexa_end>'}
    bc = cham_toan_bo([DIEU_HOA], du_doan)
    # đúng 1/2 tham số -> P = R = 50%
    assert bc["tham_so"]["precision_%"] == 50.0
    assert bc["tham_so"]["recall_%"] == 50.0


def test_ma_tran_nham_lan():
    du_doan = {MO_COP["prompt"]: "<nexa_12>()<nexa_end>"}
    bc = cham_toan_bo([MO_COP], du_doan)
    assert bc["hay_nham_nhat"][0] == {"dap_an": "open_trunk", "model_chon": "close_trunk", "so_lan": 1}


def test_thieu_du_doan_duoc_bao_cao():
    bc = cham_toan_bo(_tap_test(), {})
    assert bc["thieu_du_doan"] == 4
    assert bc["so_cau_da_cham"] == 0


def test_danh_sach_cau_sai_co_kieu_sai():
    du_doan = {MO_COP["prompt"]: "<nexa_24>()<nexa_end>"}
    ds = cau_sai([MO_COP], du_doan)
    assert len(ds) == 1
    assert ds[0]["kieu_sai"] == TU_CHOI_NHAM
    assert ds[0]["cau"] == "Mở cốp xe"


@pytest.mark.parametrize("nhom_phu", ["oos_xe_khong_ho_tro", "oos_kien_thuc"])
def test_tach_duoc_nhom_phu(nhom_phu):
    row = {**TAN_GAU, "meta": {**TAN_GAU["meta"], "nhom_phu": nhom_phu}}
    bc = cham_toan_bo([row], {row["prompt"]: "<nexa_24>()<nexa_end>"})
    assert bc["theo_nhom_phu"][nhom_phu]["so_cau"] == 1


# ── LƯỢT 2: chấm câu trả lời tự nhiên (mẫu `reply`) ─────────────────────────
XANG = ReplyRecord(
    "Còn bao nhiêu xăng?",
    "get_fuel_level",
    {},
    {"fuel_percent": 42, "range_km": 380},
    "Xăng còn 42%, đi được khoảng 380 km.",
).to_row()


def test_tra_loi_dung_so_thi_dung():
    assert cham_cau_tra_loi(XANG["completion"], XANG["prompt"]) == DUNG


def test_dien_dat_khac_dap_an_van_dung():
    """Không so chuỗi với đáp án — nhiều cách nói đều đúng."""
    khac = "Bình xăng của bạn đang ở mức 42 phần trăm nhé."
    assert cham_cau_tra_loi(khac, XANG["prompt"]) == DUNG


def test_cham_luot_2_bat_duoc_so_bia():
    """Nguy hiểm nhất trên xe: báo sai mức xăng."""
    assert cham_cau_tra_loi("Xăng còn 60%.", XANG["prompt"]) == BIA_SO


def test_tra_loi_rong_la_sai():
    assert cham_cau_tra_loi("  <|im_end|>", XANG["prompt"]) == TRA_LOI_RONG


def test_luot_2_ma_lai_goi_ham_tiep_la_sai():
    assert cham_cau_tra_loi("<nexa_0>()<nexa_end>", XANG["prompt"]) == TRA_LOI_RONG


def test_mau_reply_dung_nhom_rieng_khong_lan_vao_ro_rang():
    """Gộp lượt 2 vào 'rõ ràng' sẽ làm hỏng cả hai con số."""
    bc = cham_toan_bo([XANG], {XANG["prompt"]: XANG["completion"]})
    assert bc["theo_nhom"]["tra_loi"]["so_cau"] == 1
    assert "ro_rang" not in bc["theo_nhom"]


def test_mau_reply_khong_con_bi_tinh_la_khong_parse():
    """Lỗi đã gặp thật: 210/600 mẫu reply bị chấm là khong_parse vì chưa có luật riêng."""
    bc = cham_toan_bo([XANG], {XANG["prompt"]: XANG["completion"]})
    assert bc["ty_le_loi_%"]["khong_parse"] == 0.0
    assert bc["theo_nhom"]["tra_loi"]["dung_%"] == 100.0


def test_cau_sai_ghi_nhan_duoc_bia_so():
    ds = cau_sai([XANG], {XANG["prompt"]: "Xăng còn 60%."})
    assert len(ds) == 1 and ds[0]["kieu_sai"] == BIA_SO
