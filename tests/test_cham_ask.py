"""Test quy trình chấm mẫu hỏi lại mới (gen/cham_ask.py) — phần CODE, không cần GPU.

Canh đúng các lỗi đã thấy thật:
  - câu ra lệnh lọt vào nhãn hỏi lại ("Bật điều hoà lên 22 độ", "Chỉ cho tôi trạm đổ xăng")
  - bắt oan "chỉ" phó từ ("Chỉ cần ngồi trong xe là mệt") — 20 câu vòng 1
  - giám khảo gật đầu với mọi thứ: kappa phải lộ ra điều đó dù % đồng ý cao
"""

import pytest

from gen.cham_ask import (
    cham_mot,
    co_ra_lenh,
    cohen_kappa,
    doc_lua_chon,
    hieu_chinh,
    kiem_luat,
    quyet_dinh,
)
from gen.teacher_prompts import build_chon_de_xuat_prompt
from schema.loader import CATALOG

AC = [("set_ac", {"state": "on"})]


# ── câu ra lệnh ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "cau",
    ["Bật điều hoà lên 22 độ", "Khoá hết cửa lại cho tôi", "Chỉ cho tôi trạm đổ xăng nào quanh đây",
     "Làm ơn mở kính ra", "Giúp tôi tìm quán cà phê", "Hãy tắt nhạc đi", "Chỉ đường về nhà"],
)
def test_bat_duoc_cau_ra_lenh(cau):
    assert co_ra_lenh(cau)


@pytest.mark.parametrize(
    "cau",
    ["Nóng như cái lò trong này", "Chỉ cần ngồi trong xe là mình thấy mệt", "Chỉ thấy mờ mờ",
     "Chỉ muốn yên tĩnh một chút", "Bụng đói meo rồi", "Trời tối thui chẳng thấy đường đâu"],
)
def test_khong_bat_oan_cau_than_phien(cau):
    assert not co_ra_lenh(cau)


# ── lớp luật ────────────────────────────────────────────────────────────────
def test_luat_loai_ham_nhay_cam():
    ok, ly_do = kiem_luat("Có người đứng đợi ngoài kia", [("unlock_doors", {})])
    assert not ok and "nhạy cảm" in ly_do


def test_luat_loai_khi_chi_toan_ham_chi_doc():
    ok, _ = kiem_luat("Không biết xăng còn nhiều không nhỉ", [("get_fuel_level", {})])
    assert not ok


def test_luat_giu_cap_chi_doc_kem_hanh_dong():
    """Luật đã nới: phải có ít nhất 1 hàm KHÔNG chỉ đọc (vd kiểm xăng + tìm trạm xăng)."""
    de_xuat = [("get_fuel_level", {}), ("find_nearby", {"category": "gas_station"})]
    ok, _ = kiem_luat("Xe sắp hết xăng rồi", de_xuat)
    assert ok


def test_luat_loai_cau_ra_lenh_truoc_moi_thu_khac():
    ok, ly_do = kiem_luat("Bật điều hoà lên", AC)
    assert not ok and "ra lệnh" in ly_do


# ── đọc lựa chọn của giám khảo ──────────────────────────────────────────────
def test_doc_lua_chon_bo_ten_ham_bia():
    assert doc_lua_chon('{"ham": ["set_ac", "bat_quat_tran"], "ly_do": "x"}') == ["set_ac"]


def test_doc_lua_chon_toi_da_hai():
    ra = doc_lua_chon('{"ham": ["set_ac", "set_fan_speed", "control_window"]}')
    assert ra == ["set_ac", "set_fan_speed"]


def test_doc_lua_chon_chiu_duoc_chu_thua_quanh_json():
    assert doc_lua_chon('Đây: {"ham": "lock_doors"} xong') == ["lock_doors"]


def test_doc_lua_chon_khong_co_json_thi_bao_loi():
    with pytest.raises(ValueError):
        doc_lua_chon("tôi nghĩ là bật điều hoà")


# ── quyết định: cả hai giám khảo phải chọn hàm chính ────────────────────────
def test_giu_khi_ca_hai_cung_chon():
    assert quyet_dinh("lock_doors", [["lock_doors"], ["lock_doors", "set_hazard_lights"]])


def test_loai_khi_mot_ben_khong_chon():
    """Đúng ca thật: 'chỗ đỗ không an toàn' gán đèn khẩn cấp; một giám khảo chọn khoá cửa."""
    assert not quyet_dinh("set_hazard_lights", [["set_hazard_lights"], ["lock_doors"]])


def test_khong_co_giam_khao_nao_thi_loai():
    assert not quyet_dinh("set_ac", [])


def test_cham_mot_loai_voi_ly_do_doc_duoc():
    de_xuat = [("set_hazard_lights", {"state": "on"})]
    giu, ly_do = cham_mot("Chỗ này vắng quá", de_xuat, [["lock_doors"], ["lock_doors"]])
    assert not giu and "set_hazard_lights" in ly_do


# ── kappa ───────────────────────────────────────────────────────────────────
def test_kappa_hoan_hao_bang_1():
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0


def test_giam_khao_gat_dau_moi_thu_thi_kappa_bang_0_du_phan_tram_cao():
    """Đúng bài học Thakur 2024: 90% mẫu đúng -> gật đầu hết vẫn '90% đồng ý', kappa = 0."""
    dap_an = [True] * 9 + [False]
    assert cohen_kappa(dap_an, [True] * 10) == 0.0


def test_kappa_am_khi_nguoc_hoan_toan():
    assert cohen_kappa([True, False], [False, True]) < 0


def test_kappa_do_dai_khac_nhau_thi_bao_loi():
    with pytest.raises(ValueError):
        cohen_kappa([True], [True, False])


# ── hiệu chỉnh chạy đúng quy trình thật ─────────────────────────────────────
def test_hieu_chinh_bat_duoc_giam_khao_gat_dau():
    bo = [("Nóng quá", AC, True), ("Bụng đói meo", AC, False), ("Bật điều hoà lên", AC, False)]
    gat_dau = [{"Nóng quá": ["set_ac"], "Bụng đói meo": ["set_ac"], "Bật điều hoà lên": ["set_ac"]}]
    kq = hieu_chinh(bo, gat_dau)
    # câu ra lệnh vẫn bị code chặn, nhưng câu "đói -> điều hoà" lọt vì giám khảo chọn bừa
    assert kq["loai_dung"] == "1/2"
    assert kq["kappa"] < 1


def test_hieu_chinh_hoan_hao():
    bo = [("Nóng quá", AC, True), ("Bụng đói meo", AC, False)]
    tot = [{"Nóng quá": ["set_ac"], "Bụng đói meo": ["find_nearby"]}] * 2
    assert hieu_chinh(bo, tot)["kappa"] == 1.0


# ── prompt tự chọn ──────────────────────────────────────────────────────────
def test_prompt_tu_chon_khong_lo_nhan():
    """Cốt lõi của Zheng 2023: giám khảo KHÔNG được thấy đáp án."""
    p = build_chon_de_xuat_prompt("Nóng quá")
    phan_giua = p.lower().split("người trong xe vừa nói")[1].split("chỉ trả về")[0]
    assert "set_ac(state" not in phan_giua  # không có lời gọi mẫu nào để giám khảo "đồng ý theo"
    assert "valid" not in p


def test_prompt_tu_chon_co_du_24_ham():
    p = build_chon_de_xuat_prompt("Nóng quá")
    for f in CATALOG.functions:
        if f.name != "irrelevant_function":
            assert f"{f.name}(" in p


# ── luật bổ sung sau khi đo bằng 400 câu nhãn người ─────────────────────────
import pytest  # noqa: E402

from gen.cham_ask import co_tieu_tu_sai_bao, noi_ve_xe_khac  # noqa: E402


@pytest.mark.parametrize(
    "cau",
    [
        "Quạt gió mạnh hơn tí đi, tôi thấy hơi ngạt.",
        "Nhiên, mở quạt lên đi, tôi thấy khó chịu quá.",
        "Gió mát tí nữa đi",
        "Tạo cảm giác mát mẻ đi!",
    ],
)
def test_bat_cau_ra_lenh_co_tieu_tu_di_o_cuoi(cau):
    """Lọt thật qua luật cũ vì động từ không đứng đầu câu."""
    assert co_tieu_tu_sai_bao(cau)


@pytest.mark.parametrize(
    "cau",
    [
        "Đường này toàn cao tốc, làm sao đi?",
        "Chỉ có một mình, không biết đi đâu",
        "Trời tối quá, không biết đường đi",  # bắt oan thật ở bản đầu
        "Trời tối quá, không nhìn rõ lối đi",  # bắt oan thật ở bản đầu
        "Mệt quá sau chuyến đi, muốn nghỉ",
        "Thôi mình đi",
        "Đồ đã vào hết, có thể đi được rồi",
    ],
)
def test_khong_bat_oan_chu_di_la_dong_tu_di_chuyen(cau):
    assert not co_tieu_tu_sai_bao(cau)


@pytest.mark.parametrize(
    "cau",
    [
        "Xe phía sau cứ chớp đèn, có chuyện gì vậy?",
        "Hình như có xe bị tai nạn, đèn cảnh báo nháy liên tục.",
        "Đèn pha của xe phía sau sáng quá, chói mắt quá.",
        "Đúng là lúc trời tối mới thấy rõ đèn hậu của xe khác.",
    ],
)
def test_nhan_ra_cau_noi_ve_den_xe_khac(cau):
    assert noi_ve_xe_khac(cau)


@pytest.mark.parametrize(
    "cau",
    [
        "Đèn pha của xe mình yếu quá",
        "Nhìn mờ mờ, không biết có xe phía trước không",  # bắt oan thật ở bản đầu
        "Đèn xe phía trước yếu quá, nhìn không rõ",  # = đèn TRƯỚC của xe mình
        "Xe mình chết máy giữa đường, mấy xe đằng sau đang lao tới",  # tình huống đèn khẩn cấp THẬT
    ],
)
def test_khong_bat_oan_xe_minh(cau):
    assert not noi_ve_xe_khac(cau)


def test_luat_xe_khac_chi_ap_cho_ham_den():
    """Câu nhắc xe khác nhưng đề xuất hạ kính thì không liên quan luật này."""
    cau = "Xe phía trước gặp sự cố, khói bay mù mịt"
    assert not kiem_luat(cau, [("set_hazard_lights", {"state": "on"})])[0]
    assert kiem_luat(cau, [("control_window", {"window": "all", "position": "up"})])[0]
