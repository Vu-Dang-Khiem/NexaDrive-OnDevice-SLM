"""Cặp đối chứng thứ ba — câu HỎI LẠI kể hoàn cảnh (PLAN.md mục U)."""

from gen.teacher_prompts import BAY_TU_KHOA, VAI_NGUOI_NOI, build_contextual_ask_prompt
from schema.loader import CATALOG


def test_prompt_hoi_lai_cung_khung_voi_hai_phia_kia_va_khong_cau_vi_du():
    p = build_contextual_ask_prompt(3, CATALOG.by_name["set_ac"], VAI_NGUOI_NOI[0], BAY_TU_KHOA[0])
    assert VAI_NGUOI_NOI[0] in p and BAY_TU_KHOA[0] in p
    assert "KHÔNG nhờ" in p and "HỎI LẠI" in p
    fn = CATALOG.by_name["set_ac"]
    assert fn.description.strip() in p
    # không câu ví dụ nào (khác prompt câu ngầm hiểu cũ có đưa implicit_examples) -> teacher khỏi chép khuôn
    assert fn.implicit_examples and not any(vd in p for vd in fn.implicit_examples)


def test_prompt_hoi_lai_kem_huong_dan_dung_tinh_huong_khi_co():
    """Đèn khẩn cấp: phải là xe MÌNH gặp sự cố — HUONG_DAN_NGAM đưa vào prompt."""
    from gen.teacher_prompts import HUONG_DAN_NGAM

    fn = CATALOG.by_name["set_hazard_lights"]
    p = build_contextual_ask_prompt(3, fn, VAI_NGUOI_NOI[1], BAY_TU_KHOA[1])
    if fn.name in HUONG_DAN_NGAM:
        assert HUONG_DAN_NGAM[fn.name] in p


def test_bay_theo_ham_du_moi_ham_hoi_lai_va_khong_mau_thuan_huong_dan():
    """Chạy thử 2: cửa sổ trời + bẫy 'nóng bức' -> 0/10 câu khớp. Hàm có hướng dẫn 'KHÔNG than nóng
    lạnh' thì không được nhận bẫy nóng lạnh."""
    from gen.teacher_prompts import BAY_THEO_HAM, BAY_TU_KHOA, HUONG_DAN_NGAM

    ham_hoi_lai = {f.name for f in CATALOG.functions if f.implicit_examples}
    assert set(BAY_THEO_HAM) == ham_hoi_lai and all(BAY_THEO_HAM.values())
    for ten, bay in BAY_THEO_HAM.items():
        if "KHÔNG than nóng lạnh" in HUONG_DAN_NGAM.get(ten, ""):
            assert BAY_TU_KHOA[0] not in bay, ten


def test_loc_tieng_anh_bat_tu_lot_o_lan_chay_thu_nhung_tha_am_tiet_viet_khong_dau():
    from gen.loc import tu_tieng_anh

    assert tu_tieng_anh("Đi meetings từ sáng tới giờ, xe lạnh buốt") == ["meetings"]
    assert tu_tieng_anh("lỡ Forget water vô xe") == ["Forget", "water"]
    assert tu_tieng_anh("hôm nay mood nó cứ ủ rũ") == ["mood"]
    # chữ không dấu nhưng là tiếng Việt / tên riêng Việt / từ mượn quen -> không bắt
    assert tu_tieng_anh("Con sao quanh khoang nghe giai thoang tuy Quang Long Hai Sen") == []
    assert tu_tieng_anh("Mở Bluetooth, rút tiền ở ATM, đi 20 km tới gara") == []


def test_loc_ra_lenh_bat_cau_nho_thang_nhung_tha_cau_ke():
    from gen.cham_ask import co_ra_lenh, co_tieu_tu_sai_bao

    assert co_ra_lenh("Bật điều hoà lên giùm, trong xe nóng như lò")
    assert co_tieu_tu_sai_bao("Trong xe nóng quá, mở điều hoà đi")
    ke = "Chiều nay nắng gắt, xe đỗ ngoài bãi cả buổi nên vừa ngồi vào đã thấy hầm hập"
    assert not co_ra_lenh(ke) and not co_tieu_tu_sai_bao(ke)
