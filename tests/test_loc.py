"""Test bộ lọc chương trình — 2 lỗi THẬT tìm thấy trong dataset full2 (27.327 mẫu)."""

from gen.loc import cau_hop_le, co_chu_dong_a, lo_ten_ham, loc_rows, tham_so_hop_le
from gen.records import CallRecord, negative_record


# ── chữ Trung/Nhật/Hàn (119 mẫu bị lẫn trong dataset đầu tiên) ───────────────
def test_cau_tieng_viet_sach_thi_giu():
    assert co_chu_dong_a("Đóng cốp sau xe lại đi, cảm ơn.") is False
    assert cau_hop_le("Bật điều hoà 22 độ ghế lái")[0] is True


def test_bat_duoc_chu_trung_lan_trong_cau():
    """Mẫu thật: 'Đóng cốp sau xe lại đi,谢谢.'"""
    assert co_chu_dong_a("Đóng cốp sau xe lại đi,谢谢.") is True
    ok, ly_do = cau_hop_le("Dẫn tôi đến bệnh viện Từ Dũ, tránh cao速.")
    assert ok is False
    assert "Trung" in ly_do


def test_bat_duoc_chu_han_nhat():
    assert co_chu_dong_a("ドアを開けて") is True  # tiếng Nhật
    assert co_chu_dong_a("문 열어") is True  # tiếng Hàn


def test_bat_duoc_chu_trung_trong_tham_so():
    """Mẫu thật: navigate_to(destination="火車站")."""
    ok, ly_do = tham_so_hop_le({"destination": "火車站"})
    assert ok is False
    assert "destination" in ly_do


def test_tham_so_sach_va_tham_so_so_thi_qua():
    assert tham_so_hop_le({"destination": "Sân bay Nội Bài", "temperature": 22})[0] is True


# ── lộ tên hàm (target leakage — theo Jafarzadeh Asl et al. 2026 §A.7) ──────
def test_bat_duoc_cau_lo_ten_ham():
    assert lo_ten_ham("hãy gọi set_ac giúp tôi") == "set_ac"
    assert cau_hop_le("chạy open_trunk đi")[0] is False


def test_bat_duoc_functional_token_trong_cau():
    assert lo_ten_ham("sinh <nexa_6> cho tôi") == "<nexa_"


def test_khong_bat_nham_tu_tieng_viet_binh_thuong():
    """Người dùng nói 'bật điều hoà', 'mở cốp' là hợp lệ — không phải lộ nhãn."""
    for cau in ["Bật điều hoà lên", "Mở cốp xe giúp tôi", "Khoá cửa lại", "Hạ kính xuống"]:
        assert lo_ten_ham(cau) is None, cau
        assert cau_hop_le(cau)[0] is True, cau


# ── lọc cả dataset ──────────────────────────────────────────────────────────
def test_loc_rows_giu_mau_sach_bo_mau_ban():
    rows = [
        CallRecord("Bật điều hoà lên", "set_ac", {"state": "on"}).to_row(),
        CallRecord("Đóng cốp lại,谢谢", "close_trunk", {}).to_row(),
        CallRecord("Dẫn tôi tới ga tàu", "navigate_to", {"destination": "火車站"}).to_row(),
    ]
    sach, bo = loc_rows(rows)
    assert len(sach) == 1
    assert len(bo) == 2
    assert all("ly_do_loai" in r for r in bo)


def test_loc_rows_bat_chu_la_trong_nhan_khong_chi_trong_cau_hoi():
    """Câu hỏi sạch nhưng THAM SỐ bẩn -> vẫn phải loại (nhãn nằm ở completion)."""
    rows = [CallRecord("Dẫn tôi tới ga tàu", "navigate_to", {"destination": "火車站"}).to_row()]
    sach, bo = loc_rows(rows)
    assert not sach
    assert "nhãn" in bo[0]["ly_do_loai"]


# ── câu lai ngữ pháp tiếng Anh ──────────────────────────────────────────────
from gen.loc import lai_tieng_anh  # noqa: E402


def test_bat_duoc_cau_lai_tieng_anh_that():
    """Khuôn có thật trong dữ liệu vòng 1."""
    assert lai_tieng_anh("bạn có thể telling me about your favorite music?")


def test_khong_bat_tu_muon_nguoi_viet_hay_noi():
    for c in ("Bật radio FM 99.5", "Mở sunroof ra hết cỡ", "Tăng volume lên 10",
              "Kết nối Bluetooth đi", "Đi 195km nữa có đủ xăng không", "bật cruise control"):
        assert not lai_tieng_anh(c), c


def test_khong_bat_ten_sach_mot_hu_tu():
    assert not lai_tieng_anh("ai là tác giả của War and Peace?")
    assert not lai_tieng_anh("ai là người viết The Great Gatsby?")


def test_bo_qua_phan_trong_ngoac():
    assert not lai_tieng_anh("Phát bài 'You Are My Sunshine' đi")


def test_khong_nham_am_tiet_viet_voi_hu_tu_anh():
    """'an toàn', 'to', 'can' là tiếng Việt — cách tách từ cũ từng bắt nhầm hàng nghìn câu."""
    assert not lai_tieng_anh("Khoá cửa lại cho an toàn, nhạc to quá")


def test_loc_rows_mien_cho_ham_nhac():
    """Tên bài hát không có ngoặc vẫn phải giữ."""
    cau = "Cho tôi nghe bài You Are My Sunshine"
    r = CallRecord(cau, "play_media", {"query": "You Are My Sunshine"}).to_row()
    sach, _ = loc_rows([r])
    assert len(sach) == 1


def test_loc_rows_loai_cau_lai_o_ham_khac():
    r = negative_record("bạn có thể telling me about your hobbies?").to_row()
    sach, bo = loc_rows([r])
    assert not sach and bo[0]["ly_do_loai"] == "câu lai ngữ pháp tiếng Anh"


# ── câu cùng khuôn (câu từ chối kể hoàn cảnh, PLAN.md mục P) ─────────────────
def test_cau_cung_khuon_doi_vai_chu_chi_giu_mot():
    from gen.loc import bo_cung_khuon

    cau = [
        "Vừa đỗ xe xong, trời nắng quá, anh có thể kéo rèm che nắng cho tôi được không?",
        "Trời nắng gắt, tôi mới đỗ xe xong, anh có thể kéo rèm che nắng cho tôi được không?",
        "Mưa to quá, kính mờ hết, bật đèn nội thất lên hộ tớ đi.",
    ]
    assert bo_cung_khuon(cau) == [0, 2]


def test_cau_khac_khuon_giu_het():
    from gen.loc import bo_cung_khuon

    cau = [
        "Đường này lạ quá, bật camera 360 giúp tớ xem quanh xe được không?",
        "Nóng quá chảy mồ hôi, anh hạ nhiệt ghế giúp em đi.",
        "Tiếng còi inh ỏi quá, mua dùm em chai nước đi.",
    ]
    assert bo_cung_khuon(cau) == [0, 1, 2]


def test_cau_tu_choi_ghep_lenh_ham_that_bi_bat():
    from gen.loc import co_lenh_ham_that

    for c in ("Đóng cốp sau lại, dạo này hay quên đồ, đặt hộ tôi suất phở",
              "Khóa cửa xe kỹ lại, cậu book dùm tôi cái grab",
              "Khói bụi ngập xe, mở cửa sổ cho thông thoáng, tắt máy từ xa hông?"):
        assert co_lenh_ham_that(c), c


def test_do_vat_cho_khac_va_viec_khong_ho_tro_khong_bi_bat():
    from gen.loc import co_lenh_ham_that

    for c in ("Mệt muốn ngã, giúp tớ khoá cửa nhà đi",
              "Quên tắt điều hoà phòng ngủ rồi, tắt giúp mình",
              "Mưa to kính mờ, bật đèn nội thất lên hộ tớ",
              "Cửa xe đã khoá kỹ rồi, cho tui biết kết quả trận đấu",
              "Nắng quá, kéo rèm cửa sổ trời lên giúp tôi"):
        assert co_lenh_ham_that(c) is None, c


def test_cau_tu_choi_an_viec_lam_duoc_bi_bat():
    """Lần thử 3: hai câu "từ chối" lọt giám khảo nhưng thật ra xem thời tiết / tìm chỗ gần đây được."""
    from gen.loc import co_viec_lam_duoc_an

    for c in ("Nắng chang chang, anh cho tui biết nhiệt độ bao nhiêu độ ở ngoài trời?",
              "Lạnh quá, em muốn ăn bún bò, anh biết chỗ nào ngon gần đây không?",
              "Mai đi Đà Lạt, xem dự báo giúp mình"):
        assert co_viec_lam_duoc_an(c), c


def test_tu_choi_that_khong_bi_bat_nham():
    from gen.loc import co_viec_lam_duoc_an

    for c in ("Mưa to kính mờ quá, bật gạt nước nhanh hơn giúp mình",
              "Trời nóng quá, mồ hôi đầy trán, nhớ nhắc tui 10 giờ mai đi làm",
              "Lạnh căm căm, bật sưởi ghế cho em ấm lên đi"):
        assert co_viec_lam_duoc_an(c) is None, c


def test_thoi_tiet_o_phan_hoan_canh_la_bay_khong_bi_bo():
    """Bản đầu bỏ oan câu từ chối tốt chỉ vì phần HOÀN CẢNH nhắc thời tiết (đúng loại bẫy cố ý tạo)."""
    from gen.loc import co_viec_lam_duoc_an

    for c in ("Thời tiết hôm nay đẹp quá, nắng ấm, chuyển sang chế độ lái thể thao được không?",
              "Dạo này thời tiết lạnh quá, bật sưởi ghế cho em ấm lên đi."):
        assert co_viec_lam_duoc_an(c) is None, c


def test_gan_day_giua_cau_van_bi_bat():
    """Đọc mẫu sinh thật vòng 4: "gần đây" nằm giữa câu, lời nhờ thật ra là tìm quán cà phê."""
    from gen.loc import co_viec_lam_duoc_an

    cau = "Chỗ đỗ này rộng quá, Google Map nói gần đây có quán cà phê, em quên tên, anh có biết không?"
    assert co_viec_lam_duoc_an(cau)


def test_cau_goi_ham_phai_co_loi_nho():
    from gen.loc import co_loi_nho

    assert co_loi_nho("Trong xe hơi nóng mà sắp chở thêm người, bật điều hòa lên giúp mình nhé.")
    assert co_loi_nho("Sắp tới ngã rẽ mà chưa biết đường, dẫn mình tới Vincom Bắc Ninh đi.")
    assert co_loi_nho("Xe kêu lạ quá, lốp có non không?")
    assert not co_loi_nho("Nóng quá, mồ hôi ướt hết cả áo rồi")  # chỉ than -> là câu HỎI LẠI
