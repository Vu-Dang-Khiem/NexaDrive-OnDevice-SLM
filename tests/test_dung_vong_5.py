"""Cứu câu hỏi lại bị lọc cùng khuôn quá tay (PLAN.md mục U)."""

from gen.dung_vong_5 import cuu_cung_khuon


def _r(q, fn, loai):
    return {"query": q, "fn_dich": fn, "vai": "v", "bay": "b", "loai_vi": loai}


def test_chi_so_cung_ham_va_bo_cau_gan_trung_that():
    ct = [
        _r("Trời nóng quá, ngồi trong xe như cái lò nướng vậy", "set_ac", ""),
        # gần trùng câu đã giữ CÙNG hàm -> vẫn bỏ
        _r("Trời nóng quá, ngồi trong xe như cái lò hấp vậy", "set_ac", "cung_khuon"),
        # chung vài chữ với câu set_ac nhưng KHÁC hàm -> không so -> cứu
        _r("Trời nóng quá mà kính sau lại mờ hết, nhìn ra sau chẳng thấy gì", "set_defrost", "cung_khuon"),
        # cùng hàm nhưng chỉ chung vài chữ -> cứu
        _r("Đi cả buổi trưa nắng gắt, áo dính bết vào lưng khó chịu", "set_ac", "cung_khuon"),
    ]
    cuu = [r["query"] for r in cuu_cung_khuon(ct)]
    assert "Trời nóng quá, ngồi trong xe như cái lò hấp vậy" not in cuu
    assert len(cuu) == 2


def _ask(cau, calls):
    from gen.records import SuggestRecord

    return SuggestRecord(cau, tuple(calls)).to_row()


def test_sua_loi_he_thong_bat_dung_5_cau_nguoi_kiem_danh_sai():
    from gen.dung_vong_5 import sua_loi_he_thong

    def ly_do(cau, calls):
        return sua_loi_he_thong(_ask(cau, calls))[1]

    on = "Anh ơi, đài hôm nay hay ghê, nhưng mà tiếng ồn từ bên ngoài cũng to quá, nghe không rõ."
    assert ly_do(on, [("set_volume", {})]) == "on_ngoai_am_luong"  # dòng 29
    nang = "Nắng gắt quá, kính xe như cái lò sưởi, chói mắt quá."
    assert ly_do(nang, [("set_defrost", {"state": "on"})]) == "say_kinh_khong_mo"  # dòng 47
    den = "Chẹp, tối thế này mà đèn pha lại không sáng bằng mấy xe khác."
    assert ly_do(den, [("set_headlights", {"mode": "low_beam"})]) == "den_da_bat_yeu"  # dòng 49
    quat = "Anh ơi, gió điều hoà thổi vào chân em ồn quá, làm em không tập trung được."
    r, lo = sua_loi_he_thong(_ask(quat, [("set_fan_speed", {"level": 0})]))
    assert lo == "quat_tat_thanh_giam" and "level=2" in r["completion"]
    assert r["completion"].startswith("<nexa_ask>")


def test_sua_loi_he_thong_khong_dung_cau_dung():
    from gen.dung_vong_5 import sua_loi_he_thong

    giu = [
        # đèn PHỐ yếu -> gợi ý bật đèn xe là đúng
        ("Đèn đường yếu quá, chẳng thấy gì", [("set_headlights", {"mode": "low_beam"})]),
        ("Kính trước mờ hết, chẳng thấy đường", [("set_defrost", {"state": "on"})]),
        ("Nhạc to quá, con đang ngủ", [("set_volume", {})]),  # ồn TRONG xe -> âm lượng đúng
    ]
    for cau, calls in giu:
        r, lo = sua_loi_he_thong(_ask(cau, calls))
        assert lo == "" and r is not None, cau


def test_khong_cuu_cau_hong_that_o_buoc_code():
    ct = [_r("Đi meetings cả ngày mệt quá", "set_ac", "tieng_anh"), _r("Nóng ghê", "set_ac", "qua_ngan")]
    assert cuu_cung_khuon(ct) == []
