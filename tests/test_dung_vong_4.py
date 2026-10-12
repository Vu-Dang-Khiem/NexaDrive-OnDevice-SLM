"""Vòng 4: câu từ chối kể hoàn cảnh — chỉ lấy câu giám khảo giữ, không trùng test/dev."""

from gen.dung_vong_4 import TU_CHOI_HOAN_CANH, cau_moi_sach
from gen.records import negative_record


def _nguon(cau: str, giu: bool = True) -> dict:
    return {"query": cau, "giu": giu, "loai": "xe"}


def test_chi_lay_cau_giam_khao_giu_va_gan_nhom_phu():
    rows, bc = cau_moi_sach(
        [
            _nguon("Mưa to kính mờ quá, bật cái gạt nước nhanh hơn giúp mình"),
            _nguon("Nóng quá, bật điều hoà lên đi", giu=False),
        ],
        [],
    )
    assert len(rows) == 1 and bc["giam_khao_giu"] == 1
    assert rows[0]["meta"]["nhom_phu"] == TU_CHOI_HOAN_CANH and rows[0]["meta"]["fn"] == "irrelevant_function"


def test_cau_trung_test_hoac_dev_bi_bo():
    cau = "Đỗ ngoài nắng lâu quá ghế nóng ran, làm mát ghế lái giúp mình với"
    rows, bc = cau_moi_sach([_nguon(cau)], [negative_record(cau).to_row()])
    assert rows == [] and bc["nhiem_test_dev"] == 1


def test_cau_lai_tieng_anh_bi_loc():
    rows, bc = cau_moi_sach([_nguon("Can you please turn on the seat massage for me now")], [])
    assert rows == [] and bc["loc_code"] == 1


def test_cau_goi_ham_hoan_canh_thanh_mau_call_dung_ham():
    from gen.dung_vong_4 import GOI_HAM_HOAN_CANH, goi_ham_moi_sach

    nguon = [
        {
            "query": "Trời nóng mà sắp chở thêm người, bật điều hoà 22 độ giúp mình nhé",
            "fn": "set_ac",
            "args": {"state": "on", "temperature": 22},
            "giu": True,
        },
        {
            "query": "Nóng quá, mồ hôi ướt hết cả áo rồi đây này",
            "fn": "set_ac",  # chỉ than -> bỏ
            "args": {"state": "on"},
            "giu": True,
        },
    ]
    rows, bc = goi_ham_moi_sach(nguon, [])
    assert len(rows) == 1 and bc["giu_co_loi_nho"] == 1
    assert rows[0]["meta"]["fn"] == "set_ac" and rows[0]["meta"]["nhom_phu"] == GOI_HAM_HOAN_CANH
    assert rows[0]["completion"].startswith('<nexa_6>(state="on", temperature=22)')


def test_prompt_goi_ham_hoan_canh_doi_xung_voi_tu_choi():
    from gen.teacher_prompts import BAY_TU_KHOA, VAI_NGUOI_NOI, build_contextual_positive_prompt
    from schema.loader import CATALOG

    p = build_contextual_positive_prompt(3, CATALOG.by_name["set_ac"], VAI_NGUOI_NOI[0], BAY_TU_KHOA[0])
    assert VAI_NGUOI_NOI[0] in p and BAY_TU_KHOA[0] in p and "LÀM ĐƯỢC" in p and "Chỉ than thở" in p
    assert p.count('"') == p.count('"câu') * 2 + 2  # không câu ví dụ


def test_cau_goi_ham_loi_hinh_thuc_va_hoi_trang_thai_bi_bo_am_luong_theo_quy_uoc():
    """Ba lỗi đọc tay thấy trong 40 câu sinh thật."""
    from gen.dung_vong_4 import goi_ham_moi_sach

    def g(cau, fn, args):
        return {"query": cau, "fn": fn, "args": args, "giu": True}

    rows, bc = goi_ham_moi_sach(
        [
            g(
                "Trời nóng quá, mở nắp bình xăng giùm tui., Đi họp trễ rồi, mở nắp xăng nhanh lên",
                "open_fuel_flap",
                {},
            ),
            g("Khách vừa xuống, cốp điện đóng lại chưa vậy?", "close_trunk", {}),
            g("Ồn quá trời, anh giảm âm lượng xuống 20 được không?", "set_volume", {"delta": -20}),
        ],
        [],
    )
    assert bc["bo_dinh_cau_chu_la"] == 1 and bc["bo_hoi_trang_thai"] == 1
    assert len(rows) == 1 and rows[0]["completion"].startswith("<nexa_19>(set_value=20)")
