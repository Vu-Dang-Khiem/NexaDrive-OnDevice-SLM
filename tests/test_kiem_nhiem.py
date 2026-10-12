"""Test kiểm nhiễm chéo test ↔ train.

Lỗi thật đã gặp: 43/296 câu test (14,5%) đã xuất hiện trong train/val, làm nhóm "rõ ràng"
đội từ 65,5% lên 71,2%. Lệnh xe rất ngắn và ai cũng nói giống nhau ("còn bao nhiêu xăng"),
nên người viết tay và teacher sinh ra trùng câu là chuyện bình thường — phải kiểm bằng máy.
"""

from eval.kiem_nhiem import tach_sach, tim_nhiem
from gen.records import CallRecord, negative_record


def _r(cau: str, fn: str = "open_trunk"):
    return CallRecord(cau, fn, {}).to_row()


def test_bat_duoc_cau_trung_y_het():
    test = [_r("Mở cốp xe"), _r("Xem giúp áp suất lốp", "get_tire_pressure")]
    kq = tim_nhiem(test, [_r("Mở cốp xe")])
    assert kq["so_bi_nhiem"] == 1
    assert kq["kieu"][0] == "trung_y_het"


def test_khac_dau_cau_van_tinh_la_trung():
    """"Mở cốp xe." và "Mở cốp xe" là cùng một câu."""
    kq = tim_nhiem([_r("Mở cốp xe.")], [_r("Mở cốp xe")])
    assert kq["so_bi_nhiem"] == 1


def test_khac_hoa_thuong_van_tinh_la_trung():
    kq = tim_nhiem([_r("MỞ CỐP XE")], [_r("mở cốp xe")])
    assert kq["so_bi_nhiem"] == 1


def test_cau_khac_han_thi_khong_bao_nhiem():
    test = [_r("Bật đèn khẩn cấp lên ngay", "set_hazard_lights")]
    nguon = [_r("Tìm quán cà phê gần đây nhất", "find_nearby")]
    assert tim_nhiem(test, nguon)["so_bi_nhiem"] == 0


def test_bat_duoc_cau_gan_trung():
    kq = tim_nhiem([_r("Mở cốp sau xe giúp tôi")], [_r("Mở cốp sau xe giúp tôi với")])
    assert kq["so_bi_nhiem"] == 1


def test_tat_duoc_phan_gan_trung():
    """Muốn báo cáo chặt chẽ hơn thì chỉ tính câu trùng tuyệt đối."""
    test = [_r("Mở cốp sau xe giúp tôi")]
    nguon = [_r("Mở cốp sau xe giúp tôi với")]
    assert tim_nhiem(test, nguon, gan_trung=False)["so_bi_nhiem"] == 0


def test_tach_sach_giu_nguyen_thu_tu_va_du_so_luong():
    test = [_r("Mở cốp xe"), _r("Đóng cốp lại", "close_trunk"), _r("Khoá xe", "lock_doors")]
    kq = tim_nhiem(test, [_r("Mở cốp xe")])
    sach, da_thay = tach_sach(test, kq)
    assert len(sach) == 2 and len(da_thay) == 1
    assert sach[0]["prompt"] == test[1]["prompt"]


def test_nguon_rong_thi_khong_nhiem():
    assert tim_nhiem([_r("Mở cốp xe")], [])["so_bi_nhiem"] == 0


def test_mau_negative_cung_kiem_duoc():
    neg = negative_record("Kể chuyện cười đi").to_row()
    assert tim_nhiem([neg], [neg])["so_bi_nhiem"] == 1


def test_bao_cao_co_ty_le_phan_tram():
    test = [_r("Mở cốp xe"), _r("Đóng cốp lại", "close_trunk")]
    kq = tim_nhiem(test, [_r("Mở cốp xe")])
    assert kq["ty_le_%"] == 50.0


def test_tap_dev_khong_trung_tap_test():
    """Tập dev dùng để thử/chỉnh nhiều lần -> câu trùng test thì chỉnh theo dev cũng là chỉnh theo test."""
    import json
    from pathlib import Path

    import pytest

    dev_p = Path("data/dev_tay_2/dev_vi_2.jsonl")
    tests = [Path("data/processed/final/test_vi.jsonl"), Path("data/processed/final/test_vi_bo_sung.jsonl")]
    if not dev_p.exists() or not all(p.exists() for p in tests):
        pytest.skip("thiếu tập dev hoặc tập test ở máy này")

    def doc(p):
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

    kq = tim_nhiem(doc(dev_p), [r for p in tests for r in doc(p)])
    assert kq["so_bi_nhiem"] == 0, kq["vi_du"]


def test_vi_du_moi_trong_schema_khong_trung_tap_test():
    """Câu mồi đưa thẳng cho teacher -> trùng câu test thì teacher sinh câu na ná câu test.

    Đã xảy ra thật: 20/107 câu mồi trùng tập test, trong đó 2 câu thêm ở vòng 2 được viết
    từ trí nhớ về câu test đã xem lúc mổ xẻ lỗi. Mọi câu mồi mới đều phải qua test này.
    """
    import json
    from pathlib import Path

    import pytest

    from schema.loader import CATALOG

    p = Path("data/processed/final/test_vi.jsonl")
    if not p.exists():
        pytest.skip("không có tập test ở máy này")
    test = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    moi = [e for f in CATALOG.functions for e in list(f.examples) + list(f.implicit_examples)]
    kq = tim_nhiem([negative_record(e).to_row() for e in moi], test)
    assert kq["so_bi_nhiem"] == 0, kq["vi_du"]
