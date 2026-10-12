"""Test phần dữ liệu vòng 2 — chạy local, không cần GPU.

Canh đúng ba thứ đã chẩn đoán ở vòng 1:
  1. Mẫu hỏi lại: `set_ac` chiếm 297/803 -> phải cắt được theo từng hàm đề xuất.
  2. Mẫu từ chối khó: phải được GIỮ HẾT khi bốc, không bị hoà loãng trong ~8.000 mẫu dễ.
  3. Prompt từ chối khó: phải đòi câu MỆNH LỆNH (đó là toàn bộ lý do nó tồn tại).
"""

import random

import pytest

from gen.records import CallRecord, SuggestRecord, negative_record
from gen.subsample import TU_CHOI_KHO, can_bang_ask, subsample
from gen.teacher_prompts import TINH_NANG_KHONG_HO_TRO, build_hard_negative_prompt


def _ask(fn: str, i: int):
    return SuggestRecord(f"câu than thở số {i} về {fn}", ((fn, {}),)).to_row()


def _kho(i: int):
    r = negative_record(f"Bật sưởi ghế lên số {i}").to_row()
    r["meta"]["nhom_phu"] = TU_CHOI_KHO
    return r


def _de(i: int):
    return negative_record(f"Ai là tác giả cuốn sách số {i}?").to_row()


# ── cân bằng mẫu hỏi lại ────────────────────────────────────────────────────
def test_cat_ham_bi_lech_xuong_dung_tran():
    asks = [_ask("set_ac", i) for i in range(300)] + [_ask("control_window", i) for i in range(30)]
    ra = can_bang_ask(asks, tran_moi_ham=60, rng=random.Random(0))
    dem = {fn: sum(1 for r in ra if r["meta"]["fn"] == fn) for fn in ("set_ac", "control_window")}
    assert dem == {"set_ac": 60, "control_window": 30}


def test_ham_duoi_tran_giu_nguyen():
    asks = [_ask("navigate_to", i) for i in range(5)]
    assert len(can_bang_ask(asks, tran_moi_ham=60, rng=random.Random(0))) == 5


def test_subsample_ap_tran_ask():
    rows = [_ask("set_ac", i) for i in range(300)] + [_ask("set_volume", i) for i in range(40)]
    con = subsample(rows, per_fn=300, tran_ask_moi_ham=50, so_negative=0)
    assert sum(1 for r in con if r["meta"]["fn"] == "set_ac") == 50
    assert sum(1 for r in con if r["meta"]["fn"] == "set_volume") == 40


def test_khong_dat_tran_thi_giu_hanh_vi_cu():
    """Không truyền tham số mới -> y hệt vòng 1, tái lập được."""
    rows = [_ask("set_ac", i) for i in range(300)]
    assert len(subsample(rows, per_fn=300, so_negative=0)) == 300


# ── giữ hết mẫu từ chối khó ─────────────────────────────────────────────────
def test_mau_tu_choi_kho_duoc_giu_het():
    duong = [CallRecord(f"Mở cốp xe lần {i}", "open_trunk", {}).to_row() for i in range(100)]
    am = [_kho(i) for i in range(40)] + [_de(i) for i in range(5000)]
    con = subsample(duong + am, per_fn=100)
    assert sum(1 for r in con if r["meta"].get("nhom_phu") == TU_CHOI_KHO) == 40


def test_tong_so_am_van_giu_ti_le_1_1():
    """Giữ hết mẫu khó nhưng KHÔNG được phình tổng số âm — phần dễ phải co lại."""
    duong = [CallRecord(f"Mở cốp xe lần {i}", "open_trunk", {}).to_row() for i in range(100)]
    am = [_kho(i) for i in range(40)] + [_de(i) for i in range(5000)]
    con = subsample(duong + am, per_fn=100)
    n_am = sum(1 for r in con if r["meta"]["fn"] == "irrelevant_function")
    assert n_am == 100


def test_mau_kho_nhieu_hon_quota_thi_cat_bot():
    duong = [CallRecord(f"Mở cốp xe lần {i}", "open_trunk", {}).to_row() for i in range(10)]
    con = subsample(duong + [_kho(i) for i in range(50)], per_fn=10)
    assert sum(1 for r in con if r["meta"]["fn"] == "irrelevant_function") == 10


# ── prompt từ chối khó ──────────────────────────────────────────────────────
def test_prompt_doi_cau_menh_lenh():
    p = build_hard_negative_prompt(20)
    assert "RA LỆNH" in p
    assert "động từ sai khiến" in p


def test_prompt_cam_viet_cau_ma_tac_tu_lam_duoc():
    """Lọt câu "bật điều hoà" vào nhãn từ chối là dạy model từ chối nhầm."""
    p = build_hard_negative_prompt(20)
    assert "TUYỆT ĐỐI KHÔNG" in p
    assert "điều hoà" in p  # có liệt kê việc làm được để teacher né


def test_prompt_cam_cau_hoi_lich_su():
    assert "bạn có thể giúp tôi" in build_hard_negative_prompt(20)


def test_mac_dinh_liet_ke_du_moi_nhom_tinh_nang():
    p = build_hard_negative_prompt(20)
    assert all(t in p for t in TINH_NANG_KHONG_HO_TRO)


def test_chon_mot_nhom_thi_chi_co_nhom_do():
    nhom = TINH_NANG_KHONG_HO_TRO[0]
    p = build_hard_negative_prompt(20, nhom=nhom)
    assert nhom in p
    assert TINH_NANG_KHONG_HO_TRO[1] not in p


def test_danh_sach_tinh_nang_khong_trung_hanh_dong_minh_ho_tro():
    """Nhóm 'không hỗ trợ' mà trùng HÀNH ĐỘNG mình làm được thì dữ liệu tự mâu thuẫn.

    So với hành động chứ không so với danh từ: "hướng gió điều hoà", "rèm cửa sổ trời",
    "cốp trước" CỐ Ý nằm sát nách một hàm có thật — đó chính là câu từ chối khó cần dạy
    (model phải học "chỉnh hướng gió" ≠ "bật điều hoà"). Chỉ cấm trùng đúng việc mình làm.
    """
    lam_duoc = (
        "bật điều hoà", "tốc độ quạt", "sấy kính", "khoá cửa", "mở cốp sau", "nắp sạc",
        "nắp bình xăng", "hạ kính", "mở cửa sổ trời", "đèn pha", "đèn khẩn cấp",
        "âm lượng", "phát nhạc", "thời tiết",
    )
    for t in TINH_NANG_KHONG_HO_TRO:
        assert not any(c in t.lower() for c in lam_duoc), t


# ── giám khảo từ chối (sửa sau chạy thử: loại oan 6/6 câu sát nách hàm thật) ─
from pathlib import Path  # noqa: E402

from gen.teacher_prompts import (  # noqa: E402
    BO_HIEU_CHINH_TU_CHOI,
    VI_DU_GIAM_KHAO_TU_CHOI,
    build_verify_negative_prompt,
)
from schema.loader import CATALOG  # noqa: E402


def test_giam_khao_thay_chu_ky_chinh_xac_cua_moi_ham():
    """Bản cũ chỉ liệt kê nhóm ("điều hoà", "cốp") nên teacher tưởng "lấy gió trong" làm được."""
    p = build_verify_negative_prompt("Bật sưởi ghế")
    for f in CATALOG.functions:
        if f.name != "irrelevant_function":
            assert f"{f.name}(" in p, f"prompt chấm thiếu chữ ký {f.name}"


def test_giam_khao_thay_gia_tri_enum():
    """Có enum thì teacher mới thấy open_trunk không có lựa chọn 'cốp trước'."""
    p = build_verify_negative_prompt("x")
    assert "front_left" in p  # enum của control_window
    assert "low_beam" in p  # enum của set_headlights


def test_giam_khao_co_luat_cung_chu_de_chua_du():
    assert "CÙNG CHỦ ĐỀ CHƯA ĐỦ" in build_verify_negative_prompt("x")


def test_vi_du_mau_co_du_hai_chieu():
    """Chỉ có ví dụ 'valid = true' thì giám khảo học thành gật đầu với mọi thứ."""
    p = build_verify_negative_prompt("x")
    assert p.count("-> valid = true") >= 3
    assert p.count("-> valid = false") >= 3


def test_bo_hieu_chinh_can_bang_hai_chieu():
    dung = sum(1 for _, v in BO_HIEU_CHINH_TU_CHOI if v)
    assert dung == len(BO_HIEU_CHINH_TU_CHOI) - dung >= 8


def test_bo_hieu_chinh_khong_trung_cau_nao():
    cau = [c.lower() for c, _ in BO_HIEU_CHINH_TU_CHOI]
    assert len(set(cau)) == len(cau)


def test_vi_du_trong_prompt_khop_voi_danh_sach():
    p = build_verify_negative_prompt("x")
    assert all(c in p for c, _, _ in VI_DU_GIAM_KHAO_TU_CHOI)


def test_cau_hieu_chinh_va_vi_du_khong_lay_tu_tap_test():
    """Dùng câu test để chỉnh giám khảo thì tập test hết còn là dữ liệu chưa từng thấy.

    Test này đã bắt được thật: "Mở cốp sau ra" (trùng y hệt) và "Giảm âm lượng xuống 3 nấc"
    (gần trùng) từng nằm trong bộ hiệu chỉnh và ví dụ của prompt.
    """
    p = Path("data/processed/final/test_vi.jsonl")
    if not p.exists():
        import pytest

        pytest.skip("không có tập test ở máy này")
    import json

    from eval.kiem_nhiem import tim_nhiem

    test = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    cau = [c for c, _ in BO_HIEU_CHINH_TU_CHOI] + [c for c, _, _ in VI_DU_GIAM_KHAO_TU_CHOI]
    kq = tim_nhiem([negative_record(c).to_row() for c in cau], test)
    assert kq["so_bi_nhiem"] == 0, kq["vi_du"]


# ── giám khảo HỎI LẠI (sửa vì giữ 517/518, kể cả "chỗ đỗ không an toàn" -> đèn khẩn cấp) ─
from gen.teacher_prompts import BO_HIEU_CHINH_HOI_LAI, build_verify_ask_prompt  # noqa: E402


def test_giam_khao_hoi_lai_thay_ca_danh_sach_ham():
    """Chỉ thấy một hàm thì hàm nào cũng 'giúp được chút' -> gật đầu hết."""
    p = build_verify_ask_prompt("Nóng quá", [("set_ac", {"state": "on"})])
    assert "lock_doors(" in p and "control_window(" in p


def test_giam_khao_hoi_lai_hoi_hop_ly_nhat():
    p = build_verify_ask_prompt("Nóng quá", [("set_ac", {"state": "on"})])
    assert "HỢP LÝ NHẤT" in p
    assert "CHƯA ĐỦ" in p


def test_bo_hieu_chinh_hoi_lai_du_hai_chieu():
    giu = sum(1 for *_, v in BO_HIEU_CHINH_HOI_LAI if v)
    assert giu >= 6 and len(BO_HIEU_CHINH_HOI_LAI) - giu >= 6


def test_bo_hieu_chinh_hoi_lai_co_ca_loai_ra_lenh_thang():
    """Giám khảo phải loại được cả câu ra lệnh thẳng bị gán nhầm là hỏi lại."""
    assert any(c.startswith(("Bật", "Khoá", "Tìm")) and not v for c, _, v in BO_HIEU_CHINH_HOI_LAI)


def test_de_xuat_trong_bo_hieu_chinh_deu_hop_le_voi_schema():
    """Đề xuất sai schema thì giám khảo loại vì lý do khác -> số hiệu chỉnh vô nghĩa."""
    for cau, de_xuat, _ in BO_HIEU_CHINH_HOI_LAI:
        SuggestRecord(cau, tuple(de_xuat)).to_row()  # ném lỗi nếu tham số lạ
        for ten, args in de_xuat:
            spec = {p.name: p for p in CATALOG.by_name[ten].params}
            for k, v in args.items():
                if spec[k].enum:
                    assert v in spec[k].enum, f"{ten}.{k}={v!r}"


def test_bo_hieu_chinh_hoi_lai_khong_lay_tu_tap_test():
    p = Path("data/processed/final/test_vi.jsonl")
    if not p.exists():
        import pytest

        pytest.skip("không có tập test ở máy này")
    import json

    from eval.kiem_nhiem import tim_nhiem

    test = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    cau = sorted({c for c, _, _ in BO_HIEU_CHINH_HOI_LAI})
    kq = tim_nhiem([negative_record(c).to_row() for c in cau], test)
    assert kq["so_bi_nhiem"] == 0, kq["vi_du"]


def test_mo_ta_den_khan_cap_noi_ro_khi_nao_dung():
    """Mô tả cũ 'đèn cảnh báo nguy hiểm' khớp chữ với 'không an toàn' -> gán nhầm."""
    mo_ta = CATALOG.by_name["set_hazard_lights"].description
    assert "trên đường" in mo_ta and "chống trộm" in mo_ta


# ── dựng dữ liệu vòng 2 ─────────────────────────────────────────────────────
def _duong(i: int, uu: bool = False):
    args = {"window": "front_right", "position": "down"}
    r = CallRecord(f"Hạ kính bên phụ số {i}", "control_window", args).to_row()
    if uu:
        r["meta"]["uu_tien"] = True
    return r


def test_prompt_tu_choi_hoan_canh_du_thanh_phan_va_khong_co_cau_vi_du():
    from gen.teacher_prompts import (
        BAY_TU_KHOA,
        TINH_NANG_KHONG_HO_TRO,
        VAI_NGUOI_NOI,
        build_contextual_negative_prompt,
    )

    p = build_contextual_negative_prompt(8, TINH_NANG_KHONG_HO_TRO[0], VAI_NGUOI_NOI[0], BAY_TU_KHOA[0])
    assert TINH_NANG_KHONG_HO_TRO[0] in p and VAI_NGUOI_NOI[0] in p and BAY_TU_KHOA[0] in p
    assert "KHÔNG làm được" in p and "Chỉ than thở" in p
    # không câu ví dụ trong ngoặc kép (ngoài khuôn JSON) -> teacher khỏi chép khuôn
    assert p.count('"') == p.count('"câu') * 2 + 2


def test_viec_ngoai_xe_va_vai_khong_trung_tap_test_va_dev():
    """Danh sách mồi của prompt mới không được lấy từ câu test/dev."""
    import json
    from pathlib import Path

    from eval.kiem_nhiem import tim_nhiem
    from gen.records import negative_record
    from gen.teacher_prompts import BAY_TU_KHOA, VAI_NGUOI_NOI, VIEC_NGOAI_XE

    nguon = []
    for p in ("data/processed/final/test_vi.jsonl", "data/dev_tay_2/dev_vi_2.jsonl"):
        if Path(p).exists():
            nguon += [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]
    if not nguon:
        pytest.skip("không có tập test/dev ở máy này")
    moi = [negative_record(x).to_row() for x in VIEC_NGOAI_XE + VAI_NGUOI_NOI + BAY_TU_KHOA]
    assert tim_nhiem(moi, nguon)["so_bi_nhiem"] == 0


def test_mau_uu_tien_duoc_giu_het_khi_boc():
    """Không ưu tiên thì chỉ ~1/9 câu vòng 2 lọt vào -> sửa khoá từ vựng mất tác dụng."""
    rows = [_duong(i) for i in range(1000)] + [_duong(i, True) for i in range(1000, 1120)]
    con = subsample(rows, per_fn=300, so_negative=0)
    assert sum(1 for r in con if r["meta"].get("uu_tien")) == 120
    assert len(con) == 300


def test_khong_danh_dau_uu_tien_thi_giong_het_vong_1():
    """Tái lập được vòng 1: cùng seed, không có mẫu ưu tiên -> kết quả y hệt cách bốc cũ."""
    rows = [_duong(i) for i in range(1000)]
    a = [r["prompt"] for r in subsample(rows, per_fn=300, so_negative=0, seed=42)]
    b = [r["prompt"] for r in random.Random(42).sample(rows, 300)]
    assert sorted(a) == sorted(b)


def test_bo_cau_on_ben_ngoai_gan_nham_am_luong():
    from gen.dung_vong_2 import bo_on_ben_ngoai

    sai = SuggestRecord("Điếc tai với tiếng ồn bên ngoài", (("set_volume", {}),)).to_row()
    giu = SuggestRecord("Điếc cả tai vì tiếng ồn này", (("set_volume", {}),)).to_row()
    ra, bo = bo_on_ben_ngoai([sai, giu])
    assert bo == 1 and ra == [giu]


# ── prompt sinh câu mơ hồ có hướng dẫn tình huống (vòng 3) ──────────────────
from gen.teacher_prompts import HUONG_DAN_NGAM, build_implicit_query_prompt  # noqa: E402


@pytest.mark.parametrize("ten", sorted(HUONG_DAN_NGAM))
def test_huong_dan_tinh_huong_co_trong_prompt(ten):
    p = build_implicit_query_prompt(CATALOG.by_name[ten], 50)
    assert "ĐÚNG TÌNH HUỐNG" in p and HUONG_DAN_NGAM[ten] in p


def test_ham_khong_co_huong_dan_thi_prompt_nhu_cu():
    assert "ĐÚNG TÌNH HUỐNG" not in build_implicit_query_prompt(CATALOG.by_name["set_ac"], 50)


def test_huong_dan_khong_chua_cau_vi_du_trong_ngoac_kep():
    """Người viết đã đọc 28 câu test -> không được đưa câu ví dụ cụ thể nào vào hướng dẫn."""
    for ten, hd in HUONG_DAN_NGAM.items():
        assert '"' not in hd and "“" not in hd, ten


def test_huong_dan_khong_trung_tap_test():
    p = Path("data/processed/final/test_vi_bo_sung.jsonl")
    if not p.exists():
        pytest.skip("không có tập test ở máy này")
    import json

    from eval.kiem_nhiem import tim_nhiem

    test = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    tach = (hd.replace("\n", ". ").split(".") for hd in HUONG_DAN_NGAM.values())
    cau = [c.strip() for ds in tach for c in ds if c.strip()]
    assert tim_nhiem([negative_record(c).to_row() for c in cau], test)["so_bi_nhiem"] == 0
