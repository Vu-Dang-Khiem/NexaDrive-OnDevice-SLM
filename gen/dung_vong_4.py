"""Dựng dữ liệu VÒNG 4 — câu từ chối KỂ HOÀN CẢNH (PLAN.md mục P). Chạy ở máy, miễn phí, tái lập được.

    python -m gen.dung_vong_4

Nền: train/val vòng 3 ĐÃ DỰNG (`final_v3/`) — không dựng lại, nên mẫu và thứ tự y hệt v3. Hai nhánh, mỗi nhánh
đổi đúng một thứ so với v3:

  A — THAY  : bỏ các câu từ chối khó cũ (mệnh lệnh ngắn, `nhom_phu = tu_choi_kho`) khỏi train, thêm câu từ
              chối kể hoàn cảnh — bằng số câu cũ nếu đủ, không đủ thì dùng HẾT câu sạch (lệch vài chục trên
              16.010 mẫu, ghi trong báo cáo) -> tổng số mẫu và tỉ lệ dương:âm gần như giữ nguyên.
  B — THÊM  : giữ câu cũ, thêm CÙNG bộ câu mới đó -> trả lời "câu ngắn cũ có quan trọng không".

Lọc lại câu mới bằng bộ lọc MỚI NHẤT (`co_lenh_ham_that`, `co_viec_lam_duoc_an`) — lúc sinh trên Modal dùng
bản lọc cũ hơn.

Val GIỮ NGUYÊN như v3 ở cả hai nhánh để chọn checkpoint trên cùng một bộ. Câu mới qua: giám khảo đã giữ,
lọc code (`loc_rows`), kiểm nhiễm với tập test VÀ tập dev (dev dùng để đo đúng điểm yếu này, câu train trùng
dev thì dev hết còn sạch).
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from typing import Any

from eval.kiem_nhiem import tim_nhiem
from gen.dung_vong_2 import SEED, P, doc, ghi
from gen.loc import co_lenh_ham_that, co_loi_nho, co_viec_lam_duoc_an, loc_rows
from gen.records import CallRecord, negative_record
from gen.sua_nhan import sua_rows
from gen.subsample import TU_CHOI_KHO
from schema.loader import CATALOG

_DINH_NHIEU_CAU = re.compile(r"[.!?],\s")  # "…giùm tui., Đi họp…" — teacher dồn nhiều câu vào một chuỗi
_CHU_LA = re.compile(r"[Ѐ-ӿͰ-Ͽ]")  # Cyrillic, Hy Lạp lẫn vào câu
_HOI_TRANG_THAI = re.compile(r"\b(chưa|chưa vậy|chưa nhỉ|chưa ta|rồi chưa)\s*\?*\s*$", re.IGNORECASE)

NEN = P / "final_v3"
NGUON_MOI = P / "v4" / "v4_tu_choi_hc.jsonl"  # tải từ Modal: data/vi/v4_tu_choi_hc.jsonl
TAP_CHAN = [P / "final/test_vi_sach_v2.jsonl", P / "final/test_vi_bo_sung.jsonl", P / "final/test_vi.jsonl"]
DEV = P.parent / "dev_tay_2" / "dev_vi_2.jsonl"
TU_CHOI_HOAN_CANH = "tu_choi_hoan_canh"
GOI_HAM_HOAN_CANH = "goi_ham_hoan_canh"
NGUON_GOI_HAM = P / "v4" / "v4_goi_ham_hc.jsonl"  # tải từ Modal: data/vi/v4_goi_ham_hc.jsonl


def goi_ham_moi_sach(
    nguon: list[dict[str, Any]], chan: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict]:
    """Câu GỌI HÀM kể hoàn cảnh -> mẫu call: giám khảo giữ, có lời nhờ, qua lọc code, không trùng test/dev.

    Kiểm nhiễm với dev đặc biệt quan trọng: nhóm D của dev chính là câu làm được kể hoàn cảnh.

    Ba bước thêm sau khi đọc tay 40 câu sinh thật (PLAN.md mục P):
      - bỏ câu dính nhiều câu một chỗ ("…giùm tui., Đi họp trễ rồi, …") và câu lẫn chữ không phải Latin;
      - bỏ câu HỎI TRẠNG THÁI gán hàm HÀNH ĐỘNG ("cốp điện đóng lại chưa vậy?" -> close_trunk; đúng ra là
        xem tình trạng xe);
      - chạy bộ sửa nhãn vòng 3 (`sua_rows`): quy ước âm lượng ("giảm âm lượng xuống 20" -> set_value=20),
        xăng/tầm xa, chiều kính — câu mới phải theo CÙNG quy ước với phần dữ liệu còn lại.
    """
    giu = [r for r in nguon if r.get("giu") and co_loi_nho(r["query"])]
    rows, bo_hinh_thuc, bo_hoi_trang_thai = [], 0, 0
    for r in giu:
        cau = r["query"]
        if _DINH_NHIEU_CAU.search(cau) or _CHU_LA.search(cau):
            bo_hinh_thuc += 1
            continue
        if not CATALOG.by_name[r["fn"]].chi_doc and _HOI_TRANG_THAI.search(cau):
            bo_hoi_trang_thai += 1
            continue
        row = CallRecord(cau, r["fn"], r["args"]).to_row()
        row["meta"].update({"nhom_phu": GOI_HAM_HOAN_CANH, "vong": 4})
        rows.append(row)
    rows, bi_loc = loc_rows(rows)
    rows, bc_sua = sua_rows(rows)
    nhiem = set(tim_nhiem(rows, chan)["chi_so"])
    sach = [r for i, r in enumerate(rows) if i not in nhiem]
    return sach, {
        "giu_co_loi_nho": len(giu),
        "bo_dinh_cau_chu_la": bo_hinh_thuc,
        "bo_hoi_trang_thai": bo_hoi_trang_thai,
        "loc_code": len(bi_loc),
        "sua_nhan": bc_sua["dem"],
        "nhiem_test_dev": len(nhiem),
        "sach": len(sach),
    }


def dung_doi_chung() -> dict[str, Any]:
    """Nhánh ĐỐI CHỨNG: nhánh A (thay câu từ chối ngắn bằng câu từ chối kể hoàn cảnh) + CÙNG SỐ câu gọi hàm kể
    hoàn cảnh -> kiểu câu kể hoàn cảnh có ở CẢ HAI phía nhãn, cân bằng 1:1 (PLAN.md mục P, kết quả vòng 4)."""
    chan: list[dict[str, Any]] = []
    for p in TAP_CHAN + [DEV]:
        if p.exists():
            chan += doc(p)
    nen = doc(P / "final_v4_thay" / "train_vi.jsonl")
    so_tu_choi = sum(r["meta"].get("nhom_phu") == TU_CHOI_HOAN_CANH for r in nen)
    goi_ham, bc = goi_ham_moi_sach(doc(NGUON_GOI_HAM), chan)
    chon = random.Random(SEED).sample(goi_ham, min(len(goi_ham), so_tu_choi))
    rows = nen + chon
    random.Random(SEED).shuffle(rows)
    ra = P / "final_v4_doi_chung"
    ghi(ra / "train_vi.jsonl", rows)
    for tap in ("val_vi", "val_mau"):
        ghi(ra / f"{tap}.jsonl", doc(NEN / f"{tap}.jsonl"))  # val y hệt v3
    bao_cao = {
        "cau_goi_ham_moi": bc,
        "tu_choi_hoan_canh": so_tu_choi,
        "goi_ham_hoan_canh_dung": len(chon),
        "theo_ham": dict(Counter(r["meta"]["fn"] for r in chon)),
        "train": len(rows),
        "theo_loai": dict(Counter(r["meta"]["type"] for r in rows)),
    }
    (ra / "bao_cao_dung.json").write_text(json.dumps(bao_cao, ensure_ascii=False, indent=2), encoding="utf-8")
    return bao_cao


def cau_moi_sach(
    nguon: list[dict[str, Any]], chan: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict]:
    """Câu mới -> mẫu từ chối: chỉ câu giám khảo GIỮ, qua lọc code, không trùng test/dev."""
    giu = [r for r in nguon if r.get("giu")]
    rows = []
    lam_duoc = 0
    for r in giu:
        if co_lenh_ham_that(r["query"]) or co_viec_lam_duoc_an(r["query"]):
            lam_duoc += 1
            continue
        row = negative_record(r["query"]).to_row()
        row["meta"].update({"nhom_phu": TU_CHOI_HOAN_CANH, "vong": 4, "loai_viec": r.get("loai")})
        rows.append(row)
    rows, bi_loc = loc_rows(rows)
    nhiem = set(tim_nhiem(rows, chan)["chi_so"])
    sach = [r for i, r in enumerate(rows) if i not in nhiem]
    return sach, {
        "giam_khao_giu": len(giu),
        "loc_lai_viec_lam_duoc": lam_duoc,
        "loc_code": len(bi_loc),
        "nhiem_test_dev": len(nhiem),
        "sach": len(sach),
    }


def dung(nguon_moi: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    train = doc(NEN / "train_vi.jsonl")
    chan: list[dict[str, Any]] = []
    for p in TAP_CHAN + [DEV]:
        if p.exists():
            chan += doc(p)
    moi, bc_moi = cau_moi_sach(nguon_moi if nguon_moi is not None else doc(NGUON_MOI), chan)

    cu = [r for r in train if r["meta"].get("nhom_phu") == TU_CHOI_KHO]
    con_lai = [r for r in train if r["meta"].get("nhom_phu") != TU_CHOI_KHO]
    rng = random.Random(SEED)
    chon = rng.sample(moi, min(len(moi), len(cu)))  # CÙNG bộ câu mới cho cả hai nhánh

    nhanh = {
        "final_v4_thay": con_lai + chon,
        "final_v4_them": train + chon,
    }
    bao_cao: dict[str, Any] = {"cau_moi": bc_moi, "tu_choi_kho_cu": len(cu), "cau_moi_dung": len(chon)}
    for ten, rows in nhanh.items():
        rng_nhanh = random.Random(SEED)
        rows = list(rows)
        rng_nhanh.shuffle(rows)
        ra = P / ten
        ghi(ra / "train_vi.jsonl", rows)
        for tap in ("val_vi", "val_mau"):
            ghi(ra / f"{tap}.jsonl", doc(NEN / f"{tap}.jsonl"))  # val y hệt v3
        bao_cao[ten] = {
            "train": len(rows),
            "theo_loai": dict(Counter(r["meta"]["type"] for r in rows)),
            "tu_choi_kho_cu": sum(r["meta"].get("nhom_phu") == TU_CHOI_KHO for r in rows),
            "tu_choi_hoan_canh": sum(r["meta"].get("nhom_phu") == TU_CHOI_HOAN_CANH for r in rows),
        }
        (ra / "bao_cao_dung.json").write_text(
            json.dumps(bao_cao[ten], ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return bao_cao


def dung_cuoi() -> dict[str, Any]:
    """CẤU HÌNH CUỐI (PLAN.md mục P): nền v3 + ① 1.000 mẫu/hàm + ② mô tả hàm trong đáp án
    + ③ dữ liệu đối chứng.

    - ①: lấy NGUYÊN `final_v3_1000` (đã dựng, đã sửa nhãn vòng 3, đã kiểm nhiễm test).
    - ③: bỏ câu từ chối ngắn cũ, thêm ĐÚNG bộ câu kể hoàn cảnh hai phía đã dùng ở bản đối chứng (lấy từ
      `final_v4_doi_chung` theo `nhom_phu`) -> cùng câu, kết quả so sánh được.
    - ②: nối mô tả hàm vào mẫu call/ask (cả train lẫn val) — như thí nghiệm mô tả hàm.
    - Kiểm nhiễm lại TOÀN BỘ train với tập dev: `final_v3_1000` dựng trước khi có tập dev.
    """
    from gen.mo_ta_dap_an import them_mo_ta_rows

    nen = doc(P / "final_v3_1000" / "train_vi.jsonl")
    doi_chung = doc(P / "final_v4_doi_chung" / "train_vi.jsonl")
    cap = [r for r in doi_chung if r["meta"].get("nhom_phu") in (TU_CHOI_HOAN_CANH, GOI_HAM_HOAN_CANH)]
    rows = [r for r in nen if r["meta"].get("nhom_phu") != TU_CHOI_KHO] + cap

    dev = doc(DEV)
    nhiem = set(tim_nhiem(rows, dev)["chi_so"])
    rows = [r for i, r in enumerate(rows) if i not in nhiem]
    random.Random(SEED).shuffle(rows)
    rows = them_mo_ta_rows(rows)

    ra = P / "final_cuoi"
    ghi(ra / "train_vi.jsonl", rows)
    for tap in ("val_vi", "val_mau"):
        ghi(ra / f"{tap}.jsonl", them_mo_ta_rows(doc(NEN / f"{tap}.jsonl")))  # val như v3, có mô tả
    goi = [r for r in rows if r["meta"]["type"] == "call"]
    bao_cao = {
        "train": len(rows),
        "bo_tu_choi_ngan_cu": sum(r["meta"].get("nhom_phu") == TU_CHOI_KHO for r in nen),
        "nhiem_dev_bo": len(nhiem),
        "theo_loai": dict(Counter(r["meta"]["type"] for r in rows)),
        "nhom_phu": dict(Counter(r["meta"].get("nhom_phu") for r in rows if r["meta"].get("nhom_phu"))),
        "call_duong": sum(r["meta"]["fn"] != "irrelevant_function" for r in goi),
        "call_tu_choi": sum(r["meta"]["fn"] == "irrelevant_function" for r in goi),
        "co_mo_ta": sum(1 for r in rows if r["meta"].get("mo_ta_dap_an")),
    }
    (ra / "bao_cao_dung.json").write_text(json.dumps(bao_cao, ensure_ascii=False, indent=2), encoding="utf-8")
    return bao_cao


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if "--cuoi" in sys.argv:
        ket_qua = dung_cuoi()
    elif "--doi-chung" in sys.argv:
        ket_qua = dung_doi_chung()
    else:
        ket_qua = dung()
    print(json.dumps(ket_qua, ensure_ascii=False, indent=2))
