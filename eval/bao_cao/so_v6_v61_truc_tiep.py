"""(1) McNemar TRỰC TIẾP v6 vs v6.1 (từng hạt giống, HF không ngưỡng) và (2) "hành động sai lên xe" của bản HF
v6.1 trên test sạch. Chỉ đọc dự đoán đã lưu ở runs/v6/ — không chạy model. Rà soát 2026-10-12.
"""
import json
import math
import sys
from collections import Counter

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from eval.metrics import DUNG, cham_mot_cau, doc_output, nhom_cua_dap_an

SAI_LEN_XE = {"goi_bua", "sai_ham", "sai_tham_so", "thieu_hoi_lai"}


def doc(p):
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def ket_cuc(ban, tap):
    """Danh sách kết cục từng câu (theo đúng thứ tự file đáp án) + nhãn nhóm."""
    du = {r["prompt"]: r["output"] for r in doc(f"runs/v6/du_doan_{ban}_{tap}.jsonl")}
    rows = [r for r in doc(f"data/processed/final/{tap}.jsonl") if r["meta"]["type"] in ("call", "ask")]
    kq = [cham_mot_cau(du[r["prompt"]], r["completion"])[0] for r in rows]
    nhom = [r["meta"].get("nhom_phu") if tap == "test_v6_mot_luot" else nhom_cua_dap_an(doc_output(r["completion"]))
            for r in rows]
    return kq, nhom


# tập, bộ lọc nhóm, tên hiển thị
PHEP = [
    ("test_vi_sach_v2", "ro_rang", "Rõ ràng (74)"),
    ("test_vi_sach_v2", "mo_ho", "Mơ hồ (39)"),
    ("test_vi_sach_v2", "ngoai_pham_vi", "Ngoài phạm vi (117)"),
    ("test_vi_bo_sung", None, "Bổ sung (28)"),
    ("test_vi_mo_ho_4", None, "Mơ hồ dài (129)"),
    ("test_v6_mot_luot", ("v6_A", "v6_B"), "Câu ngắn + cách gọi (120)"),
    ("test_v6_mot_luot", ("v6_C",), "Tham số mới (30)"),
    ("test_v6_hai_luot", None, "Hai lượt (60)"),
    ("test_v6_nhieu", None, "Gõ sai (104)"),
]

print("(1) McNemar TRỰC TIẾP v6 vs v6.1 — HF, không ngưỡng; b = v6.1 đúng v6 sai, c = ngược lại")
for hat in ("s42", "s7"):
    print(f"\n== hạt giống {hat}")
    for tap, loc, ten in PHEP:
        a, nh = ket_cuc(f"v6{hat}", tap)
        m, _ = ket_cuc(f"v61{hat}", tap)
        chon = [i for i, g in enumerate(nh) if loc is None or g == loc or (isinstance(loc, tuple) and g in loc)]
        v6 = [a[i] == DUNG for i in chon]
        v61 = [m[i] == DUNG for i in chon]
        b = sum(y and not x for x, y in zip(v6, v61))
        c = sum(x and not y for x, y in zip(v6, v61))
        n = len(chon)
        print(f"  {ten:28} v6 {100*sum(v6)/n:5.1f} | v6.1 {100*sum(v61)/n:5.1f} | v6.1 hơn {b:2d}, v6 hơn {c:2d}, "
              f"p = {mcnemar(b, c):.3f}")

print("\n(2) Hành động sai lên xe, test sạch 230 câu, HF không ngưỡng "
      "(gọi bừa + sai hàm + sai tham số + thiếu hỏi lại)")
for ban in ("v61s42", "v61s7", "v6s42", "v6s7"):
    kq, _ = ket_cuc(ban, "test_vi_sach_v2")
    dem = Counter(kq)
    print(f"  {ban:7} sai lên xe {sum(dem[k] for k in SAI_LEN_XE):3d}  ({dict((k, dem[k]) for k in sorted(SAI_LEN_XE))})")
