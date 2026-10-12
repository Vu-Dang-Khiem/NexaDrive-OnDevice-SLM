"""Test MỘT lần (rà soát 2026-10-08): BẢN TRIỂN KHAI v6.1 s42 q4_K_M (không ngưỡng / 0,80 đã chọn trên tập hiệu chỉnh)
vs bản HF s42 không ngưỡng (`runs/v6/du_doan_v61s42_*`). Dự đoán q4: `eval/do_tu_tin_gguf.py --cho-phep-test`
-> `runs/gguf/test_v61s42_q4.jsonl`. Chỉ báo cáo — không chọn / đổi gì theo số này.
"""
import math
import sys
from statistics import median

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from eval.metrics import DUNG, cham_mot_cau
from eval.so_muc_nen import doc, kq_nguong

SAI_LEN_XE = {"goi_bua", "sai_ham", "sai_tham_so", "thieu_hoi_lai"}
Q4 = doc("runs/gguf/test_v61s42_q4.jsonl")


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def hf(tap):
    """Bản HF s42: (câu, đáp án) -> đúng?"""
    du = {r["prompt"]: r["output"] for r in doc(f"runs/v6/du_doan_v61s42_{tap}.jsonl")}
    rows = doc(f"data/processed/final/{tap}.jsonl")
    return [cham_mot_cau(du[r["prompt"]], r["completion"])[0] == DUNG for r in rows if r["meta"]["type"] in ("call", "ask")]


def nhom(tap, loc=lambda d: True):
    return [d for d in Q4 if d["tap"] == tap and loc(d)]


def pt(R, t):
    return 100 * sum(kq_nguong(d, t) == "dung" for d in R) / len(R)


DONG = [
    ("test sạch — rõ ràng (74)", nhom("test_vi_sach_v2", lambda d: d["nhom"] == "ro_rang")),
    ("test sạch — mơ hồ (39)", nhom("test_vi_sach_v2", lambda d: d["nhom"] == "mo_ho")),
    ("test sạch — ngoài pv (117)", nhom("test_vi_sach_v2", lambda d: d["nhom"] == "ngoai_pham_vi")),
    ("28 câu bổ sung", nhom("test_vi_bo_sung")),
    ("mo_ho_4 (129)", nhom("test_vi_mo_ho_4")),
    ("v6 A+B câu ngắn / cách gọi (120)", nhom("test_v6_mot_luot", lambda d: d["nhom_phu"] in ("v6_A", "v6_B"))),
    ("v6 C tính năng mới (30)", nhom("test_v6_mot_luot", lambda d: d["nhom_phu"] == "v6_C")),
    ("v6 D hai lượt (60)", nhom("test_v6_hai_luot")),
    ("v6 E nhiễu (104)", nhom("test_v6_nhieu")),
]
print(f"{'':36} | q4 không ngưỡng | q4 + 0,80 (TRIỂN KHAI)")
for ten, R in DONG:
    print(f"{ten:36} | {pt(R, 0.0):15.1f} | {pt(R, 0.80):10.1f}")

S = nhom("test_vi_sach_v2")
for t, ten in ((0.0, "không ngưỡng"), (0.80, "+ 0,80")):
    k = [kq_nguong(d, t) for d in S]
    print(f"test sạch 230, q4 {ten}: gọi bừa {k.count('goi_bua')} · hỏi thừa {k.count('hoi_lai_thua')} · "
          f"từ chối nhầm {k.count('tu_choi_nham')} · sai lên xe {sum(x in SAI_LEN_XE for x in k)}")

print("\nKiểm q4 (không ngưỡng) vs HF s42 cùng câu — McNemar:")
for tap in ("test_vi_sach_v2", "test_vi_bo_sung", "test_vi_mo_ho_4", "test_v6_mot_luot", "test_v6_hai_luot", "test_v6_nhieu"):
    h = hf(tap)
    q = [d["ket_qua"] == DUNG for d in nhom(tap)]
    assert len(h) == len(q), (tap, len(h), len(q))
    b = sum(x and not y for x, y in zip(h, q))
    c = sum(y and not x for x, y in zip(h, q))
    print(f"  {tap:18} HF {100*sum(h)/len(h):5.1f} | q4 {100*sum(q)/len(q):5.1f} | HF hơn {b}, q4 hơn {c}, p = {mcnemar(b, c):.3f}")

ms = sorted(d["ms_doc_prompt"] + d["ms_sinh"] for d in Q4)
print(f"\nĐộ trễ lượt 1 (q4, CPU 8 luồng, {len(ms)} câu, chạy riêng): p50 {median(ms):.0f} ms · "
      f"p95 {ms[int(0.95 * len(ms)) - 1]:.0f} ms · lớn nhất {ms[-1]:.0f} ms")
