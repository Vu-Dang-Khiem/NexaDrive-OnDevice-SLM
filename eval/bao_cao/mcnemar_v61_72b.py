"""McNemar v6.1 (HF, không ngưỡng, từng hạt giống) vs 72B SIP — cùng cách chấm `mcnemar_72b.py` (rà soát 2026-10-08).

Luật đọc (như mục T / U): "v6.1 hơn" / "72B hơn" chỉ khi p < 0,05 ở CẢ HAI hạt giống, cùng chiều; một hạt giống ->
"ngang hoặc nhỉnh hơn"; không hạt giống nào -> "không phân biệt được".
"""
import json
import math
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from eval.metrics import cham_mot_cau, doc_output, nhom_cua_dap_an


def doc(p):
    return [json.loads(x) for x in open(p, encoding="utf-8")]


def du(p):
    return {r["prompt"]: r["output"] for r in doc(p)}


def mcnemar(b, c):  # chính xác, hai phía
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    t = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (t - h), 100 * (t + h)


TAP = {  # tập -> (file đáp án, file dự đoán 72B)
    "test_vi_sach_v2": ("data/processed/final/test_vi_sach_v2.jsonl", "runs/baseline/sip_72b_test_vi_sach_v2.jsonl"),
    "test_vi_bo_sung": ("data/processed/final/test_vi_bo_sung.jsonl", "runs/baseline/sip_72b_test_vi_bo_sung.jsonl"),
    "test_vi_mo_ho_4": ("data/processed/final/test_vi_mo_ho_4.jsonl", "runs/mo_ho_4/sip_72b.jsonl"),
}
for s in ("s42", "s7"):
    print(f"\n######## v6.1 {s} (HF, không ngưỡng)  vs  72B SIP ########")
    for tap, (p, p72) in TAP.items():
        rows = doc(p)
        A = du(f"runs/v6/du_doan_v61{s}_{tap}.jsonl")
        B = du(p72)
        for nhom in ("ro_rang", "mo_ho", "ngoai_pham_vi"):
            R = [r for r in rows if nhom_cua_dap_an(doc_output(r["completion"])) == nhom]
            if not R:
                continue
            a = [cham_mot_cau(A[r["prompt"]], r["completion"])[0] == "dung" for r in R]
            k = [cham_mot_cau(B[r["prompt"]], r["completion"])[0] == "dung" for r in R]
            b = sum(x and not y for x, y in zip(a, k))  # v6.1 đúng, 72B sai
            c = sum(y and not x for x, y in zip(a, k))  # 72B đúng, v6.1 sai
            n = len(R)
            la, ha = wilson(sum(a), n)
            lk, hk = wilson(sum(k), n)
            print(f"  {tap:16} {nhom:14} n={n:3d} | v6.1 {100*sum(a)/n:5.1f} [{la:4.1f}–{ha:4.1f}]"
                  f" | 72B {100*sum(k)/n:5.1f} [{lk:4.1f}–{hk:4.1f}] | v6.1 hơn {b}, 72B hơn {c}, p = {mcnemar(b, c):.3f}")
