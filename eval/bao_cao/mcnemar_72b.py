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


TAP = {
    "test_vi_sach_v2": "data/processed/final/test_vi_sach_v2.jsonl",
    "test_vi_bo_sung": "data/processed/final/test_vi_bo_sung.jsonl",
}
for s in ("s42", "s7"):
    print(f"\n######## student đối chứng {s}  vs  72B SIP ########")
    gop = {"b": 0, "c": 0, "s": 0, "k": 0, "n": 0}
    for tap, p in TAP.items():
        rows = doc(p)
        A = du(f"runs/student-v4dc-{s}/du_doan_{tap}.jsonl")
        B = du(f"runs/baseline/sip_72b_{tap}.jsonl")
        for nhom in ("ro_rang", "mo_ho", "ngoai_pham_vi"):
            R = [r for r in rows if nhom_cua_dap_an(doc_output(r["completion"])) == nhom]
            if not R:
                continue
            a = [cham_mot_cau(A[r["prompt"]], r["completion"])[0] == "dung" for r in R]
            k = [cham_mot_cau(B[r["prompt"]], r["completion"])[0] == "dung" for r in R]
            b = sum(x and not y for x, y in zip(a, k))  # student đúng, 72B sai
            c = sum(y and not x for x, y in zip(a, k))  # 72B đúng, student sai
            n = len(R)
            la, ha = wilson(sum(a), n)
            lk, hk = wilson(sum(k), n)
            print(f"  {tap:16} {nhom:14} n={n:3d} | student {100*sum(a)/n:5.1f} [{la:4.1f}–{ha:4.1f}]"
                  f" | 72B {100*sum(k)/n:5.1f} [{lk:4.1f}–{hk:4.1f}] | student hơn {b}, 72B hơn {c}, p = {mcnemar(b, c):.3f}")
            if nhom == "mo_ho":
                gop["b"] += b
                gop["c"] += c
                gop["s"] += sum(a)
                gop["k"] += sum(k)
                gop["n"] += n
    g = gop
    print(f"  GỘP mơ hồ (39 + 28 = {g['n']}) | student {100*g['s']/g['n']:.1f} | 72B {100*g['k']/g['n']:.1f}"
          f" | student hơn {g['b']}, 72B hơn {g['c']}, p = {mcnemar(g['b'], g['c']):.3f}")
