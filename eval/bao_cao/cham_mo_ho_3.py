"""Chấm bộ mơ hồ thứ 3 đúng cách đã KHOÁ trong PLAN.md mục T (2026-10-03)."""
import collections
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


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    t = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (t - h), 100 * (t + h)


MOI = doc("data/processed/final/test_vi_mo_ho_3.jsonl")
MOI = [r for r in MOI if nhom_cua_dap_an(doc_output(r["completion"])) == "mo_ho"]  # bỏ ra_lenh / ngoài phạm vi (ở đây 0)
K72 = du("runs/mo_ho_3/du_doan_bl_sip_72b_test_vi_mo_ho_3.jsonl")
print(f"Bộ mới: {len(MOI)} câu mơ hồ\n")


def dung(out, r):
    return cham_mot_cau(out, r["completion"])[0] == "dung"


ket = {}
for s in ("s42", "s7"):
    S = du(f"runs/mo_ho_3/du_doan_v4dc{s}_test_vi_mo_ho_3.jsonl")
    a = [dung(S[r["prompt"]], r) for r in MOI]
    k = [dung(K72[r["prompt"]], r) for r in MOI]
    b = sum(x and not y for x, y in zip(a, k))
    c = sum(y and not x for x, y in zip(a, k))
    p = mcnemar(b, c)
    ket[s] = (b, c, p)
    la, ha = wilson(sum(a), len(a))
    lk, hk = wilson(sum(k), len(k))
    print(f"[CHÍNH] {s}: student {100*sum(a)/len(a):5.1f}% [{la:.1f}–{ha:.1f}] | 72B {100*sum(k)/len(k):5.1f}% [{lk:.1f}–{hk:.1f}]"
          f" | student hơn {b}, 72B hơn {c}, p = {p:.4f}")
    print("        kiểu sai student:", dict(collections.Counter(cham_mot_cau(S[r['prompt']], r['completion'])[0] for r in MOI)))
    print("        kiểu sai 72B    :", dict(collections.Counter(cham_mot_cau(K72[r['prompt']], r['completion'])[0] for r in MOI)))

# kết luận theo luật đã khoá
co_y_nghia = [s for s, (b, c, p) in ket.items() if p < 0.05]
cung_chieu = all(b > c for b, c, _ in ket.values()) or all(b < c for b, c, _ in ket.values())
if len(co_y_nghia) == 2 and cung_chieu:
    kl = "student HƠN 72B" if all(b > c for b, c, _ in ket.values()) else "72B HƠN student"
elif len(co_y_nghia) == 1:
    kl = "ngang hoặc nhỉnh hơn (chỉ 1/2 hạt giống có ý nghĩa)"
else:
    kl = "không phân biệt được"
print(f"\n=> KẾT LUẬN (luật đã khoá): {kl}\n")

# phụ: gộp với 39 + 28 câu cũ
cu = [("test_vi_sach_v2", "data/processed/final/test_vi_sach_v2.jsonl"), ("test_vi_bo_sung", "data/processed/final/test_vi_bo_sung.jsonl")]
for s in ("s42", "s7"):
    S = du(f"runs/mo_ho_3/du_doan_v4dc{s}_test_vi_mo_ho_3.jsonl")
    b = sum(dung(S[r["prompt"]], r) and not dung(K72[r["prompt"]], r) for r in MOI)
    c = sum(dung(K72[r["prompt"]], r) and not dung(S[r["prompt"]], r) for r in MOI)
    n, ds, dk = len(MOI), sum(dung(S[r["prompt"]], r) for r in MOI), sum(dung(K72[r["prompt"]], r) for r in MOI)
    for tap, p in cu:
        rows = [r for r in doc(p) if nhom_cua_dap_an(doc_output(r["completion"])) == "mo_ho"]
        A = du(f"runs/student-v4dc-{s}/du_doan_{tap}.jsonl")
        B = du(f"runs/baseline/sip_72b_{tap}.jsonl")
        for r in rows:
            x, y = dung(A[r["prompt"]], r), dung(B[r["prompt"]], r)
            b += x and not y
            c += y and not x
            ds += x
            dk += y
            n += 1
    print(f"[PHỤ, gộp {n} câu] {s}: student {100*ds/n:.1f}% | 72B {100*dk/n:.1f}% | student hơn {b}, 72B hơn {c}, p = {mcnemar(b, c):.4f}")

print("\nTheo hàm (mô tả, bộ mới): đúng / 10 — s42 · s7 · 72B")
S42 = du("runs/mo_ho_3/du_doan_v4dcs42_test_vi_mo_ho_3.jsonl")
S7 = du("runs/mo_ho_3/du_doan_v4dcs7_test_vi_mo_ho_3.jsonl")
theo = collections.defaultdict(lambda: [0, 0, 0, 0])
for r in MOI:
    h = r["meta"]["nhom_phu"]
    t = theo[h]
    t[0] += dung(S42[r["prompt"]], r)
    t[1] += dung(S7[r["prompt"]], r)
    t[2] += dung(K72[r["prompt"]], r)
    t[3] += 1
for h, (x, y, z, n) in sorted(theo.items()):
    print(f"  {h:24} {x:2d} · {y:2d} · {z:2d}  / {n}")
