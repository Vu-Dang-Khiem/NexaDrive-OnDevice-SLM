"""Chấm mo_ho_4 đúng cách đã KHOÁ (PLAN.md mục U): v5 vs đối chứng, McNemar từng hạt giống; kèm 72B, ngắn/dài, theo hàm."""
import collections
import json
import math
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from eval.metrics import cham_mot_cau
from gen.records import query_from_call_prompt as q

TEST = [json.loads(x) for x in open("data/processed/final/test_vi_mo_ho_4.jsonl", encoding="utf-8")]
DAP = {q(r["prompt"]): r for r in TEST}


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    t = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (t - h), 100 * (t + h)


def doc_student(ten):
    return {d["cau"]: d["ket_qua"] for d in (json.loads(x) for x in open(f"runs/mo_ho_4/{ten}.jsonl", encoding="utf-8"))}


K = {}
for x in open("runs/mo_ho_4/sip_72b.jsonl", encoding="utf-8"):
    r = json.loads(x)
    K[q(r["prompt"])] = cham_mot_cau(r["output"], DAP[q(r["prompt"])]["completion"])[0]
M = {t: doc_student(t) for t in ("v5_s42", "v5_s7", "v4dc_s42", "v4dc_s7")}
M["72B"] = K
cau = list(DAP)
assert all(set(m) == set(cau) for m in M.values()), "thiếu câu"
print(f"Bộ mo_ho_4: {len(cau)} câu mơ hồ\n")
for t, m in M.items():
    k = sum(m[c] == "dung" for c in cau)
    lo, hi = wilson(k, len(cau))
    print(f"  {t:9} {100 * k / len(cau):5.1f}% [{lo:.1f}–{hi:.1f}]  lỗi: {dict(collections.Counter(v for v in m.values() if v != 'dung'))}")


def so(a, b, ten):
    A, B = M[a], M[b]
    x = sum(A[c] == "dung" and B[c] != "dung" for c in cau)
    y = sum(B[c] == "dung" and A[c] != "dung" for c in cau)
    p = mcnemar(x, y)
    print(f"  {ten:28} {a} hơn {x}, {b} hơn {y}, p = {p:.4f}")
    return x, y, p


print("\n[CHÍNH] v5 vs đối chứng (cùng hạt giống):")
kq = [so("v5_s42", "v4dc_s42", "s42"), so("v5_s7", "v4dc_s7", "s7")]
co = [p < 0.05 for _, _, p in kq]
chieu_v5 = all(x > y for x, y, _ in kq)
if all(co) and chieu_v5:
    kl = "v5 TỐT HƠN bản đối chứng ở câu mơ hồ"
elif all(co) and all(x < y for x, y, _ in kq):
    kl = "v5 KÉM HƠN bản đối chứng ở câu mơ hồ"
elif any(co):
    kl = "có xu hướng (chỉ 1/2 hạt giống có ý nghĩa)"
else:
    kl = "không phân biệt được"
print(f"  => KẾT LUẬN (luật đã khoá): {kl}")

print("\n[KÈM] so với 72B:")
for t in ("v5_s42", "v5_s7", "v4dc_s42", "v4dc_s7"):
    so(t, "72B", t + " vs 72B")

print("\n[KÈM] câu ngắn ≤10 chữ / dài >10 chữ:")
ngan = [c for c in cau if len(c.split()) <= 10]
dai = [c for c in cau if len(c.split()) > 10]
for t, m in M.items():
    f = lambda ds: 100 * sum(m[c] == "dung" for c in ds) / len(ds)  # noqa: E731
    tcn = sum(m[c] == "tu_choi_nham" for c in dai)
    print(f"  {t:9} ngắn ({len(ngan)}) {f(ngan):5.1f} | dài ({len(dai)}) {f(dai):5.1f} | từ chối nhầm câu dài {tcn}")

print("\n[KÈM] theo hàm (đúng / số câu): v5 s42 · v5 s7 · đc s42 · đc s7 · 72B")
theo = collections.defaultdict(list)
for c in cau:
    theo[DAP[c]["meta"]["nhom_phu"]].append(c)
for h, ds in sorted(theo.items()):
    vals = " · ".join(f"{sum(M[t][c] == 'dung' for c in ds):2d}" for t in ("v5_s42", "v5_s7", "v4dc_s42", "v4dc_s7", "72B"))
    print(f"  {h:24} {vals}  / {len(ds)}")
