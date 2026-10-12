import collections
import json
import math
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from schema.loader import CATALOG

TU_CHOI = CATALOG.by_name["irrelevant_function"].nexa_token
ASK = CATALOG.nexa_ask
NGUONG = 0.90  # đã khoá trong PLAN.md mục S trước khi chạy test


def ket_qua_moi(d, t):
    (t1, p1), (t2, _) = d["top_buoc_1"][0], d["top_buoc_1"][1]
    if t1 in (TU_CHOI, ASK) or p1 >= t:
        return d["ket_qua"]
    q = "tu_choi" if t2 == TU_CHOI else "hoi_lai"
    if d["nhom"] == "ngoai_pham_vi":
        return "dung" if q == "tu_choi" else "hoi_lai_khi_phai_tu_choi"
    if d["nhom"] == "ro_rang":
        return "tu_choi_nham" if q == "tu_choi" else "hoi_lai_thua"
    return "mo_ho_" + q  # câu mơ hồ mà model gọi luôn -> đổi thành hỏi lại/từ chối


def mcnemar(b, c):  # chính xác, hai phía
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


tong = collections.defaultdict(list)
for s in ("s42", "s7"):
    R = [json.loads(x) for x in open(f"runs/tu_tin/test_v4dc_{s}_ck2206.jsonl", encoding="utf-8")]
    print(f"\n######## {s} ########")
    for tap in ("test_vi_sach_v2", "test_vi_bo_sung"):
        D = [d for d in R if d["tap"] == tap]
        print(f"--- {tap}: {len(D)} câu")
        for ten, t in (("không ngưỡng", 0.0), (f"ngưỡng {NGUONG}", NGUONG)):
            kq = {n: collections.Counter(ket_qua_moi(d, t) for d in D if d["nhom"] == n) for n in ("ro_rang", "mo_ho", "ngoai_pham_vi")}
            dong = []
            for n in ("ro_rang", "mo_ho", "ngoai_pham_vi"):
                N = sum(kq[n].values())
                if N:
                    dong.append(f"{n} {100 * kq[n]['dung'] / N:5.1f}% ({kq[n]['dung']}/{N})")
                    tong[(s, tap, ten, n)] = 100 * kq[n]["dung"] / N
            loi = collections.Counter()
            for n in kq:
                loi.update({k: v for k, v in kq[n].items() if k != "dung"})
            print(f"  {ten:13} | " + " | ".join(dong))
            print(f"  {'':13}   lỗi: {dict(loi)}")
        doi = collections.Counter((d["nhom"], d["ket_qua"], ket_qua_moi(d, NGUONG)) for d in D if ket_qua_moi(d, NGUONG) != d["ket_qua"])
        b = sum(v for (_, cu_, moi), v in doi.items() if cu_ != "dung" and moi == "dung")
        c = sum(v for (_, cu_, moi), v in doi.items() if cu_ == "dung" and moi != "dung")
        print(f"  đổi: {dict(doi)}")
        print(f"  McNemar (toàn tập): sai->đúng {b}, đúng->sai {c}, p = {mcnemar(b, c):.3f}")

print("\n######## TRUNG BÌNH 2 HẠT GIỐNG ########")
for tap in ("test_vi_sach_v2", "test_vi_bo_sung"):
    for ten in ("không ngưỡng", f"ngưỡng {NGUONG}"):
        xs = []
        for n in ("ro_rang", "mo_ho", "ngoai_pham_vi"):
            v = [tong[(s, tap, ten, n)] for s in ("s42", "s7") if (s, tap, ten, n) in tong]
            if v:
                xs.append(f"{n} {sum(v) / len(v):5.1f} (chênh {abs(v[0] - v[-1]):.1f})")
        print(f"  {tap:16} {ten:13} | " + " | ".join(xs))
