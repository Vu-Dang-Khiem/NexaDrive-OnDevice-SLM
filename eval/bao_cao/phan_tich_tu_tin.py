import collections
import json
import sys

sys.path.insert(0, ".")
from schema.loader import CATALOG

TU_CHOI = CATALOG.by_name["irrelevant_function"].nexa_token
ASK = CATALOG.nexa_ask
R = [json.loads(x) for x in open("runs/tu_tin/v4dc_s42_ck2206.jsonl", encoding="utf-8")]


def tok1(d):
    return d["top_buoc_1"][0][0]


def la_goi_ham(d):  # bước 1 là token hàm thật (không phải từ chối, không phải hỏi lại)
    return tok1(d) not in (TU_CHOI, ASK)


def phan_vi(xs):
    xs = sorted(xs)
    if not xs:
        return "-"
    q = lambda f: xs[min(len(xs) - 1, int(f * len(xs)))]
    return f"n={len(xs):4d} min={xs[0]:.3f} p10={q(.1):.3f} p25={q(.25):.3f} median={q(.5):.3f} p75={q(.75):.3f}"


for tap in ("val_vi", "dev_vi_2"):
    D = [d for d in R if d["tap"] == tap]
    print(f"\n================ {tap}: {len(D)} câu ================")
    for nhom in ("ro_rang", "mo_ho", "ngoai_pham_vi"):
        N = [d for d in D if d["nhom"] == nhom]
        if N:
            c = collections.Counter(d["ket_qua"] for d in N)
            print(f"{nhom:14} n={len(N):4d} đúng {100*c['dung']/len(N):5.1f}%  {dict(c)}")

    print("\nXác suất token hàm ở bước 1 (chỉ các câu model GỌI HÀM THẬT ngay bước 1):")
    print("  gọi bừa (đáp án là từ chối)  :", phan_vi([d["p_buoc_1"] for d in D if la_goi_ham(d) and d["ket_qua"] == "goi_bua"]))
    print("  đúng, câu rõ ràng            :", phan_vi([d["p_buoc_1"] for d in D if la_goi_ham(d) and d["nhom"] == "ro_rang" and d["ket_qua"] == "dung"]))
    print("  sai khác, câu rõ ràng        :", phan_vi([d["p_buoc_1"] for d in D if la_goi_ham(d) and d["nhom"] == "ro_rang" and d["ket_qua"] != "dung"]))
    print("  câu mơ hồ mà gọi luôn        :", phan_vi([d["p_buoc_1"] for d in D if la_goi_ham(d) and d["nhom"] == "mo_ho"]))

    print("\nQuét ngưỡng (chỉ chặn khi bước 1 là token hàm thật và p < ngưỡng):")
    print("  ngưỡng | gọi bừa bị chặn | câu rõ ràng ĐÚNG bị chặn (mất) | câu rõ ràng SAI bị chặn")
    gb = [d for d in D if la_goi_ham(d) and d["ket_qua"] == "goi_bua"]
    rr_d = [d for d in D if la_goi_ham(d) and d["nhom"] == "ro_rang" and d["ket_qua"] == "dung"]
    rr_s = [d for d in D if la_goi_ham(d) and d["nhom"] == "ro_rang" and d["ket_qua"] != "dung"]
    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        f = lambda L: sum(d["p_buoc_1"] < t for d in L)
        print(f"  {t:5.2f}  | {f(gb):3d}/{len(gb):3d}          | {f(rr_d):4d}/{len(rr_d):4d}                  | {f(rr_s):3d}/{len(rr_s):3d}")

    print("\nCác câu gọi bừa (p bước 1, top-3):")
    for d in sorted(gb, key=lambda d: d["p_buoc_1"]):
        top = ", ".join(f"{t}:{p:.2f}" for t, p in d["top_buoc_1"][:3])
        print(f"  p={d['p_buoc_1']:.3f} | {d['cau'][:70]} | {top}")
