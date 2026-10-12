"""v5 vs bản đối chứng — tiêu chí 'không được tụt' đã KHOÁ trong PLAN.md mục U (2026-10-04)."""
import collections
import json
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from eval.metrics import cham_mot_cau, doc_output
from schema.loader import CATALOG

TU_CHOI = CATALOG.by_name["irrelevant_function"].nexa_token
ASK = CATALOG.nexa_ask


def kq_nguong(d, t):
    (t1, p1), (t2, _) = d["top_buoc_1"][0], d["top_buoc_1"][1]
    if t1 in (TU_CHOI, ASK) or p1 >= t:
        return d["ket_qua"]
    q = "tu_choi" if t2 == TU_CHOI else "hoi_lai"
    if d["nhom"] == "ngoai_pham_vi":
        return "dung" if q == "tu_choi" else "hoi_lai_khi_phai_tu_choi"
    if d["nhom"] == "ro_rang":
        return "tu_choi_nham" if q == "tu_choi" else "hoi_lai_thua"
    return "mo_ho_" + q


def bang(path, ten):
    R = [json.loads(x) for x in open(path, encoding="utf-8")]
    out = {}
    for t, nhan in ((0.0, ""), (0.90, "+ng")):
        S = [d for d in R if d["tap"] == "test_vi_sach_v2"]
        for n in ("ro_rang", "mo_ho", "ngoai_pham_vi"):
            N = [d for d in S if d["nhom"] == n]
            out[n + nhan] = 100 * sum(kq_nguong(d, t) == "dung" for d in N) / len(N)
        out["tu_choi_nham%" + nhan] = 100 * sum(kq_nguong(d, t) == "tu_choi_nham" for d in S) / len(S)
        out["goi_bua" + nhan] = sum(kq_nguong(d, t) == "goi_bua" for d in S)
    B = [d for d in R if d["tap"] == "test_vi_bo_sung"]
    out["28_cau"] = 100 * sum(d["ket_qua"] == "dung" for d in B) / len(B)
    out["28_tu_choi_nham"] = sum(d["ket_qua"] == "tu_choi_nham" for d in B)
    return out, R


def dev(R_or_path, la_tu_tin=True):
    import json as js

    rows = [js.loads(x) for x in open("data/dev_tay_2/dev_vi_2.jsonl", encoding="utf-8")]
    if la_tu_tin:
        D = {d["cau"]: d["output"] for d in R_or_path if d["tap"] == "dev_vi_2"}
        from gen.records import query_from_call_prompt as q

        get = lambda r: D[q(r["prompt"])]  # noqa: E731
    else:
        D = {js.loads(x)["prompt"]: js.loads(x)["output"] for x in open(R_or_path, encoding="utf-8")}
        get = lambda r: D[r["prompt"]]  # noqa: E731
    tc = ld = 0
    for r in rows:
        out = get(r)
        if r["meta"]["nhom_phu"].startswith("dev_lam_duoc"):
            p, d = doc_output(out), doc_output(r["completion"])
            ld += p.doc_duoc and p.ten_ham[:1] == d.ten_ham[:1]
        else:
            tc += cham_mot_cau(out, r["completion"])[0] == "dung"
    return tc, ld


kq = collections.defaultdict(dict)
for s in ("s42", "s7"):
    kq["doi_chung"][s], _ = bang(f"runs/tu_tin/test_v4dc_{s}_ck2206.jsonl", "dc")
    kq["doi_chung"][s]["dev"] = dev(f"runs/student-v4dc-{s}/du_doan_dev_vi_2.jsonl", la_tu_tin=False)
    kq["v5"][s], R = bang(f"runs/tu_tin/test_v5_{s}_ck2416.jsonl", "v5")
    kq["v5"][s]["dev"] = dev(R)

khoa = ["ro_rang", "mo_ho", "ngoai_pham_vi", "ngoai_pham_vi+ng", "ro_rang+ng", "tu_choi_nham%", "tu_choi_nham%+ng",
        "goi_bua", "goi_bua+ng", "28_cau", "28_tu_choi_nham", "dev"]
print(f"{'chỉ số':18} | {'đối chứng s42 / s7 (tb)':30} | {'v5 s42 / s7 (tb)':30}")
for k in khoa:
    dong = []
    for m in ("doi_chung", "v5"):
        a, b = kq[m]["s42"][k], kq[m]["s7"][k]
        if k == "dev":
            dong.append(f"{a} / {b}")
        else:
            dong.append(f"{a:5.1f} / {b:5.1f} ({(a + b) / 2:5.1f})")
    print(f"{k:18} | {dong[0]:30} | {dong[1]:30}")
