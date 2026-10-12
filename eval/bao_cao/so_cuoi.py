import json
import sys

sys.stdout = open(sys.argv[1], "w", encoding="utf-8")
sys.path.insert(0, ".")
from eval.metrics import cham_mot_cau, cham_toan_bo, doc_output


def doc(p):
    return [json.loads(x) for x in open(p, encoding="utf-8")]


def du(p):
    return {r["prompt"]: r["output"] for r in doc(p)}


sach = doc("data/processed/final/test_vi_sach_v2.jsonl")
bs = doc("data/processed/final/test_vi_bo_sung.jsonl")
dev = doc("data/dev_tay_2/dev_vi_2.jsonl")
CAU_HINH = {
    "v3": {42: ("runs/student-full-v3", "runs/dev_moc/v3_s42.jsonl"), 7: ("runs/student-full-v3-s7", "runs/dev_moc/v3_s7.jsonl")},
    "v4thay": {sd: (f"runs/student-v4thay-s{sd}", f"runs/student-v4thay-s{sd}/du_doan_dev_vi_2.jsonl") for sd in (42, 7)},
    "v4them": {sd: (f"runs/student-v4them-s{sd}", f"runs/student-v4them-s{sd}/du_doan_dev_vi_2.jsonl") for sd in (42, 7)},
    "v4dc": {sd: (f"runs/student-v4dc-s{sd}", f"runs/student-v4dc-s{sd}/du_doan_dev_vi_2.jsonl") for sd in (42, 7)},
    "cuoi": {sd: (f"runs/student-cuoi-s{sd}", f"runs/student-cuoi-s{sd}/du_doan_dev_vi_2.jsonl") for sd in (42, 7)},
}


def cham_dev(du_dev):
    tc = [0, 0]
    ld = [0, 0]
    for r in dev:
        g = r["meta"]["nhom_phu"]
        out = du_dev[r["prompt"]]
        if g.startswith("dev_lam_duoc"):
            p, d = doc_output(out), doc_output(r["completion"])
            ok = p.doc_duoc and p.ten_ham[:1] == d.ten_ham[:1]
            ld[0] += ok
            ld[1] += 1
        else:
            tc[0] += cham_mot_cau(out, r["completion"])[0] == "dung"
            tc[1] += 1
    return tc[0], ld[0]


CHI = ("ro_rang", "mo_ho", "ngoai_pham_vi", "goi_bua", "moi28", "dev_tu_choi", "dev_lam_duoc")
diem = {}
for ten, sds in CAU_HINH.items():
    for sd, (thu_muc, dev_p) in sds.items():
        a = cham_toan_bo(sach, du(f"{thu_muc}/du_doan_test_vi_sach_v2.jsonl"))
        b = cham_toan_bo(bs, du(f"{thu_muc}/du_doan_test_vi_bo_sung.jsonl"))
        tc, ld = cham_dev(du(dev_p))
        diem[(ten, sd)] = {
            "ro_rang": a["theo_nhom"]["ro_rang"]["dung_%"], "mo_ho": a["theo_nhom"]["mo_ho"]["dung_%"],
            "ngoai_pham_vi": a["theo_nhom"]["ngoai_pham_vi"]["dung_%"], "goi_bua": a["ty_le_loi_%"]["goi_bua"],
            "tu_choi_nham": a["ty_le_loi_%"]["tu_choi_nham"],
            "moi28": b["theo_nhom"]["mo_ho"]["dung_%"], "dev_tu_choi": tc, "dev_lam_duoc": ld,
        }
        print(ten, sd, diem[(ten, sd)])

print("\n=== TRUNG BÌNH 2 hạt giống | (chênh giữa 2 hạt giống) ===")
print(f"{'':8}" + "".join(f"{c:>16}" for c in CHI))
for ten in CAU_HINH:
    a, b = diem[(ten, 42)], diem[(ten, 7)]
    print(f"{ten:8}" + "".join(f"{(a[c] + b[c]) / 2:>9.1f} ({abs(a[c] - b[c]):>4.1f})" for c in CHI))
print("\ntừ chối nhầm:", {k: v["tu_choi_nham"] for k, v in diem.items()})



