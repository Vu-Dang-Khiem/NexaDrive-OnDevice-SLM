import collections
import json
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from schema.loader import CATALOG

TU_CHOI = CATALOG.by_name["irrelevant_function"].nexa_token
ASK = CATALOG.nexa_ask
print("token từ chối =", TU_CHOI)
R = [json.loads(x) for x in open("runs/tu_tin/v4dc_s42_ck2206.jsonl", encoding="utf-8")]


def quyet(d, t):
    """Trả về hành động sau khi áp ngưỡng: None = giữ nguyên output model."""
    (t1, p1), (t2, _p2) = d["top_buoc_1"][0], d["top_buoc_1"][1]
    if t1 in (TU_CHOI, ASK) or p1 >= t:
        return None
    return "tu_choi" if t2 == TU_CHOI else "hoi_lai"


def ket_qua_moi(d, t):
    q = quyet(d, t)
    if q is None:
        return d["ket_qua"]
    if d["nhom"] == "ngoai_pham_vi":
        return "dung" if q == "tu_choi" else "hoi_lai_khi_phai_tu_choi"
    if d["nhom"] == "ro_rang":
        return "tu_choi_nham" if q == "tu_choi" else "hoi_lai_thua"
    return "mo_ho_" + q


for tap in ("val_vi", "dev_vi_2", "ca_hai"):
    D = R if tap == "ca_hai" else [d for d in R if d["tap"] == tap]
    print(f"\n===== {tap} =====")
    print(" ngưỡng | ngoài pv đúng | gọi bừa | rõ ràng đúng | từ chối nhầm | hỏi thừa | đổi kết quả (cũ->mới)")
    for t in (0.0, 0.8, 0.9, 0.95, 0.97):
        c = {n: collections.Counter(ket_qua_moi(d, t) for d in D if d["nhom"] == n) for n in ("ro_rang", "ngoai_pham_vi", "mo_ho")}
        no, rr = c["ngoai_pham_vi"], c["ro_rang"]
        nno, nrr = sum(no.values()), sum(rr.values())
        doi = collections.Counter((d["ket_qua"], ket_qua_moi(d, t)) for d in D if ket_qua_moi(d, t) != d["ket_qua"])
        print(f"  {t:4.2f}  | {no['dung']:4d}/{nno:4d}    | {no['goi_bua']:3d}     | {rr['dung']:4d}/{nrr:4d}   | {rr['tu_choi_nham']:3d}          | {rr['hoi_lai_thua']:3d}      | {dict(doi)}")
