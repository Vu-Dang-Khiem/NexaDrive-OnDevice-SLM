import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")
from eval.metrics import cham_toan_bo

d = [json.loads(x) for x in open("data/processed/final_v3/val_mau.jsonl", encoding="utf-8")]
CK = {"v4dc": (1103, 2206, 3309)}
for t, cks in CK.items():
    for sd in (42, 7):
        tot = None
        for ep, ck in enumerate(cks, 1):
            du = {r["prompt"]: r["output"] for r in map(json.loads, open(f"runs/chon_checkpoint_v4dc/{t}_s{sd}_ck{ck}.jsonl", encoding="utf-8"))}
            b = cham_toan_bo(d, du)
            n, e = b["theo_nhom"], b["ty_le_loi_%"]
            rr = n["ro_rang"]["dung_%"]
            ok = e["tu_choi_nham"] <= 5
            if ok and (tot is None or rr > tot[1]):
                tot = (ck, rr)
            print(f"{t} s{sd} ep{ep} ck{ck}: " + " | ".join(f"{k} {v['dung_%']}" for k, v in sorted(n.items()))
                  + f" | tcn {e['tu_choi_nham']} | goi_bua {e['goi_bua']}")
        print(f"   -> CHỌN {t} s{sd}: checkpoint-{tot[0]}")

