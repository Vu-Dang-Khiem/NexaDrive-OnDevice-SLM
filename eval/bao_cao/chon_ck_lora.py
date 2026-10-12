"""Chọn checkpoint LoRA theo LUẬT CŨ (PLAN.md): rõ ràng trên val mẫu cao nhất, từ chối nhầm ≤ 5%; bằng nhau -> epoch sớm."""
import json
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from eval.metrics import cham_toan_bo

val = [json.loads(x) for x in open("data/processed/final_v5/val_mau.jsonl", encoding="utf-8")]
for s in ("s42", "s7"):
    kq = []
    for ck in ("1208", "2416", "3624"):
        p = f"runs/chon_checkpoint_v5/du_doan_lorav5{s}ck{ck}_val_mau.jsonl"
        du = {json.loads(x)["prompt"]: json.loads(x)["output"] for x in open(p, encoding="utf-8")}
        a = cham_toan_bo(val, du)
        rr, mh = a["theo_nhom"]["ro_rang"]["dung_%"], a["theo_nhom"]["mo_ho"]["dung_%"]
        tcn = a["ty_le_loi_%"]["tu_choi_nham"]
        kq.append((ck, rr, mh, tcn))
        print(f"LoRA {s} ck{ck}: rõ ràng {rr:5.1f} | mơ hồ {mh:5.1f} | từ chối nhầm {tcn:4.1f}")
    hop = [k for k in kq if k[3] <= 5]
    best = max(k[1] for k in hop)
    print(f"  -> CHỌN LoRA {s} checkpoint-{next(k for k in hop if k[1] == best)[0]}\n")
