"""Chọn checkpoint v6.1 hạt giống 2026 theo LUẬT CŨ (giống s42 / s7): rõ ràng trên val mẫu cao nhất, từ chối nhầm ≤ 5%;
bằng nhau -> epoch sớm. Tải dự đoán từ volume (kiểm kích thước, tải lại tới khi đủ). PLAN.md "HẠT GIỐNG THỨ BA CHO v6.1".
"""
import json
import sys
from pathlib import Path

import modal

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from eval.metrics import cham_toan_bo  # noqa: E402

RA = Path("runs/v6/chon_ck_s2026")
RA.mkdir(parents=True, exist_ok=True)
VOL = "/data/vi/final_v6_1"
vol = modal.Volume.from_name("nexa-drive")
kt = {e.path.rsplit("/", 1)[-1]: e.size for e in vol.listdir(VOL)}
for ten in ["val_mau.jsonl"] + [f"du_doan_v61s2026ck{ck}_val_mau.jsonl" for ck in (2096, 4192, 6288)]:
    for _ in range(6):
        du = b"".join(vol.read_file(f"{VOL}/{ten}"))
        if len(du) == kt[ten]:
            break
    else:
        raise RuntimeError(f"tải {ten} không đủ")
    (RA / ten).write_bytes(du)

val = [json.loads(x) for x in open(RA / "val_mau.jsonl", encoding="utf-8")]
kq = []
for ck in ("2096", "4192", "6288"):
    du = {json.loads(x)["prompt"]: json.loads(x)["output"]
          for x in open(RA / f"du_doan_v61s2026ck{ck}_val_mau.jsonl", encoding="utf-8")}
    assert set(du) == {r["prompt"] for r in val}, ck  # val mẫu có 1 câu trùng (600 dòng / 599 câu) — như s42 / s7
    a = cham_toan_bo(val, du)
    rr, mh = a["theo_nhom"]["ro_rang"]["dung_%"], a["theo_nhom"]["mo_ho"]["dung_%"]
    tcn = a["ty_le_loi_%"]["tu_choi_nham"]
    kq.append((ck, rr, mh, tcn))
    print(f"v6.1 s2026 ck{ck}: rõ ràng {rr:5.1f} | mơ hồ {mh:5.1f} | từ chối nhầm {tcn:4.1f}")
hop = [k for k in kq if k[3] <= 5]
best = max(k[1] for k in hop)
print(f"  -> CHỌN checkpoint-{next(k for k in hop if k[1] == best)[0]}")
