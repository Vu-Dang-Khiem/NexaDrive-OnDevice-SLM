"""Mốc "if-else": chấm bộ luật regex của model giả demo (serve/mo_hinh.py MoHinhGia) trên tập test đóng băng.
Không đổi gì trong test / train — chỉ đo thêm một mốc so sánh."""
import json
import sys

from eval.metrics import cham_toan_bo
from serve.mo_hinh import MoHinhGia

sys.stdout.reconfigure(encoding="utf-8")
m = MoHinhGia()
for tep in ("test_vi_sach_v2", "test_vi_mo_ho_4"):
    rows = [json.loads(d) for d in open(f"data/processed/final/{tep}.jsonl", encoding="utf-8")]
    du_doan = {}
    for r in rows:
        cau = r["prompt"].split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
        du_doan[r["prompt"]] = m.goi_ham(cau)
    kq = cham_toan_bo(rows, du_doan)
    print(tep, kq["so_cau_da_cham"], json.dumps(kq["theo_nhom"], ensure_ascii=False))
    print("  lỗi:", kq["loi_theo_kieu"])
