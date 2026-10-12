"""Tách 74 câu RÕ RÀNG: làm đúng ngay / hỏi lại (đề xuất đúng hàm hay không) / làm sai / từ chối nhầm — không ngưỡng vs 0,85."""
import json
import sys
from collections import Counter

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from schema.loader import CATALOG

TU_CHOI = CATALOG.by_name["irrelevant_function"].nexa_token
ASK = CATALOG.nexa_ask

for s in ("s42", "s7"):
    R = [json.loads(x) for x in open(f"runs/tu_tin/test_lorav5_{s}.jsonl", encoding="utf-8")]
    R = [d for d in R if d["tap"] == "test_vi_sach_v2" and d["nhom"] == "ro_rang"]
    for t in (0.0, 0.85):
        c = Counter()
        for d in R:
            (t1, p1), (t2, _) = d["top_buoc_1"][0], d["top_buoc_1"][1]
            if t1 in (TU_CHOI, ASK) or p1 >= t:
                k = d["ket_qua"]
                c["đúng ngay" if k == "dung" else "hỏi lại (model tự hỏi)" if k == "hoi_lai_thua"
                  else "từ chối nhầm" if k == "tu_choi_nham" else "LÀM SAI"] += 1
            elif t2 == TU_CHOI:
                c["từ chối nhầm"] += 1
            else:
                c["hỏi lại, đề xuất ĐÚNG việc" if d["ket_qua"] == "dung" else "hỏi lại, đề xuất sai/thiếu"] += 1
        print(f"{s} ngưỡng {t or 'không'}: n={len(R)}", dict(c))
