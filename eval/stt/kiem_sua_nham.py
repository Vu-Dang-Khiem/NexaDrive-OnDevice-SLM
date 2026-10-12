"""Kiểm SỬA NHẦM của serve.sua_tu_khoa trên câu VIẾT ĐÚNG (train + val v6.1 — không đụng bộ ghi âm / test):
câu nào bị đổi là sửa nhầm. PLAN.md "cách (b)".

    conda run -n vdk1 python -m eval.stt.kiem_sua_nham
"""
import json
import sys
import unicodedata
from collections import Counter

from gen.records import query_from_call_prompt
from serve.sua_tu_khoa import sua

sys.stdout.reconfigure(encoding="utf-8")
cau = []
for t in ("train_vi", "val_vi"):
    for x in open(f"data/processed/final_v6_1/{t}.jsonl", encoding="utf-8"):
        q = query_from_call_prompt(json.loads(x)["prompt"])
        cau.append(q.split("\n")[-1])  # bỏ dòng "Lượt trước: …" (chỉ xét câu người nói)
cau = list(dict.fromkeys(cau))
# STT luôn ra chữ CÓ dấu -> chỉ xét câu có dấu (train có ~15% câu cố ý bỏ dấu: module chỉ thêm lại dấu)
def co_dau(s: str) -> bool:
    return "đ" in s.lower() or any(unicodedata.category(c) == "Mn" for c in unicodedata.normalize("NFD", s))


tat_ca = len(cau)
cau = [c for c in cau if co_dau(c)]
print(f"bỏ {tat_ca - len(cau)} câu không dấu")
doi = [(c, sua(c)) for c in cau if sua(c) != c]
print(f"{len(cau)} câu khác nhau · bị sửa {len(doi)} ({100 * len(doi) / len(cau):.2f}%)")
cap = Counter()
for a, b in doi:
    cap[" | ".join(f"{x}->{y}" for x, y in zip(a.split(), b.split(), strict=False) if x != y)] += 1
for k, v in cap.most_common(25):
    print(f"  {v:4d}  {k}")
