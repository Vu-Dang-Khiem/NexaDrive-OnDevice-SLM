"""Áp serve.sua_tu_khoa lên chữ STT đã lưu -> file mới cùng định dạng (chấm bằng eval.stt.cham_chuoi).
PLAN.md "cách (b)".

    conda run -n vdk1 python -m eval.stt.ap_sua_tu_khoa <vào.jsonl> <ra.jsonl>
"""
import json
import sys

from serve.so_chu import doi_so
from serve.sua_tu_khoa import sua

vao, ra = sys.argv[1:3]
n = 0
with open(ra, "w", encoding="utf-8") as f:
    for x in open(vao, encoding="utf-8-sig"):
        r = json.loads(x)
        moi = sua(r["chu_stt"])
        n += moi != r["chu_stt"]
        f.write(json.dumps({**r, "stt": r["stt"] + "+sua", "chu_stt": moi, "chu_doi_so": doi_so(moi)},
                           ensure_ascii=False) + "\n")
print(f"{vao}: sửa {n} câu")
