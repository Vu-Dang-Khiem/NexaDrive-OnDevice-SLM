"""Tách runs/stt/duong_mic/ket_qua.jsonl (eval/stt/thu_duong_mic.py) thành 2 file định dạng
`eval.stt.cham_chuoi` (thang.jsonl, mic_demo.jsonl) để chấm CẢ CHUỖI bằng bản triển khai.
`ms` = 0 (không đo độ trễ STT ở đây).

    python -m eval.stt.tach_duong_mic
    python -m eval.stt.cham_chuoi runs/stt/duong_mic/thang.jsonl runs/stt/duong_mic/mic_demo.jsonl \
        --gguf=models/gguf/lora-v61-s42-q4_k_m.gguf
"""
import json
from pathlib import Path

from serve.so_chu import doi_so

D = Path("runs/stt/duong_mic")
goc = {}
for bo in ("bo_c", "bo_d"):
    for x in open(f"data/stt/{bo}/danh_sach.jsonl", encoding="utf-8"):
        r = json.loads(x)
        goc[r["id"]] = r
kq = [json.loads(x) for x in open(D / "ket_qua.jsonl", encoding="utf-8")]
assert len(kq) == 60, len(kq)
for ten, khoa in (("thang", "thang"), ("mic_demo", "mic_demo")):
    with open(D / f"{ten}.jsonl", "w", encoding="utf-8") as f:
        for d in kq:
            r, chu = goc[d["id"]], d[khoa] or ""
            dong = {"id": d["id"], "cau": r["cau"], "prompt": r["prompt"], "completion": r["completion"],
                    "stt": ten, "chu_stt": chu, "chu_doi_so": doi_so(chu), "ms": 0,
                    "giay_am_thanh": d["giay"]}
            f.write(json.dumps(dong, ensure_ascii=False) + "\n")
print("đã tách 2 file")
