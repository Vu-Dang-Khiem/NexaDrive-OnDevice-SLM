"""Bộ C (giọng THẬT, câu MỚI — PLAN.md Phase 6 "KẾT QUẢ BỘ B"): kiểm KHÁCH QUAN bản sửa đổi số
(`serve/so_chu.py`) vì bộ B đã dùng để tìm lỗi. 30 câu KHÔNG trùng bộ B, xáo seed 2028, bốc xong là khoá:
  • 12 câu rõ ràng CÓ SỐ (chữ số hoặc số viết bằng chữ) — 12 câu đầu sau khi xáo, trong 21 câu có số của
    test sạch còn lại + test v6 một lượt;
  • 8 câu rõ ràng không số (4 test sạch + 4 test v6) · 5 mơ hồ · 5 ngoài phạm vi (test sạch).
Nguồn đều là tập đánh giá đóng băng (không train), đáp án lấy nguyên từ đó.

    conda run -n vdk1 python -m eval.stt.chon_bo_c
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

from eval.metrics import doc_output, nhom_cua_dap_an
from gen.records import query_from_call_prompt
from serve.so_chu import doi_so

NGUON = {"test_sach": Path("data/processed/final/test_vi_sach_v2.jsonl"),
         "test_v6": Path("data/processed/final/test_v6_mot_luot.jsonl")}
BO_B = Path("data/stt/bo_b/danh_sach.jsonl")
RA = Path("data/stt/bo_c/danh_sach.jsonl")


def co_so(cau: str) -> bool:
    return bool(re.search(r"\d", doi_so(cau)))


def main() -> None:
    if RA.exists():
        raise SystemExit(f"{RA} đã có — danh sách bộ C đã khoá, không bốc lại.")
    da_b = {json.loads(x)["cau"] for x in BO_B.open(encoding="utf-8")}
    rows = []
    for ten, p in NGUON.items():
        for x in p.open(encoding="utf-8"):
            r = json.loads(x)
            r["cau"] = query_from_call_prompt(r["prompt"])
            if r["cau"] in da_b:
                continue
            r["nguon"], r["nhom"] = ten, nhom_cua_dap_an(doc_output(r["completion"]))
            rows.append(r)
    random.Random(2028).shuffle(rows)
    rr = [r for r in rows if r["nhom"] == "ro_rang"]
    so = [r for r in rr if co_so(r["cau"])][:12]
    khong_so = [r for r in rr if not co_so(r["cau"])]
    chon = (so
            + [r for r in khong_so if r["nguon"] == "test_sach"][:4]
            + [r for r in khong_so if r["nguon"] == "test_v6"][:4]
            + [r for r in rows if r["nhom"] == "mo_ho"][:5]
            + [r for r in rows if r["nhom"] == "ngoai_pham_vi"][:5])
    RA.parent.mkdir(parents=True, exist_ok=True)
    with RA.open("w", encoding="utf-8") as f:
        for i, r in enumerate(chon):
            dong = {"id": f"c{i:02d}", "tep": f"c{i:02d}.wav", "cau": r["cau"], "nhom": r["nhom"],
                    "nguon": r["nguon"], "co_so": co_so(r["cau"]), "prompt": r["prompt"],
                    "completion": r["completion"]}
            f.write(json.dumps(dong, ensure_ascii=False) + "\n")
    print(f"{len(chon)} câu ({sum(co_so(r['cau']) for r in rr)} câu rõ ràng có số trong nguồn) -> {RA}")
    for i, r in enumerate(chon):
        print(f"c{i:02d} [{r['nhom']}{', số' if co_so(r['cau']) else ''}] {r['cau']}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
