"""Bộ D (giọng THẬT, câu MỚI — PLAN.md Phase 6 "cải thiện STT: cách 1 + 2"): đo KHÁCH QUAN gợi ý từ vựng +
beam search, vì bộ B / C đã dùng để soi lỗi STT. 30 câu KHÔNG trùng bộ B và C, xáo seed 2029, bốc xong là
khoá.
Tỉ lệ nhóm theo bộ B (bộ đại diện, 20/10/10 trên 40): 15 rõ ràng (tối đa 7 câu có số) · 8 mơ hồ · 7 ngoài pv.
Nguồn: test sạch + test v6 một lượt (đều đóng băng, không train).

    conda run -n vdk1 python -m eval.stt.chon_bo_d
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from eval.metrics import doc_output, nhom_cua_dap_an
from eval.stt.chon_bo_c import NGUON, co_so
from gen.records import query_from_call_prompt

DA_CO = [Path("data/stt/bo_b/danh_sach.jsonl"), Path("data/stt/bo_c/danh_sach.jsonl")]
RA = Path("data/stt/bo_d/danh_sach.jsonl")


def main() -> None:
    if RA.exists():
        raise SystemExit(f"{RA} đã có — danh sách bộ D đã khoá, không bốc lại.")
    da_dung = {json.loads(x)["cau"] for p in DA_CO for x in p.open(encoding="utf-8")}
    rows = []
    for ten, p in NGUON.items():
        for x in p.open(encoding="utf-8"):
            r = json.loads(x)
            r["cau"] = query_from_call_prompt(r["prompt"])
            if r["cau"] in da_dung:
                continue
            r["nguon"], r["nhom"] = ten, nhom_cua_dap_an(doc_output(r["completion"]))
            rows.append(r)
    random.Random(2029).shuffle(rows)
    rr = [r for r in rows if r["nhom"] == "ro_rang"]
    so = [r for r in rr if co_so(r["cau"])][:7]
    chon = (so + [r for r in rr if not co_so(r["cau"])][: 15 - len(so)]
            + [r for r in rows if r["nhom"] == "mo_ho"][:8]
            + [r for r in rows if r["nhom"] == "ngoai_pham_vi"][:7])
    RA.parent.mkdir(parents=True, exist_ok=True)
    with RA.open("w", encoding="utf-8") as f:
        for i, r in enumerate(chon):
            dong = {"id": f"d{i:02d}", "tep": f"d{i:02d}.wav", "cau": r["cau"], "nhom": r["nhom"],
                    "nguon": r["nguon"], "co_so": co_so(r["cau"]), "prompt": r["prompt"],
                    "completion": r["completion"]}
            f.write(json.dumps(dong, ensure_ascii=False) + "\n")
    print(f"{len(chon)} câu -> {RA}")
    for i, r in enumerate(chon):
        print(f"d{i:02d} [{r['nhom']}{', số' if co_so(r['cau']) else ''}] {r['cau']}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
