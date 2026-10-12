"""Bộ B (giọng THẬT, dùng để CHỐT STT — PLAN.md Phase 6 (3)): bốc 40 câu từ test sạch, xáo seed 2027,
phân tầng 20 rõ ràng (10 câu CÓ chữ số + 10 câu không — thử phần đổi số) · 10 mơ hồ · 10 ngoài phạm vi.
Bốc xong là khoá.

    conda run -n vdk1 python -m eval.stt.chon_bo_b
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

from eval.metrics import doc_output, nhom_cua_dap_an
from gen.records import query_from_call_prompt

VAO = Path("data/processed/final/test_vi_sach_v2.jsonl")
RA = Path("data/stt/bo_b/danh_sach.jsonl")


def main() -> None:
    if RA.exists():
        raise SystemExit(f"{RA} đã có — danh sách bộ B đã khoá, không bốc lại.")
    rows = [json.loads(x) for x in VAO.open(encoding="utf-8")]
    for r in rows:
        r["cau"] = query_from_call_prompt(r["prompt"])
        r["nhom"] = nhom_cua_dap_an(doc_output(r["completion"]))
    random.Random(2027).shuffle(rows)
    rr = [r for r in rows if r["nhom"] == "ro_rang"]
    co_so = [r for r in rr if re.search(r"\d", r["cau"])]
    khong_so = [r for r in rr if not re.search(r"\d", r["cau"])]
    chon = (co_so[:10] + khong_so[: 20 - min(10, len(co_so))]
            + [r for r in rows if r["nhom"] == "mo_ho"][:10]
            + [r for r in rows if r["nhom"] == "ngoai_pham_vi"][:10])
    RA.parent.mkdir(parents=True, exist_ok=True)
    with RA.open("w", encoding="utf-8") as f:
        for i, r in enumerate(chon):
            dong = {"id": f"b{i:02d}", "tep": f"b{i:02d}.wav", "cau": r["cau"], "nhom": r["nhom"],
                    "prompt": r["prompt"], "completion": r["completion"]}
            f.write(json.dumps(dong, ensure_ascii=False) + "\n")
    print(f"{len(chon)} câu ({len(co_so)} câu rõ ràng có chữ số trong test) -> {RA}")
    for i, r in enumerate(chon):
        print(f"b{i:02d} [{r['nhom']}] {r['cau']}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
