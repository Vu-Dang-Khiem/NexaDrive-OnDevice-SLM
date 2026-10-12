"""Áp chính sách đề xuất lên nhóm mẫu "ask" ĐÃ SINH — chạy local, miễn phí.

Vì sao cần: chính sách (ví dụ "đừng gợi ý đèn pha xa") được chốt SAU khi đã tốn tiền
GPU sinh data. Thay vì sinh lại, chỉ cần sửa nhãn — câu người dùng giữ nguyên, chỉ đổi
giá trị tham số trong lời gọi được đề xuất.

Dùng chung đúng một hàm chính sách `ap_dung_chinh_sach_de_xuat` với pipeline sinh data,
nên data cũ và data mới không bao giờ lệch nhau.

    python -m gen.don_ask --input data/processed/ask/ask_train_vi.jsonl \\
                          --output data/processed/ask/ask_train_vi_sach.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gen.records import CallRecord, SuggestRecord, query_from_call_prompt
from gen.teacher_prompts import ap_dung_chinh_sach_de_xuat, de_xuat_hop_le, tham_so_duoc_noi_ro
from schema.call_format import parse_calls
from schema.loader import CATALOG


def don_row(row: dict[str, Any]) -> tuple[dict[str, Any], bool] | tuple[None, bool]:
    """Trả về (row đã áp chính sách, có thay đổi gì không). Row bị loại -> (None, True)."""
    query = query_from_call_prompt(row["prompt"])
    calls = []
    for c in parse_calls(row["completion"]):
        # BỔ SUNG lại tham số câu nói rõ nhưng đã bị quy ước cũ cắt mất
        # ("Kính sau đọng hơi" -> set_defrost(state="on") -> thiếu zone="rear").
        args = {**c.args, **tham_so_duoc_noi_ro(c.name, query)}
        calls.append((c.name, ap_dung_chinh_sach_de_xuat(c.name, args)))
    calls = [(n, a) for n, a in calls if de_xuat_hop_le(n, a, query)]
    if not calls:  # hết đề xuất hợp lệ -> bỏ cả mẫu
        return None, True

    # Luật "chỉ đọc không hỏi lại" (xem gen/teacher_prompts.py): mẫu ask cũ mà toàn hàm
    # tra cứu thì chuyển thành lệnh thường — đọc số liệu không có gì để hỏi xác nhận.
    if all(CATALOG.by_name[n].chi_doc for n, _ in calls):
        ten, args = calls[0]
        moi = {**row, **CallRecord(query, ten, args, row["meta"].get("lang", "vi")).to_row()}
        return moi, True

    moi = SuggestRecord(
        query=query,
        calls=tuple(calls),
        lang=row["meta"].get("lang", "vi"),
    ).to_row()
    # giữ lại các khoá phụ (vd "nguon") mà pipeline sinh data đính kèm
    moi = {**row, **moi}
    return moi, moi["completion"] != row["completion"]


def don_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
    """Trả về (rows đã dọn, số nhãn bị sửa, số mẫu bị loại)."""
    ket_qua: list[dict[str, Any]] = []
    n_doi = n_bo = 0
    for r in rows:
        moi, doi = don_row(r)
        if moi is None:
            n_bo += 1
            continue
        ket_qua.append(moi)
        n_doi += int(doi)
    return ket_qua, n_doi, n_bo


def main() -> None:
    ap = argparse.ArgumentParser(description="Áp chính sách đề xuất lên mẫu ask đã sinh")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    moi, n_doi, n_bo = don_rows(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in moi:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"vào     : {len(rows)} mẫu")
    print(f"đã sửa  : {n_doi} nhãn")
    print(f"đã loại : {n_bo} mẫu (đề xuất vô lý)")
    print(f"còn lại : {len(moi)} mẫu")
    print(f"ghi vào : {args.output}")


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
