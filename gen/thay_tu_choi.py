"""Thay câu trả lời lượt 2 của nhóm NGOÀI PHẠM VI bằng bản sinh lại — local, miễn phí.

Bản cũ bị lỗi: teacher chỉ nhận kết quả `{"handled": true}` nên tưởng đã xử lý xong và
TRẢ LỜI THẲNG câu hỏi ngoài phạm vi (652/1128 mẫu). Bản mới sinh bằng prompt từ chối
riêng (gen/teacher_prompts.build_refusal_reply_prompt).

    python -m gen.thay_tu_choi --goc data/processed/full2/train_vi_sach.jsonl \\
                               --moi data/processed/tu_choi/tu_choi_reply.jsonl \\
                               --output data/processed/full2/train_vi_sach.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

NEGATIVE_FN = "irrelevant_function"


def la_reply_tu_choi(row: dict[str, Any]) -> bool:
    return row["meta"].get("type") == "reply" and row["meta"].get("fn") == NEGATIVE_FN


def thay(goc: list[dict[str, Any]], moi: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
    """Bỏ mọi mẫu reply của nhóm từ chối trong `goc`, nối `moi` vào.

    Trả về (rows kết quả, số mẫu cũ đã bỏ, số mẫu mới thêm).
    """
    giu = [r for r in goc if not la_reply_tu_choi(r)]
    them = [r for r in moi if la_reply_tu_choi(r)]
    return giu + them, len(goc) - len(giu), len(them)


def main() -> None:
    ap = argparse.ArgumentParser(description="Thay câu trả lời từ chối bằng bản sinh lại")
    ap.add_argument("--goc", required=True, type=Path)
    ap.add_argument("--moi", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    def doc(p: Path) -> list[dict[str, Any]]:
        with open(p, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    ket_qua, da_bo, da_them = thay(doc(args.goc), doc(args.moi))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in ket_qua:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"bỏ mẫu cũ : {da_bo}")
    print(f"thêm mẫu mới: {da_them}")
    print(f"tổng      : {len(ket_qua)} -> {args.output}")


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
