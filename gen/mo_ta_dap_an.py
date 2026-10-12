"""Thí nghiệm "mô tả hàm trong đáp án" — phần phương pháp Octopus v2 mình chưa làm (PLAN.md Phase 4).

Octopus v2 train với đáp án ``<nexa_i>(...)<nexa_end>`` + ``Function description: {mô tả}``, để model
học NGHĨA của từng functional token (nguyên văn: "incorporate the function descriptions into the training
dataset, enabling the model to learn the importance of these specialized tokens"). Lúc chạy thật model dừng ở
``<nexa_end>`` nên phần mô tả KHÔNG tốn độ trễ — nó chỉ có mặt lúc train.

Chỉ biến đổi dữ liệu có sẵn, không gọi teacher: mẫu `call`/`ask` được nối thêm mô tả (tiếng Việt, lấy từ
`schema/functions.yaml`) sau ``<nexa_end>``; mẫu trả lời lượt 2 giữ nguyên. Mọi thứ khác (mẫu, nhãn, thứ tự)
giữ y hệt để chênh lệch chỉ đến từ đúng biến này.
"""

from __future__ import annotations

import re
from typing import Any

from gen.records import IM_END
from schema.call_format import CallParseError, parse_calls
from schema.loader import CATALOG

TIEN_TO = "\n\nMô tả hàm: "


def mo_ta_cua(ten: str) -> str:
    return re.sub(r"\s+", " ", CATALOG.by_name[ten].description).strip()


def them_mo_ta(row: dict[str, Any]) -> dict[str, Any]:
    """Một mẫu -> mẫu có mô tả hàm sau <nexa_end>. Không phải call/ask hoặc không đọc được thì giữ nguyên."""
    if row["meta"]["type"] not in ("call", "ask"):
        return row
    comp = row["completion"]
    ket = CATALOG.nexa_end
    if ket not in comp:
        return row
    try:
        calls = parse_calls(comp)
    except CallParseError:
        return row
    ten: list[str] = []
    for c in calls:
        if c.name not in ten:
            ten.append(c.name)
    mo_ta = " ".join(f"{t}: {mo_ta_cua(t)}" for t in ten) if len(ten) > 1 else mo_ta_cua(ten[0])
    goi = comp.split(ket, 1)[0] + ket
    meta = {**row["meta"], "mo_ta_dap_an": True}
    return {**row, "completion": goi + TIEN_TO + mo_ta + IM_END, "meta": meta}


def them_mo_ta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [them_mo_ta(r) for r in rows]
