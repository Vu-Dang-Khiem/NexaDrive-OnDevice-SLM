"""Áp QUY ƯỚC gán nhãn lên output model lúc CHẠY (PLAN.md Phase 7, nhóm A — lỗi người dùng gặp trên demo).

Dữ liệu v6 đã được ép theo quy ước bằng code (`gen/v6.ap_quy_uoc`): nhiệt độ không nói số -> ±2, quạt -> ±1,
âm lượng -> ±3, "đèn pha" -> chiếu xa… Model học gần đúng nhưng đôi khi lệch (đo 2026-10-06: [đã giảm 2 độ]
"nữa" -> delta=-1). Áp lại ĐÚNG hàm đã dùng lúc gán nhãn -> runtime nhất quán với dữ liệu train; hàm báo mâu
thuẫn (None) thì giữ nguyên lời gọi của model.

Thêm một quy ước chỉ có ở runtime: "hạ 2 kính ghế sau" — số đứng trước "kính" là SỐ LƯỢNG kính, không phải
phần trăm (model ra amount=2 -> hạ 2%).
"""

from __future__ import annotations

import re
from typing import Any

from gen.v6 import ap_quy_uoc

_SO_LUONG_KINH = re.compile(r"\b(\d+|hai|ba|bốn|cả hai)\s+(cái\s+)?(kính|cửa kính|cửa sổ)\b")
_CO_MUC = re.compile(r"%|phần trăm|\bnửa\b|\bhé\b|\bchút\b|\btí\b|\bxíu\b")


def ap_quy_uoc_chay(ten: str, args: dict[str, Any], cau: str, de_xuat: bool = False) -> dict[str, Any]:
    """`de_xuat`: model chỉ ĐỀ XUẤT (hỏi lại) cho câu than phiền -> KHÔNG áp quy ước đèn pha / cốt: câu than
    phiền nhắc "xa" để tả VẤN ĐỀ ("đèn rọi xa quá làm người ta chói") — áp luật chữ sẽ đổi đề xuất chiếu gần
    thành chiếu xa (ngược hẳn; lộ ra khi kiểm nhãn tập test_grpo 2026-10-07). Quy ước đèn chỉ cho lệnh."""
    if de_xuat and ten == "set_headlights":
        return dict(args)
    moi = ap_quy_uoc(ten, cau, args)
    a = dict(args if moi is None else moi)
    c = cau.lower()
    if ten == "control_window" and "amount" in a and _SO_LUONG_KINH.search(c) and not _CO_MUC.search(c):
        a.pop("amount")  # "hạ 2 kính" = hạ hẳn hai kính (mặc định 100%), không phải hạ 2%
    return a
