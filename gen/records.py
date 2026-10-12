"""Ghép các mảnh rời (câu hỏi, tham số, kết quả, câu trả lời) thành đúng dòng
JSONL để train — logic thuần, test được ở local, không cần GPU.

Mỗi mẫu train gồm 2 trường:
    prompt     — phần model NHÌN THẤY, kết thúc ở "<|im_start|>assistant\\n"
    completion — phần model phải SINH RA, kết thúc bằng "<|im_end|>" để học cách dừng

Ba loại mẫu:
    type="call"  — Prompt A -> "<nexa_i>(...)<nexa_end><|im_end|>"
    type="reply" — Prompt B -> "Đã bật điều hoà 22 độ.<|im_end|>"
    type="ask"   — Prompt A, câu ngầm hiểu -> "<nexa_ask><nexa_i>(...)[;<nexa_j>(...)]<nexa_end><|im_end|>"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from schema.call_format import render_completion
from schema.loader import CATALOG

IM_END = "<|im_end|>"



@dataclass(frozen=True)
class CallRecord:
    """Mẫu loại A — dạy model sinh lời gọi hàm từ câu hỏi người dùng."""

    query: str
    fn_name: str
    args: dict[str, Any] = field(default_factory=dict)
    lang: str = "vi"

    def call_string(self) -> str:
        return render_completion([(self.fn_name, self.args)])

    def to_row(self) -> dict[str, Any]:
        return {
            "prompt": CATALOG.render_call_prompt(self.query),
            "completion": self.call_string() + IM_END,
            "meta": {"lang": self.lang, "type": "call", "fn": self.fn_name},
        }


@dataclass(frozen=True)
class ReplyRecord:
    """Mẫu loại B — dạy model viết câu trả lời tự nhiên từ kết quả hàm."""

    query: str
    fn_name: str
    args: dict[str, Any]
    result: dict[str, Any]
    reply: str
    lang: str = "vi"

    def to_row(self) -> dict[str, Any]:
        call_str = render_completion([(self.fn_name, self.args)])
        return {
            "prompt": CATALOG.render_reply_prompt(self.query, call_str, self.result),
            "completion": self.reply.strip() + IM_END,
            "meta": {"lang": self.lang, "type": "reply", "fn": self.fn_name},
        }


@dataclass(frozen=True)
class SuggestRecord:
    """Mẫu loại "ask" — câu NGẦM HIỂU ("Ôi nóng quá"): model ĐỀ XUẤT 1–2 lời gọi,
    runtime hỏi lại user ("Bật điều hoà nhé?") rồi mới thực thi.

    Dùng cùng Prompt A với mẫu call — model phải tự phân biệt câu nào làm luôn,
    câu nào chỉ đề xuất, dựa vào nội dung câu nói."""

    query: str
    calls: tuple[tuple[str, dict[str, Any]], ...]
    lang: str = "vi"

    def __post_init__(self) -> None:
        if not self.calls:
            raise ValueError("mẫu đề xuất phải có ít nhất 1 lời gọi")

    def to_row(self) -> dict[str, Any]:
        return {
            "prompt": CATALOG.render_call_prompt(self.query),
            "completion": render_completion(list(self.calls), ask=True) + IM_END,
            "meta": {
                "lang": self.lang,
                "type": "ask",
                "fn": self.calls[0][0],  # hàm đề xuất đầu tiên — để datasheet/subsample nhóm được
                "fns": [name for name, _ in self.calls],
            },
        }


def negative_record(query: str, lang: str = "vi") -> CallRecord:
    """Mẫu ngoài phạm vi — nhãn luôn là `irrelevant_function()`, không tham số."""
    return CallRecord(query=query, fn_name="irrelevant_function", args={}, lang=lang)


_USER_BLOCK = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.DOTALL)


def query_from_call_prompt(prompt: str) -> str:
    """Lấy lại câu người dùng từ Prompt A đã render (để chấm lại mẫu cũ trong dataset)."""
    m = _USER_BLOCK.search(prompt)
    if not m:
        raise ValueError(f"không tìm thấy khối user trong prompt: {prompt[:80]!r}")
    return m.group(1).strip()


_TOOL_BLOCK = re.compile(r"<\|im_start\|>tool\n(.*?)<\|im_end\|>", re.DOTALL)


def result_from_reply_prompt(prompt: str) -> dict[str, Any]:
    """Lấy lại kết quả hàm (khối `tool`) từ Prompt B đã render.

    Cần khi chấm lại mẫu loại reply: muốn hỏi "câu trả lời có bịa số không" thì phải
    đưa kèm số liệu gốc mà hàm trả về.
    """
    m = _TOOL_BLOCK.search(prompt)
    if not m:
        raise ValueError(f"không tìm thấy khối tool trong prompt: {prompt[:80]!r}")
    import json

    return json.loads(m.group(1).strip())


def datasheet(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Thống kê nhanh dataset — đếm theo loại / hàm / ngôn ngữ, để kiểm tra cân bằng
    trước khi đem train (tránh trường hợp 1 hàm có 900 mẫu, hàm khác chỉ có 3)."""
    theo_loai: dict[str, int] = {}
    theo_ham: dict[str, int] = {}
    theo_ngon_ngu: dict[str, int] = {}
    for r in rows:
        m = r["meta"]
        theo_loai[m["type"]] = theo_loai.get(m["type"], 0) + 1
        theo_ham[m["fn"]] = theo_ham.get(m["fn"], 0) + 1
        theo_ngon_ngu[m["lang"]] = theo_ngon_ngu.get(m["lang"], 0) + 1
    thieu = [f.name for f in CATALOG.functions if f.name not in theo_ham]
    return {
        "tong": len(rows),
        "theo_loai": dict(sorted(theo_loai.items())),
        "theo_ngon_ngu": dict(sorted(theo_ngon_ngu.items())),
        "theo_ham": dict(sorted(theo_ham.items())),
        "ham_chua_co_mau": thieu,
        "ti_le_reply": round(theo_loai.get("reply", 0) / max(len(rows), 1), 3),
    }
