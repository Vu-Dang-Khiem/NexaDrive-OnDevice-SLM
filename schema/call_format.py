"""Render và parse chuỗi lời gọi hàm dạng functional token (Octopus v2).

Định dạng này là hợp đồng dùng chung cho 3 giai đoạn:
  • sinh dữ liệu   — target lúc train được tạo bằng `render_completion`
  • huấn luyện     — model học sinh ra đúng chuỗi này
  • lúc chạy thật  — `parse_calls` biến output của model về dạng có cấu trúc

Văn phạm (single / parallel; nested chấp nhận parse được nhưng ta không sinh ra):

    [<nexa_ask>] <nexa_i>(arg, name=value, ...) [ ; <nexa_j>(...) ] <nexa_end>

`<nexa_ask>` ở đầu = các lời gọi chỉ là ĐỀ XUẤT cho câu mơ hồ ("Ôi nóng quá"),
runtime phải hỏi lại user trước khi thực thi (`ParsedCall.suggested = True`).

Giá trị: số để trần, bool là true/false, null, chuỗi trong nháy kép.
Tham số truyền theo vị trí được map về tên theo đúng thứ tự khai báo trong catalog.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from schema.loader import CATALOG

_NEXA_HEAD = re.compile(r"^<nexa_(\d+)>\((.*)\)$", re.DOTALL)
_NESTED_CALL = re.compile(r"^<nexa_\d+>\(.*\)$", re.DOTALL)
_IDENT = re.compile(r"^[A-Za-z_]\w*$")


class CallParseError(ValueError):
    """Ném ra khi output của model không parse được thành lời gọi hàm."""


@dataclass
class ParsedCall:
    name: str
    args: dict[str, Any]
    index: int
    raw: str
    warnings: list[str] = field(default_factory=list)
    suggested: bool = False  # True nếu output mở đầu bằng <nexa_ask> -> hỏi lại, chưa làm

    @property
    def nexa_token(self) -> str:
        return f"<nexa_{self.index}>"


# ── render ──────────────────────────────────────────────────────────────────
def _render_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (int, float)):
        return repr(v)
    escaped = str(v).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def render_call(name: str, args: dict[str, Any] | None = None) -> str:
    """Render một lời gọi: ``<nexa_i>(name=value, ...)`` (chưa có token kết thúc)."""
    spec = CATALOG.by_name.get(name)
    if spec is None:
        raise KeyError(f"không có hàm tên {name!r}")
    args = args or {}
    order = [p.name for p in spec.params]
    unknown = [k for k in args if k not in order]
    if unknown:
        raise KeyError(f"{name}: tham số lạ {unknown}")
    parts = [f"{k}={_render_value(args[k])}" for k in order if k in args]
    return f"{spec.nexa_token}({', '.join(parts)})"


def render_readable(name: str, args: dict[str, Any] | None = None) -> str:
    """Render lời gọi ở dạng NGƯỜI/TEACHER ĐỌC ĐƯỢC: ``set_ac(state="on", temperature=22)``.

    Dùng khi cần đưa lời gọi cho teacher xem (vd bước chấm ở Phase 2) — teacher
    chưa từng thấy `<nexa_i>` nên nhìn token đó không hiểu gì, đặc biệt với hàm
    không có tham số (`<nexa_0>()` trông hoàn toàn vô nghĩa) -> chấm sai hàng loạt.
    """
    spec = CATALOG.by_name.get(name)
    if spec is None:
        raise KeyError(f"không có hàm tên {name!r}")
    args = args or {}
    order = [p.name for p in spec.params]
    parts = [f"{k}={_render_value(args[k])}" for k in order if k in args]
    return f"{name}({', '.join(parts)})"


_READABLE_HEAD = re.compile(r"^([A-Za-z_]\w*)\((.*)\)$", re.DOTALL)


def parse_readable(text: str) -> list[tuple[str, dict[str, Any]]]:
    """Đọc lời gọi dạng NGƯỜI VIẾT: ``set_ac(state="on", temperature=26)``.

    Dùng cho tập test người viết tay: người gán nhãn viết tên hàm cho dễ đọc, không
    phải viết `<nexa_6>`. Nhiều lời gọi ngăn bằng `;` như định dạng chính.
    """
    ket_qua: list[tuple[str, dict[str, Any]]] = []
    for doan in _split_top_level(text.strip(), ";"):
        seg = doan.strip()
        if not seg:
            continue
        m = _READABLE_HEAD.match(seg)
        if not m:
            raise CallParseError(f"không phải lời gọi dạng tên_hàm(...): {seg!r}")
        name = m.group(1)
        spec = CATALOG.by_name.get(name)
        if spec is None:
            raise CallParseError(f"không có hàm tên {name!r}")
        order = [p.name for p in spec.params]
        args: dict[str, Any] = {}
        body = m.group(2).strip()
        if body:
            for pos, chunk in enumerate(_split_top_level(body, ",")):
                chunk = chunk.strip()
                if not chunk:
                    continue
                kv = _split_top_level(chunk, "=")
                if len(kv) >= 2 and _IDENT.match(kv[0].strip()):
                    key, val = kv[0].strip(), "=".join(kv[1:]).strip()
                elif pos < len(order):
                    key, val = order[pos], chunk
                else:
                    raise CallParseError(f"{name}: thừa tham số vị trí {chunk!r}")
                if key not in order:
                    raise CallParseError(f"{name}: tham số lạ {key!r} (có: {order})")
                args[key] = _coerce_scalar(val)
        ket_qua.append((name, args))
    if not ket_qua:
        raise CallParseError("chuỗi lời gọi rỗng")
    return ket_qua


def render_completion(calls: list[tuple[str, dict[str, Any]]], ask: bool = False) -> str:
    """Render target hoàn chỉnh lúc train: một hoặc nhiều lời gọi + token kết thúc.
    `ask=True` -> thêm `<nexa_ask>` ở đầu (lời gọi chỉ là đề xuất, cần hỏi lại)."""
    bodies = [render_call(name, args) for name, args in calls]
    prefix = CATALOG.nexa_ask if ask else ""
    return prefix + ";".join(bodies) + CATALOG.nexa_end


# ── parse ───────────────────────────────────────────────────────────────────
def _split_top_level(s: str, sep: str) -> list[str]:
    """Tách theo `sep` nhưng bỏ qua những chỗ nằm trong nháy hoặc trong ngoặc."""
    out: list[str] = []
    buf: list[str] = []
    depth = 0
    quote = ""
    i = 0
    while i < len(s):
        c = s[i]
        if quote:
            buf.append(c)
            if c == "\\" and i + 1 < len(s):
                buf.append(s[i + 1])
                i += 2
                continue
            if c == quote:
                quote = ""
        elif c in "\"'":
            quote = c
            buf.append(c)
        elif c == "(":
            depth += 1
            buf.append(c)
        elif c == ")":
            depth -= 1
            buf.append(c)
        elif c == sep and depth == 0:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(c)
        i += 1
    out.append("".join(buf))
    return out


def _coerce_scalar(token: str) -> Any:
    t = token.strip()
    if len(t) >= 2 and t[0] in "\"'" and t[-1] == t[0]:
        return t[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    low = t.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    if _NESTED_CALL.match(t):
        return t  # lời gọi lồng nhau — giữ nguyên chuỗi (tính năng stretch)
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        pass
    return t  # từ trần không nháy -> coi là chuỗi (giá trị enum, tên địa điểm...)


def _parse_one(segment: str) -> ParsedCall:
    seg = segment.strip()
    m = _NEXA_HEAD.match(seg)
    if not m:
        raise CallParseError(f"không phải lời gọi functional token: {seg!r}")
    idx = int(m.group(1))
    spec = CATALOG.by_index.get(idx)
    if spec is None:
        raise CallParseError(f"không có functional token <nexa_{idx}>")

    warnings: list[str] = []
    body = m.group(2).strip()
    args: dict[str, Any] = {}
    if body:
        order = [p.name for p in spec.params]
        for pos, chunk in enumerate(_split_top_level(body, ",")):
            chunk = chunk.strip()
            if not chunk:
                continue
            kv = _split_top_level(chunk, "=")
            if len(kv) >= 2 and _IDENT.match(kv[0].strip()):
                key = kv[0].strip()
                val = "=".join(kv[1:]).strip()
            elif pos < len(order):
                key, val = order[pos], chunk
            else:
                warnings.append(f"bỏ qua tham số vị trí thừa #{pos}: {chunk!r}")
                continue
            if key not in order:
                warnings.append(f"tham số lạ {key!r}, vẫn giữ nguyên")
            args[key] = _coerce_scalar(val)

    return ParsedCall(name=spec.name, args=args, index=idx, raw=seg, warnings=warnings)


def parse_calls(text: str) -> list[ParsedCall]:
    """Parse output thô của model thành một hoặc nhiều lời gọi hàm.

    Khoan dung với việc thiếu token kết thúc (giống cách paper chấm điểm nới lỏng)
    và với phần thừa phía sau; ghi lại sự nới lỏng đó vào ``warnings`` của mỗi call.
    """
    s = text.strip()
    # Bỏ đuôi ChatML nếu model sinh kèm (student là bản Instruct).
    s = s.split("<|im_end|>", 1)[0].strip()
    suggested = s.startswith(CATALOG.nexa_ask)
    if suggested:
        s = s[len(CATALOG.nexa_ask) :].strip()
    end = CATALOG.nexa_end
    missing_end = end not in s
    if not missing_end:
        s = s.split(end, 1)[0].strip()
    if not s:
        raise CallParseError("chuỗi lời gọi rỗng")
    calls = [_parse_one(seg) for seg in _split_top_level(s, ";") if seg.strip()]
    for c in calls:
        c.suggested = suggested
        if missing_end:
            c.warnings.append(f"thiếu {end}")
    return calls
