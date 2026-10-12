"""Sửa nhãn CÓ HỆ THỐNG bằng code — dữ liệu vòng 3 (PLAN.md VÒNG 3).

Bốn lỗi do người kiểm 195 mẫu tìm ra (PLAN.md VÒNG 2 — K), mỗi lỗi một luật:

  1. `get_fuel_level` cho câu hỏi "đủ đi N km không" -> `get_range_estimate(trip_km=N)`.
     Câu hỏi tầm xa KHÔNG nhắc xăng ("Xe còn đi được bao xa nữa?") cũng thuộc
     `get_range_estimate` — chính câu đó là ví dụ mẫu của hàm này trong schema.
     Số km vô lý (> 2000, teacher sinh "đi thêm 5.000.000.000 km") -> bỏ mẫu.
  2. Quy ước âm lượng (người dùng chốt 2026-09-28):
       "lên/xuống/về N", "mức N"         -> set_value=N   (đặt về mức N)
       "thêm/bớt/đi N", "N nấc/mức/..."  -> delta=±N      (đổi N nấc)
     "Tăng âm lượng 5" (không có từ nào ở trên) để nguyên — không đủ căn cứ.
  3. `control_window`: "đóng" là NÂNG kính (`up`), "mở/hạ" là `down`. Câu vừa "đóng"
     vừa "xuống" ("đóng cửa sổ xuống 50%") tự mâu thuẫn -> bỏ mẫu.
  4. Từ chối câu mà `find_nearby` làm được ("tìm cây ATM gần đây") -> `find_nearby`.
     Giữ từ chối khi yêu cầu vượt quá tìm địa điểm: tự lái, đặt bàn, lịch chiếu, số
     điện thoại, giờ mở cửa...

  5. Cùng một câu mà hai nhãn khác nhau trong cùng tập -> bỏ mọi bản.

Mẫu TRẢ LỜI lượt 2 dựng trên lời gọi bị sửa/bỏ -> bỏ: kết quả hàm và câu trả lời trong
đó thuộc về lời gọi sai, không sửa được nếu không gọi lại teacher.

Chỉ đụng mẫu `call` và `reply`. Mẫu hỏi lại (`ask`) là câu ngầm hiểu, không có động từ
tường minh để luật dựa vào.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from gen.records import IM_END, query_from_call_prompt
from schema.call_format import CallParseError, parse_calls, render_completion

GIU, SUA, BO = "giu", "sua", "bo"
KM_TOI_DA = 2000

# ── 1. xăng / tầm xa ────────────────────────────────────────────────────────
_KM = re.compile(r"(\d[\d.,]*)\s*(?:km|kilô?m[eé]t|kilomet|cây số|cây)\b", re.IGNORECASE)
_NOI_XANG = re.compile(r"xăng|nhiên liệu", re.IGNORECASE)
_HOI_TAM_XA = re.compile(r"bao xa|bao nhiêu (?:km|cây|kilô?m[eé]t|kilomet)|quãng đường", re.IGNORECASE)


def _doc_km(chuoi: str) -> float:
    s = chuoi.rstrip(".,")
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", s):
        return float(re.sub(r"[.,]", "", s))
    return float(s.replace(",", "."))


def _so_gon(x: float) -> int | float:
    return int(x) if x == int(x) else x


def _sua_xang(cau: str, ten: str, args: dict[str, Any]) -> tuple[str, str, dict[str, Any], str]:
    m = _KM.search(cau)
    if ten == "get_range_estimate":
        trip = args.get("trip_km")
        if isinstance(trip, (int, float)) and trip > KM_TOI_DA:
            return BO, ten, args, "km_vo_ly"
        return GIU, ten, args, ""
    if m:
        km = _doc_km(m.group(1))
        if km > KM_TOI_DA:
            return BO, ten, args, "km_vo_ly"
        return SUA, "get_range_estimate", {"trip_km": _so_gon(km)}, "xang_hoi_km_chuyen_di"
    if not _NOI_XANG.search(cau) and _HOI_TAM_XA.search(cau):
        return SUA, "get_range_estimate", {}, "tam_xa_khong_noi_xang"
    return GIU, ten, args, ""


# ── 2. âm lượng ─────────────────────────────────────────────────────────────
_VOL_DON_VI = re.compile(r"(\d+)\s*(?:nấc|đơn vị|điểm|mức|bậc|vạch)")
_VOL_THEM = re.compile(r"\b(?:thêm|bớt|đi)\s+(\d+)\b")
_VOL_DAT = re.compile(r"\b(?:lên|xuống|về|tới|đến|ở)\s+(?:mức\s+)?(\d+)\b|\bmức\s+(\d+)\b")
_VOL_GIAM = re.compile(r"\b(?:giảm|bớt|nhỏ|hạ|xuống)\b")
_VOL_TANG = re.compile(r"\b(?:tăng|thêm|to|lớn|lên)\b")


def _dau(cau: str, args: dict[str, Any]) -> int | None:
    d = args.get("delta")
    if isinstance(d, (int, float)) and d != 0:
        return 1 if d > 0 else -1
    giam, tang = bool(_VOL_GIAM.search(cau)), bool(_VOL_TANG.search(cau))
    if giam != tang:
        return -1 if giam else 1
    return None


def _sua_am_luong(cau: str, ten: str, args: dict[str, Any]) -> tuple[str, str, dict[str, Any], str]:
    c = cau.lower()
    moi: dict[str, Any] | None = None
    m = _VOL_DON_VI.search(c) or _VOL_THEM.search(c)
    if m:
        dau = _dau(c, args)
        if dau is None:
            return GIU, ten, args, ""
        moi = {"delta": dau * int(m.group(1))}
    else:
        m = _VOL_DAT.search(c)
        if m:
            moi = {"set_value": int(m.group(1) or m.group(2))}
    if moi is None or moi == args:
        return GIU, ten, args, ""
    if "set_value" in moi and not 0 <= moi["set_value"] <= 40:
        return BO, ten, args, "am_luong_ngoai_mien"
    return SUA, ten, moi, "am_luong_quy_uoc"


# ── 3. kính cửa sổ ─────────────────────────────────────────────────────────
_DONG = re.compile(r"\bđóng\b")
_MO = re.compile(r"\b(?:mở|hạ|xuống)\b")
_NANG = re.compile(r"\b(?:nâng|lên)\b")


def _sua_kinh(cau: str, ten: str, args: dict[str, Any]) -> tuple[str, str, dict[str, Any], str]:
    c = cau.lower()
    pos = args.get("position")
    dong, mo, nang = bool(_DONG.search(c)), bool(_MO.search(c)), bool(_NANG.search(c))
    if pos == "down" and dong:
        if mo:
            return BO, ten, args, "kinh_mau_thuan"
        return SUA, ten, {**args, "position": "up"}, "kinh_nguoc_chieu"
    if pos == "up" and mo and not (dong or nang):
        return SUA, ten, {**args, "position": "down"}, "kinh_nguoc_chieu"
    return GIU, ten, args, ""


# ── 4. từ chối nhầm câu find_nearby làm được ────────────────────────────────
_NHOM_DIA_DIEM = [
    ("atm", re.compile(r"\bATM\b|cây rút tiền", re.IGNORECASE)),
    ("gas_station", re.compile(r"cây xăng|trạm xăng|cây xăng", re.IGNORECASE)),
    ("charging_station", re.compile(r"trạm sạc", re.IGNORECASE)),
    ("parking", re.compile(r"bãi đỗ|chỗ đỗ|bãi giữ xe", re.IGNORECASE)),
    ("coffee", re.compile(r"cà phê|cafe|coffee", re.IGNORECASE)),
    ("hospital", re.compile(r"bệnh viện", re.IGNORECASE)),
    ("restaurant", re.compile(r"nhà hàng|quán ăn|quán cơm", re.IGNORECASE)),
]
_GAN = re.compile(r"gần|quanh đây|lân cận", re.IGNORECASE)
_VUOT_TIM_DIA_DIEM = re.compile(
    r"lái|đặt (?:một )?bàn|đặt chỗ|lịch|giá vé|số điện thoại|liên hệ|thời gian hoạt động|giờ mở cửa|"
    r"sửa|xem phim|giao|gọi món|đánh giá",
    re.IGNORECASE,
)


def _sua_tu_choi(cau: str, ten: str, args: dict[str, Any]) -> tuple[str, str, dict[str, Any], str]:
    if not _GAN.search(cau) or _VUOT_TIM_DIA_DIEM.search(cau):
        return GIU, ten, args, ""
    for nhom, pat in _NHOM_DIA_DIEM:
        if pat.search(cau):
            return SUA, "find_nearby", {"category": nhom}, "tu_choi_nham_find_nearby"
    return GIU, ten, args, ""


_LUAT = {
    "get_fuel_level": _sua_xang,
    "get_range_estimate": _sua_xang,
    "set_volume": _sua_am_luong,
    "control_window": _sua_kinh,
    "irrelevant_function": _sua_tu_choi,
}


def sua_loi_goi(cau: str, ten: str, args: dict[str, Any]) -> tuple[str, str, dict[str, Any], str]:
    """Một lời gọi -> (GIU|SUA|BO, tên mới, tham số mới, lý do)."""
    luat = _LUAT.get(ten)
    return luat(cau, ten, args) if luat else (GIU, ten, args, "")


_CALL_TRONG_PROMPT = re.compile(r"(?:<nexa_ask>)?<nexa_\d+>\(.*?\)<nexa_end>", re.DOTALL)


def _loi_goi_cua_reply(prompt: str) -> tuple[str, dict[str, Any]] | None:
    m = _CALL_TRONG_PROMPT.search(prompt)
    if not m:
        return None
    try:
        calls = parse_calls(m.group(0))
    except CallParseError:
        return None
    return (calls[0].name, dict(calls[0].args)) if len(calls) == 1 else None


def _cau_cua_reply(prompt: str) -> str:
    return query_from_call_prompt(prompt)


def sua_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Sửa/bỏ cả kho. Trả (kho mới, báo cáo đếm theo lý do + vài ví dụ mỗi lý do)."""
    ra: list[dict[str, Any]] = []
    dem: Counter[str] = Counter()
    vi_du: dict[str, list[str]] = {}
    for r in rows:
        loai = r["meta"]["type"]
        if loai not in ("call", "reply"):
            ra.append(r)
            continue
        if loai == "call":
            try:
                calls = parse_calls(r["completion"])
            except CallParseError:
                ra.append(r)
                continue
            if len(calls) != 1 or calls[0].suggested:
                ra.append(r)
                continue
            ten, args = calls[0].name, dict(calls[0].args)
        else:
            goc = _loi_goi_cua_reply(r["prompt"])
            if goc is None:
                ra.append(r)
                continue
            ten, args = goc
        cau = _cau_cua_reply(r["prompt"])
        kq, ten_moi, args_moi, ly_do = sua_loi_goi(cau, ten, args)
        if kq == GIU:
            ra.append(r)
            continue
        khoa = f"call:{kq}:{ly_do}" if loai == "call" else f"reply:bo:{ly_do}"
        dem[khoa] += 1
        mau = vi_du.setdefault(khoa, [])
        if len(mau) < 5:
            mau.append(f"{cau} -> {ten_moi}({args_moi})" if kq == SUA else cau)
        if loai == "reply" or kq == BO:
            continue  # reply theo lời gọi sai: bỏ, không sửa được bằng code
        moi = {**r, "meta": {**r["meta"], "fn": ten_moi, "sua_nhan": ly_do}}
        moi["completion"] = render_completion([(ten_moi, args_moi)]) + IM_END
        ra.append(moi)
    ra, xung_dot = bo_cau_trung_khac_nhan(ra)
    if xung_dot:
        dem["call:bo:trung_cau_khac_nhan"] = len(xung_dot)
        cau_xau = {query_from_call_prompt(r["prompt"]) for r in xung_dot}
        vi_du["call:bo:trung_cau_khac_nhan"] = sorted(cau_xau)[:5]
    return ra, {"dem": dict(sorted(dem.items())), "vi_du": vi_du}


def bo_cau_trung_khac_nhan(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Cùng một câu mà có hai nhãn khác nhau ("Pin còn bao nhiêu phần trăm?" -> vừa
    `get_battery_range` vừa `get_vehicle_status`) -> bỏ MỌI bản: nhãn tự mâu thuẫn là nhãn
    không đáng tin, và bỏ hết thì không phải tự tay chọn đáp án. Chỉ xét mẫu call/ask (cùng Prompt A)."""
    nhan: dict[str, set[str]] = {}
    for r in rows:
        if r["meta"]["type"] in ("call", "ask"):
            nhan.setdefault(r["prompt"], set()).add(r["completion"])
    xau = {p for p, c in nhan.items() if len(c) > 1}
    giu = [r for r in rows if r["meta"]["type"] not in ("call", "ask") or r["prompt"] not in xau]
    bo = [r for r in rows if r["meta"]["type"] in ("call", "ask") and r["prompt"] in xau]
    return giu, bo
