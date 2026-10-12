"""CỔNG MODEL — chỗ duy nhất luồng chạy thật nói chuyện với mô hình (PLAN.md Phase 5).

Luồng (serve/tac_tu.py) chỉ biết hai việc của model:
  - `goi_ham(cau)`  lượt 1: câu người lái -> chuỗi "<nexa_i>(...)<nexa_end>" (đúng định dạng lúc train);
  - `tra_loi(cau, loi_goi, ket_qua)`  lượt 2: viết câu trả lời; trả None -> luồng dùng câu mẫu.

Đằng sau cổng có thể là model giả (file này), student GGUF chạy llama.cpp trên CPU, hay một model khác — đổi
model KHÔNG phải sửa luồng. Đó là lời khuyên "làm flow trước, model tính sau".
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Protocol

from schema.call_format import render_completion
from schema.loader import CATALOG
from serve.chuan_hoa import chuan_hoa_chu

LoiGoi = tuple[str, dict[str, Any]]


class MoHinh(Protocol):
    # True = model đã học ngữ cảnh lượt trước (v6): router đưa lời gọi lượt trước vào prompt và TẮT luật
    # bộ nhớ
    nho_luot_truoc: bool

    def goi_ham(self, cau: str, luot_truoc: str | None = None) -> str: ...

    def tra_loi(self, cau: str, loi_goi: str, ket_qua: dict[str, Any]) -> str | None: ...


# ── model giả: luật từ khoá, CHỈ để chạy thử luồng và viết test ─────────────
_KINH_THEO_VI_TRI = [
    (re.compile(r"sau (bên )?trái"), "rear_left"),
    (re.compile(r"sau (bên )?phải"), "rear_right"),
    (re.compile(r"ghế phụ|bên phụ"), "front_right"),
    (re.compile(r"ghế lái|bên lái|bên tài"), "front_left"),
]
_PHAN_TRAM = re.compile(r"(\d+)\s*(%|phần trăm)")


def _kinh(m: re.Match[str], vi_tri: str) -> list[LoiGoi]:
    c = m.string
    args: dict[str, Any] = {"position": vi_tri}
    if (pt := _PHAN_TRAM.search(c)) is not None:
        args["amount"] = int(pt.group(1))
    elif "một nửa" in c:
        args["amount"] = 50
    for pat, cua in _KINH_THEO_VI_TRI:
        if pat.search(c):
            return [("control_window", {"window": cua, **args})]
    # hai kính sau / trước (schema v6: window="rear" / "front"); cho phép chữ chen giữa ("kính XE sau")
    if re.search(r"(kính|cửa sổ)( \w+)? sau|hàng (ghế )?sau|phía sau", c):
        return [("control_window", {"window": "rear", **args})]
    if re.search(r"(kính|cửa sổ)( \w+)? trước|hai kính trước", c):
        return [("control_window", {"window": "front", **args})]
    if re.search(r"\b(hết|tất cả|cả|toàn bộ|mọi)\b", c):
        return [("control_window", {"window": "all", **args})]
    # không rõ kính nào -> KHÔNG đoán "tất cả": bỏ trống `window` để validator hỏi lại "kính cửa nào?"
    return [("control_window", args)]


def _am_luong(m: re.Match[str]) -> list[LoiGoi]:
    if (so := re.search(r"mức (\d+)", m.string)) is not None:
        return [("set_volume", {"set_value": int(so.group(1))})]
    return [("set_volume", {"delta": -2 if re.search(r"giảm|nhỏ", m.string) else 2})]


# (mẫu, cách dựng lời gọi, là đề xuất?) — xét theo thứ tự, khớp đầu tiên thắng
_LUAT: list[tuple[re.Pattern[str], Callable[[re.Match[str]], list[LoiGoi]], bool]] = [
    (re.compile(r"mở kho[áa]"), lambda m: [("unlock_doors", {})], False),
    (re.compile(r"kho[áa] (hết |tất cả )?cửa"), lambda m: [("lock_doors", {})], False),
    (re.compile(r"đóng cốp"), lambda m: [("close_trunk", {})], False),
    (re.compile(r"mở cốp"), lambda m: [("open_trunk", {})], False),
    (re.compile(r"nắp sạc|cổng sạc"), lambda m: [("open_charge_port", {})], False),
    (re.compile(r"nắp (bình )?xăng"), lambda m: [("open_fuel_flap", {})], False),
    (re.compile(r"tắt (điều h(?:oà|òa|oa)|máy lạnh)"), lambda m: [("set_ac", {"state": "off"})], False),
    (
        re.compile(r"(điều h(?:oà|òa|oa)|máy lạnh)\D*(\d+) ?độ"),
        lambda m: [("set_ac", {"state": "on", "temperature": int(m.group(2))})],
        False,
    ),
    (re.compile(r"(bật|mở) (điều h(?:oà|òa|oa)|máy lạnh)"), lambda m: [("set_ac", {"state": "on"})], False),
    (re.compile(r"(bật|mở) sưởi"), lambda m: [("set_ac", {"state": "on", "mode": "heat"})], False),
    (
        re.compile(r"sấy kính"),
        lambda m: [("set_defrost", {"state": "on", "zone": "rear" if "sau" in m.string else "front"})],
        False,
    ),
    (
        re.compile(r"(mở|đóng|hé) cửa sổ trời"),
        lambda m: [
            ("operate_sunroof", {"action": {"mở": "open", "đóng": "close", "hé": "tilt"}[m.group(1)]})
        ],
        False,
    ),
    (re.compile(r"(hạ|mở) (hết )?(kính|cửa sổ)"), lambda m: _kinh(m, "down"), False),
    (re.compile(r"(nâng|đóng|kéo) (hết )?(kính|cửa sổ)"), lambda m: _kinh(m, "up"), False),
    (
        re.compile(r"đèn khẩn cấp"),
        lambda m: [("set_hazard_lights", {"state": "off" if "tắt" in m.string else "on"})],
        False,
    ),
    (re.compile(r"âm lượng|loa|vặn (to|nhỏ)"), _am_luong, False),
    (
        re.compile(r"(tìm|có) (cây xăng|trạm xăng)"),
        lambda m: [("find_nearby", {"category": "gas_station"})],
        False,
    ),
    (re.compile(r"xăng"), lambda m: [("get_fuel_level", {})], False),
    (re.compile(r"\bpin\b"), lambda m: [("get_battery_range", {})], False),
    (re.compile(r"lốp"), lambda m: [("get_tire_pressure", {})], False),
    (re.compile(r"tình trạng xe|kiểm tra xe"), lambda m: [("get_vehicle_status", {})], False),
    (re.compile(r"thời tiết"), lambda m: [("get_weather", {})], False),
    (
        re.compile(r"dẫn đường (?:tới|đến|về) (.+)"),
        lambda m: [("navigate_to", {"destination": m.group(1).strip(" .!?")})],
        False,
    ),
    (
        re.compile(r"(?:mở|phát|bật) nhạc(?: của)? ?(.*)"),
        lambda m: [("play_media", {"query": m.group(1).strip(" .!?")} if m.group(1).strip(" .!?") else {})],
        False,
    ),
    # câu than thở -> chỉ ĐỀ XUẤT (<nexa_ask>), luồng sẽ hỏi lại
    (re.compile(r"nóng quá|nóng thế|nóng nực"), lambda m: [("set_ac", {"state": "on"})], True),
    (
        re.compile(r"bí quá|ngột ngạt"),
        lambda m: [("set_ac", {"state": "on"}), ("control_window", {"window": "all", "position": "down"})],
        True,
    ),
]


class MoHinhGia:
    """Model giả theo luật từ khoá. `bang` = output thô ép sẵn cho từng câu (để test tình huống model
    sinh sai, vd `operate_sunroof(action="fly")`); `cau_tra_loi` = câu lượt 2 ép sẵn theo tên hàm."""

    nho_luot_truoc = False

    def __init__(self, bang: dict[str, str] | None = None, cau_tra_loi: dict[str, str] | None = None) -> None:
        # khoá theo dạng chuẩn (hoa chữ đầu): router đưa câu đã chuẩn hoá vào model (serve/chuan_hoa.py)
        self.bang = {chuan_hoa_chu(k): v for k, v in (bang or {}).items()}
        self.cau_tra_loi = cau_tra_loi or {}

    def goi_ham(self, cau: str, luot_truoc: str | None = None) -> str:
        if chuan_hoa_chu(cau) in self.bang:
            return self.bang[chuan_hoa_chu(cau)]
        c = cau.lower().strip()
        for pat, dung, de_xuat in _LUAT:
            if (m := pat.search(c)) is not None:
                return render_completion(dung(m), ask=de_xuat)
        return render_completion([("irrelevant_function", {})])

    def tra_loi(self, cau: str, loi_goi: str, ket_qua: dict[str, Any]) -> str | None:
        for ten, cau_tl in self.cau_tra_loi.items():
            if ten == "*" or CATALOG.by_name[ten].nexa_token + "(" in loi_goi:
                return cau_tl
        return None
