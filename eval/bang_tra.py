"""Sinh bảng tra 24 hàm (Markdown) cho người viết tập test tay.

Sinh thẳng từ schema/functions.yaml nên không bao giờ lệch với code. Chạy lại mỗi khi
schema đổi:

    python -m eval.bang_tra --output data/test_tay/HUONG_DAN.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from schema.loader import CATALOG

MO_DAU = """# Bảng tra hàm — dùng khi viết tập test tay

Cách dùng: viết câu vào cột `cau` trong `test_vi.csv`. Cột `nhan` ghi theo mẫu ở đây.

| Kiểu câu | Cách ghi cột `nhan` | Ví dụ |
|---|---|---|
| Ra lệnh rõ ràng | `tên_hàm(tham_số=giá_trị)` | `set_ac(state="on", temperature=26)` |
| Mơ hồ, agent nên hỏi lại | thêm `ask: ` ở đầu | `ask: set_ac(state="on")` |
| Mơ hồ, có 2 cách xử lý | ngăn bằng dấu `;` | `ask: set_ac(state="on"); operate_sunroof(action="open")` |
| Ngoài phạm vi | luôn là | `irrelevant_function()` |

Quy tắc: **chỉ ghi tham số mà câu nói có nhắc tới**. "Bật điều hòa" thì chỉ ghi
`set_ac(state="on")`, đừng tự thêm nhiệt độ.

---

## Cặp hàm dễ nhầm — đọc trước khi kiểm

| Câu nói về | Hàm đúng | Hay bị nhầm thành |
|---|---|---|
| còn bao nhiêu xăng / pin **nói chung** | `get_fuel_level` / `get_battery_range` | — |
| đi **một quãng cụ thể** ("đi thêm 175km") | `get_range_estimate(trip_km=175)` | `get_fuel_level()` |
| **hạ / nâng kính** (kính hai bên) | `control_window` | `set_defrost` (sấy kính) |
| **sấy / sưởi / khử sương** kính | `set_defrost` | `control_window` |
| đèn **pha / cốt** (chiếu sáng) | `set_headlights` | `set_hazard_lights` |
| đèn **khẩn cấp / xi nhan đôi** — xe MÌNH gặp sự cố | `set_hazard_lights` | `set_headlights` |
| đèn / sự cố của **xe khác** | không làm gì (`irrelevant_function`) | `set_hazard_lights` |
| **gió** mạnh / yếu / ồn (điều hoà đang chạy) | `set_fan_speed` | `set_ac` |
| **nóng / lạnh** | `set_ac` | `set_fan_speed` |
| đến **một nơi có tên** | `navigate_to(destination=...)` | `find_nearby` |
| tìm **một chỗ bất kỳ gần đây** (cây xăng, quán ăn...) | `find_nearby(category=...)` | `navigate_to` |
| âm lượng **về một mức** ("mức 15") | `set_volume(set_value=15)` | `set_volume(delta=...)` |
| âm lượng **thêm / bớt** ("to thêm 3") | `set_volume(delta=3)` | `set_volume(set_value=...)` |
| nhiệt độ **về / xuống / lên một số** ("xuống 20 độ") | `set_ac(state="on", temperature=20)` | `set_ac(delta=...)` |
| nhiệt độ **tăng / giảm một khoảng** ("giảm 3 độ") | `set_ac(state="on", delta=-3)` | `set_ac(temperature=...)` |
| nhiệt độ tăng / giảm **không nói số** ("mát hơn chút") | `set_ac(state="on", delta=-2)` (quy ước ±2) | `set_ac(state="off")` |
| **sưởi / làm lạnh / tự động** nói rõ | `set_ac(state="on", mode="heat")` / `"cool"` / `"auto"` | bỏ `mode` |
| **cả hai** kính sau / trước ("hạ hai kính sau") | `control_window(window="rear", ...)` / `"front"` | chỉ một bên |

**Tham số tuỳ chọn bỏ trống là bình thường** — xe dùng giá trị mặc định (cột "Mặc định" bên dưới). Chỉ tính
là sai khi thiếu thứ người dùng **đã nói**, hoặc bịa thêm thứ người dùng **không nói**.

---

"""

# Nghĩa tiếng Việt của các giá trị enum — để người kiểm biết `front_left` là cửa nào.
NGHIA = {
    "front_left": "trước trái (ghế lái)", "front_right": "trước phải (ghế phụ)",
    "rear_left": "sau trái", "rear_right": "sau phải", "all": "tất cả",
    "driver": "ghế lái", "passenger": "ghế phụ", "rear": "hàng sau",
    "front": "trước", "both": "cả trước và sau",
    "on": "bật", "off": "tắt", "up": "nâng lên (đóng)", "down": "hạ xuống (mở)",
    "open": "mở", "close": "đóng", "tilt": "hé",
    "auto": "tự động", "low_beam": "đèn cốt (chiếu gần)", "high_beam": "đèn pha (chiếu xa)",
    "parking": "đèn đỗ xe",
    "km": "ki-lô-mét", "mi": "dặm",
    "radio": "đài FM/AM", "bluetooth": "điện thoại qua Bluetooth", "usb": "USB",
    "streaming": "nhạc trực tuyến",
    "none": "không tránh", "tolls": "tránh trạm thu phí", "highways": "tránh cao tốc",
    "gas_station": "cây xăng", "charging_station": "trạm sạc",
    "restaurant": "quán ăn", "hospital": "bệnh viện", "atm": "ATM", "coffee": "quán cà phê",
}


# Cùng một giá trị, nghĩa khác nhau tuỳ tham số: `parking` của set_headlights là đèn đỗ xe,
# của find_nearby là bãi đỗ.
NGHIA_THEO_THAM_SO = {
    ("category", "parking"): "bãi đỗ xe",
    ("window", "front"): "cả hai kính trước",
    ("window", "rear"): "cả hai kính sau",
    ("mode", "cool"): "làm lạnh",
    ("mode", "heat"): "sưởi ấm",
}


def _nghia(ten_tham_so: str, e: str) -> str | None:
    return NGHIA_THEO_THAM_SO.get((ten_tham_so, e), NGHIA.get(e))


def _gia_tri(p) -> str:
    if p.enum:
        return ", ".join(f"`{e}` {_nghia(p.name, e)}" if _nghia(p.name, e) else f"`{e}`" for e in p.enum)
    if p.minimum is not None and p.maximum is not None:
        return f"số từ {p.minimum} đến {p.maximum}"
    if p.minimum is not None:
        return f"số ≥ {p.minimum}"
    if p.maximum is not None:
        return f"số ≤ {p.maximum}"
    return "chữ tự do" if p.type == "string" else p.type


def sinh_bang() -> str:
    dong = [MO_DAU]
    theo_nhom: dict[str, list] = {}
    for f in CATALOG.functions:
        theo_nhom.setdefault(f.category, []).append(f)

    ten_nhom = {
        "status": "Tra cứu trạng thái xe",
        "climate": "Điều hoà",
        "access": "Cửa, cốp, nắp",
        "opening": "Cửa sổ, cửa sổ trời",
        "lights": "Đèn",
        "media": "Nhạc, âm lượng",
        "navigation": "Dẫn đường",
        "info": "Thông tin ngoài",
        "fallback": "Ngoài phạm vi",
    }

    for nhom, ds in theo_nhom.items():
        dong.append(f"## {ten_nhom.get(nhom, nhom)}\n")
        for f in ds:
            dong.append(f"### `{f.name}`")
            dong.append(f"{' '.join(f.description.split())}\n")
            if f.params:
                dong.append("| Tham số | Bắt buộc | Giá trị hợp lệ | Mặc định (khi bỏ trống) |")
                dong.append("|---|---|---|---|")
                for p in f.params:
                    co_md = p.default is not None
                    mac_dinh = "—" if p.required else (f"`{p.default}`" if co_md else "tuỳ xe")
                    bat_buoc = "**có**" if p.required else "không"
                    dong.append(f"| `{p.name}` | {bat_buoc} | {_gia_tri(p)} | {mac_dinh} |")
                dong.append("")
            else:
                dong.append("*Không có tham số.*\n")
            dong.append("Câu ví dụ:")
            for e in f.examples:
                dong.append(f'- "{e}"')
            if f.implicit_examples:
                dong.append("\nCâu mơ hồ (ghi kèm `ask:`):")
                for e in f.implicit_examples:
                    dong.append(f'- "{e}"')
            dong.append("")
    return "\n".join(dong)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sinh bảng tra hàm cho người viết test")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(sinh_bang(), encoding="utf-8")
    print(f"đã ghi {args.output} ({len(CATALOG.functions)} hàm)")


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
