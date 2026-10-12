"""BỘ NHỚ LƯỢT TRƯỚC bằng LUẬT (người dùng chọn 2026-10-06, PLAN.md Phase 7 — cách 1 của "cả hai").

Model v5 chỉ thấy câu hiện tại (đúng thiết kế Octopus v2) -> "mở cốp" rồi "đóng đi" thì "đóng đi" bị từ
chối. Luật này nhớ HÀNH ĐỘNG vừa thực hiện và hiểu câu nối tiếp CHỈ GỒM một động từ ("đóng đi", "tắt",
"giảm nữa", "thêm chút nữa"). Câu có nêu bộ phận ("đóng cốp") vẫn để model. Lời gọi suy ra vẫn qua
validator + luật an toàn như mọi lời gọi khác.

Cách 2 (vòng v6): đưa lời gọi lượt trước vào prompt để MODEL học — model đạt thì bỏ luật này; bài so sánh
hai cách.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from serve.tra_loi import BUOC_AM_LUONG, BUOC_NHIET

LoiGoi = tuple[str, dict[str, Any]]


@dataclass(frozen=True)
class LuotTruoc:
    ten: str
    args: dict[str, Any]
    huong: int = 0  # +1 / −1 khi lượt trước là tăng / giảm (nhiệt, âm lượng) — để hiểu "nữa", "thêm chút"


# từ đệm bỏ đi trước khi so ("đóng lại đi nhé" -> "đóng")
_DEM = {
    "đi", "nhé", "nha", "luôn", "giúp", "giùm", "hộ", "với", "ạ", "nào", "cái", "bạn", "em", "anh", "chị",
    "tôi", "mình", "cho", "lại", "đã", "vào", "hết",
}
# từ "lặp lại / thêm nữa" — câu chỉ còn những từ này = làm tiếp theo hướng lượt trước
_NUA = {"nữa", "thêm", "tiếp", "chút", "xíu", "tí", "một"}

_CUM = {
    "mở": "mo", "mở ra": "mo",
    "đóng": "dong", "khép": "dong", "sập": "dong",
    "bật": "bat", "bật lên": "bat",
    "tắt": "tat",
    "hạ": "ha", "hạ xuống": "ha", "kéo xuống": "ha", "xuống": "ha",
    "nâng": "nang", "nâng lên": "nang", "kéo lên": "nang", "lên": "nang",
    "khoá": "khoa",
    "giảm": "giam", "giảm xuống": "giam", "nhỏ": "giam", "nhỏ xuống": "giam", "bé": "giam", "mát": "giam",
    "tăng": "tang", "tăng lên": "tang", "to": "tang", "to lên": "tang", "ấm": "tang",
}


def _hanh_dong(cau: str) -> str | None:
    """Câu CHỈ gồm một động từ (+ từ đệm) -> mã hành động; câu có nội dung khác -> None (để model)."""
    c = re.sub(r"[.,!?…]", " ", cau.lower()).replace("khóa", "khoá")
    tu = c.split()
    if not tu or len(tu) > 6:
        return None
    co_nua = any(t in _NUA for t in tu)
    loi = " ".join(t for t in tu if t not in _DEM and t not in _NUA)
    if not loi:
        return "lap" if co_nua else None
    return _CUM.get(loi)


def noi_tiep(cau: str, truoc: LuotTruoc | None, s: Any) -> LoiGoi | None:
    """Lời gọi suy ra từ câu nối tiếp + hành động lượt trước; None = không áp (để model xử lý)."""
    if truoc is None or (hd := _hanh_dong(cau)) is None:
        return None
    ten, a = truoc.ten, truoc.args
    if ten in ("open_trunk", "close_trunk"):
        return {"mo": ("open_trunk", {}), "dong": ("close_trunk", {})}.get(hd)
    if ten == "control_window":
        vi_tri = {"mo": "down", "ha": "down", "dong": "up", "nang": "up"}.get(hd)
        return ("control_window", {"window": a.get("window", "all"), "position": vi_tri}) if vi_tri else None
    if ten == "operate_sunroof":
        hanh = {"mo": "open", "dong": "close"}.get(hd)
        return ("operate_sunroof", {"action": hanh}) if hanh else None
    if ten in ("lock_doors", "unlock_doors"):  # "đóng" cửa = đóng cánh cửa, KHÔNG suy ra khoá
        return {"mo": ("unlock_doors", {}), "khoa": ("lock_doors", {})}.get(hd)
    if ten in ("set_defrost", "set_hazard_lights"):
        trang_thai = {"bat": "on", "tat": "off"}.get(hd)
        if trang_thai is None:
            return None
        giu = {"zone": a["zone"]} if "zone" in a else {}
        return (ten, {"state": trang_thai, **giu})
    if ten == "set_headlights":
        return ("set_headlights", {"mode": "off"}) if hd == "tat" else None
    huong = {"giam": -1, "ha": -1, "tang": 1, "nang": 1, "lap": truoc.huong}.get(hd, 0)
    if ten == "set_ac":
        if hd in ("bat", "tat"):
            return ("set_ac", {"state": "on" if hd == "bat" else "off"})
        return ("set_ac", {"state": "on", "delta": huong * BUOC_NHIET}) if huong else None
    if ten == "set_volume":
        return ("set_volume", {"delta": huong * BUOC_AM_LUONG}) if huong else None
    if ten == "set_fan_speed":
        if hd == "tat":
            return ("set_fan_speed", {"level": 0})
        return ("set_fan_speed", {"delta": huong}) if huong else None  # v6: quạt có `delta`, bước ±1
    return None  # nắp sạc / nắp xăng (không có hàm đóng), dẫn đường, nhạc... -> để model


def huong_cau(cau: str, truoc: LuotTruoc | None) -> int:
    """Hướng mà câu nối tiếp yêu cầu ("giảm nữa" -> −1, "nữa" -> theo lượt trước)."""
    hd = _hanh_dong(cau)
    if hd == "lap":
        return truoc.huong if truoc else 0
    return {"giam": -1, "ha": -1, "tang": 1, "nang": 1}.get(hd or "", 0)


def huong_cua(ten: str, args: dict[str, Any], huong_cau: int = 0) -> int:
    """Hướng tăng / giảm của một lời gọi vừa chạy (để "nữa" lần sau đi tiếp đúng chiều)."""
    d = args.get("delta", 0)
    if d:
        return (d > 0) - (d < 0)
    return huong_cau if ten in ("set_ac", "set_volume", "set_fan_speed") else 0
