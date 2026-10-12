"""Đóng nắp bình xăng / nắp cổng sạc bằng LUẬT RUNTIME (người dùng chọn "cách A", 2026-10-12).

Schema chỉ có `open_fuel_flap` / `open_charge_port` (24 hàm, không có hàm đóng) -> model không có lựa chọn
đúng cho "đóng nắp xăng" và dễ gọi nhầm hàm MỞ gần nghĩa nhất. Trong lúc chưa thêm hàm đóng vào model (cách
B: token mới + train lại), code bắt câu đóng nắp TRƯỚC khi hỏi model:
  • câu có động từ đóng (đóng / khép / đậy / gập) đi trước "nắp xăng", "nắp sạc", "cổng sạc"…;
  • câu nối tiếp chỉ gồm động từ đóng ("đóng đi", "đóng lại giúp mình") ngay sau khi vừa MỞ một trong hai nắp.
Đóng nắp không có rủi ro an toàn nên không cần xác nhận. Đây là luật viết tay — ghi là hạn chế trong báo cáo.
"""

from __future__ import annotations

import re

XANG, SAC = "xang", "sac"
TEN_HAM = {XANG: "close_fuel_flap", SAC: "close_charge_port"}  # frontend có sẵn hoạt cảnh cho hai tên này
HAM_MO = {"open_fuel_flap": XANG, "open_charge_port": SAC}

_DONG = r"(?:đóng|khép|đậy|gập)"
_NAP_XANG = r"(?:nắp|cửa)\s+(?:bình\s+|thùng\s+)?(?:xăng|nhiên\s+liệu)"
_NAP_SAC = r"(?:(?:nắp|cửa)\s+(?:cổng\s+)?sạc|(?:cổng|ổ)\s+sạc)"
# động từ đóng đứng TRƯỚC tên nắp, cách nhau tối đa vài chữ ("đóng giúp mình cái nắp xăng")
_CAU_DONG = re.compile(
    rf"\b{_DONG}\b(?:\s+\S+){{0,4}}?\s+(?:(?P<xang>{_NAP_XANG})|(?P<sac>{_NAP_SAC}))", re.IGNORECASE
)
_TU_DEM = r"(?:\s+(?:lại|vào|đi|nó|giúp|giùm|hộ|mình|tôi|nhé|nha|luôn|với|ạ|cái))*"
_CHI_DONG = re.compile(rf"^\s*{_DONG}{_TU_DEM}\s*[.!?]*\s*$", re.IGNORECASE)


def nap_can_dong(cau: str, ham_vua_lam: str | None = None) -> str | None:
    """"xang" / "sac" nếu câu là lệnh đóng nắp, không thì None.
    `ham_vua_lam`: hành động lượt trước (cho "đóng đi")."""
    if m := _CAU_DONG.search(cau):
        return XANG if m.group("xang") else SAC
    if ham_vua_lam in HAM_MO and _CHI_DONG.match(cau):
        return HAM_MO[ham_vua_lam]
    return None
