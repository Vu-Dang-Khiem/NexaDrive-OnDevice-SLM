"""Câu MẪU tiếng Việt — lưới an toàn cho lượt 2 và các câu hỏi lại (PLAN.md Phase 5).

Lượt 2 ưu tiên câu do model viết; câu mẫu dùng khi model không viết được, viết rỗng, sinh lại lời gọi hàm,
hoặc BỊA SỐ (kiểm bằng `khong_bia_so` — cùng hàm với bộ chấm). Câu mẫu chỉ dùng số lấy từ kết quả thật nên
không bao giờ bịa.
"""

from __future__ import annotations

import re
from typing import Any

# Quy ước người dùng chốt (PLAN.md mục U): than lạnh mà KHÔNG nhắc điều hoà -> lạnh do trời ->
# set_ac(state="on") nhưng phải NÓI là "bật sưởi"; câu có nhắc điều hoà / máy lạnh thì model đề xuất tắt.
_LANH = re.compile(r"\b(lạnh|rét|cóng|buốt|giá rét)\b")
# "hoà" / "hòa" (hai kiểu đặt dấu) / "hoa" — thiếu "hòa" thì "bật điều hòa lạnh" bị coi là than lạnh
# -> báo "bật sưởi" (người dùng gặp 2026-10-06)
_NHAC_DIEU_HOA = re.compile(r"điều h(?:oà|òa|oa)|máy lạnh|làm lạnh|gió lạnh từ|cửa gió|\bac\b")


def la_than_lanh(cau: str) -> bool:
    """Câu than lạnh KHÔNG nhắc điều hoà ("trong xe lạnh quá", "trời rét tay cóng")."""
    c = cau.lower()
    return bool(_LANH.search(c)) and not _NHAC_DIEU_HOA.search(c)


# Tăng / giảm nhiệt độ TƯƠNG ĐỐI không kèm số ("giảm nhiệt độ điều hoà", "cho mát hơn chút"): schema `set_ac`
# chưa có tham số tăng / giảm (chỉ nhiệt độ cụ thể) -> model đoán (đo 2026-10-06: "giảm nhiệt độ điều hòa" ra
# TẮT điều hoà). Luật runtime (cách A, PLAN.md Phase 7): nhiệt độ hiện tại ± 2 độ. Cách B (thêm `delta`
# + train lại) ở vòng dữ liệu sau.
BUOC_NHIET = 2
_GIAM_NHIET = re.compile(
    r"\b(giảm|hạ|bớt)\b[^\d]*\bnhiệt\b|\bnhiệt( độ)?\b[^\d]*\b(xuống|thấp hơn)\b|\bmát hơn\b|\bcho mát\b"
)
_TANG_NHIET = re.compile(
    r"\b(tăng|nâng)\b[^\d]*\bnhiệt\b|\bnhiệt( độ)?\b[^\d]*\b(lên|cao hơn)\b|\bấm hơn\b|\bcho ấm\b"
)


def huong_nhiet(cau: str) -> int:
    """−1 = muốn mát hơn, +1 = muốn ấm hơn, 0 = không phải câu chỉnh nhiệt tương đối (hoặc đã nói con số)."""
    c = cau.lower()
    if re.search(r"\d", c):
        return 0
    if _GIAM_NHIET.search(c):
        return -1
    if _TANG_NHIET.search(c):
        return 1
    return 0


# Than âm lượng ("nhạc to vãi", "chẳng nghe rõ gì"): model hiểu là `set_volume` và hỏi lại nhưng KHÔNG đưa mức
# (đo 2026-10-06: `<nexa_ask><nexa_19>()`) -> validator coi là thiếu -> trước đây báo "Bạn muốn tôi làm gì cụ
# thể ạ?". Đoán hướng từ câu để đề xuất ±3 mức (đúng quy ước dữ liệu: "ồn ào bên ngoài" -> delta=3). "ồn"
# không dùng vì hai nghĩa (nhạc ồn / đường ồn).
BUOC_AM_LUONG = 3
_AM_TO = re.compile(r"\b(to|nhức|ầm|inh|chói|điếc|đinh tai)\b")
_AM_NHO = re.compile(r"\b(nhỏ|bé)\b|\b(không|chẳng|chả|khó) nghe\b|\bnghe (không|chẳng|chả) (rõ|thấy)\b")


def canh_bao_khi_chay(s: Any) -> str:
    """Chuyển sang "Đang chạy" mà cốp / cửa / nắp còn mở -> cảnh báo như xe thật (người dùng gặp 2026-10-06:
    đang mở cốp vẫn chạy bình thường). Chỉ CẢNH BÁO, không tự đóng — đóng gì là việc người lái quyết."""
    mo = (["cốp sau"] if s.trunk_open else []) + [TEN_CUA.get(c, c) for c in s.doors_open] + (
        ["nắp cổng sạc"] if s.charge_port_open else []) + (["nắp bình xăng"] if s.fuel_flap_open else [])
    return f"Cảnh báo: {', '.join(mo)} đang mở. Bạn nên đóng lại trước khi chạy." if mo else ""


def huong_am_luong(cau: str) -> int:
    """−1 = than to (giảm), +1 = than nhỏ / khó nghe (tăng), 0 = không rõ."""
    c = cau.lower()
    if _AM_NHO.search(c):
        return 1
    return -1 if _AM_TO.search(c) else 0


def _dau(x: int) -> int:
    return (x > 0) - (x < 0)


def la_suoi(ten: str, a: dict[str, Any], cau: str) -> bool:
    """Lời gọi điều hoà nên NÓI là "bật sưởi": model v6 ghi `mode="heat"`; model v5 chưa có `mode` -> đoán
    từ câu than lạnh không nhắc điều hoà."""
    if ten != "set_ac" or a.get("state") == "off":
        return False
    return a.get("mode") == "heat" or ("mode" not in a and la_than_lanh(cau))


# Trợ lý xưng "mình" ở MỌI câu (người dùng chọn câu chào "Chào bạn, mình là NexaAI…" 2026-10-07; trước đó câu
# mẫu xưng "tôi" — đổi cho đồng bộ, kể cả lý do luật an toàn trong schema/functions.yaml)
TU_CHOI = (
    "Xin lỗi, việc này mình chưa làm được. Mình có thể giúp điều hoà, cửa, kính, đèn, âm lượng, "
    "dẫn đường hoặc kiểm tra tình trạng xe."
)
CHUA_HIEU = "Xin lỗi, mình chưa hiểu ý bạn, bạn nói lại giúp mình nhé."
CHUA_RO = "Bạn muốn mình làm gì cụ thể ạ?"
HUY = "Dạ, vậy mình không làm nữa."


def bao_dong_nap(nap: str, da_dong_san: bool) -> str:
    """Câu trả lời cho luật đóng nắp (serve/dong_nap.py): "xang" / "sac"."""
    ten = "nắp bình xăng" if nap == "xang" else "nắp cổng sạc"
    return f"{ten[0].upper()}{ten[1:]} đang đóng rồi." if da_dong_san else f"Đã đóng {ten}."


def con_lai(doan: list[str]) -> str:
    """Câu nhiều lệnh dừng giữa chừng để hỏi lại: nói rõ việc nào phía sau CHƯA làm (serve/tach_lenh.py)."""
    return "Mình chưa làm tiếp: " + ", ".join(f"«{d}»" for d in doan) + " — bạn nói lại sau nhé."

# ── Câu NGOÀI PHẠM VI: đáp theo loại thay vì một câu từ chối cứng (người dùng 2026-10-07: "xin chào thì
# cũng phải biết trả lời 'Chào bạn, mình là NexaAI…'"). Model vẫn chỉ ra nhãn TỪ CHỐI; câu chữ chọn theo
# luật — xét theo thứ tự, khớp đầu tiên thắng. Câu NHỜ việc xe không làm được vẫn xin lỗi như cũ (TU_CHOI).
TEN_TRO_LY = "NexaAI"
KHA_NANG = ("điều hoà, quạt gió, sấy kính, khoá cửa, cốp, kính, cửa sổ trời, đèn, âm lượng, nhạc, dẫn đường, "
            "tìm chỗ gần đây, xem thời tiết và kiểm tra tình trạng xe")
_NGOAI_PHAM_VI: list[tuple[str, re.Pattern[str], tuple[str, ...]]] = [
    ("tam_biet", re.compile(r"tạm biệt|\b(bye|bai bai|bái bai)\b|hẹn gặp lại|gặp lại sau"),
     ("Tạm biệt bạn, chúc bạn đi đường bình an!",)),
    ("cam_on", re.compile(r"cảm ơn|cám ơn|\b(thanks|thank you|tks|thank)\b"),
     ("Không có gì đâu, cần gì bạn cứ gọi NexaAI nhé!", "Dạ không có gì, mình luôn sẵn sàng giúp bạn!")),
    ("chao", re.compile(r"\b(xin chào|chào|hello|hi|alo|a lô|hế lô|hey)\b"),
     (f"Chào bạn, mình là {TEN_TRO_LY}, mình có thể giúp gì cho bạn nhỉ?",)),
    ("danh_tinh", re.compile(r"\b(bạn|em|cậu|mày|nexa\w*) (là ai|là gì|tên (là )?gì)|\btên (của )?bạn\b|"
                             r"^(ai đấy|ai vậy|ai thế)"),
     (f"Mình là {TEN_TRO_LY}, trợ lý giọng nói trên xe, chạy ngay trên máy, không cần mạng. Mình giúp được "
      f"{KHA_NANG}.",)),
    ("kha_nang", re.compile(r"làm được (gì|những gì|những việc gì)|giúp (được )?(gì|những gì)|"
                            r"(có thể|biết) làm (gì|những gì)|chức năng"),
     (f"Mình giúp được {KHA_NANG}. Bạn cứ nói tự nhiên, ví dụ \"bật điều hoà 22 độ\" hay \"mở cốp\" nhé.",)),
    ("khen", re.compile(r"\b(giỏi|thông minh|tuyệt|hay) (quá|thế|ghê|vậy|đấy)\b|\bngoan\b|\bxịn\b"),
     ("Cảm ơn bạn đã khen, mình sẽ cố gắng hơn nữa!",)),
]
# ĐÃ BỎ (2026-10-07, người dùng gặp): câu "tán gẫu" (không có lời nhờ) từng được đáp vui vẻ xoay
# vòng ("Đúng vậy nhỉ!…") -> "hỏng xe giữa đường rồi" (model từ chối nhầm) nhận "Đúng vậy nhỉ!". Câu
# than phiền / kể sự cố cũng không có lời nhờ nên luật đó không phân biệt được -> mọi câu model từ
# chối ngoài 6 loại xã giao rõ ràng: xin lỗi trung tính. Lối tắt trước model chỉ cho câu NGẮN:
# "Chào buổi sáng, trời lạnh quá" có "chào" nhưng là than lạnh -> để model.
_TOI_DA_TU_LOI_TAT = 5


# động từ ĐIỀU KHIỂN xe — câu xã giao có các động từ này ("chào, mở cốp giúp") là LỆNH, phải để model
_DONG_TU_XE = re.compile(r"\b(bật|tắt|mở|đóng|hạ|nâng|kéo|khoá|khóa|tăng|giảm|vặn|chỉnh|tìm|dẫn|phát|sấy|xem|"
                         r"kiểm tra|đi tới|đi đến|về nhà)\b")


def xa_giao(cau: str) -> str | None:
    """Câu XÃ GIAO THUẦN (chào / cảm ơn / tạm biệt / hỏi danh tính / hỏi làm được gì / khen) không kèm
    lệnh, câu NGẮN (≤ 5 từ) -> câu đáp ngay, KHÔNG qua model (model có lúc hỏi lại thừa với câu tán gẫu).
    None = để model."""
    c = cau.lower()
    if len(re.findall(r"\w+", c)) > _TOI_DA_TU_LOI_TAT:
        return None
    if _DONG_TU_XE.search(c) and not re.search(r"làm được|giúp (được )?(gì|những gì)|làm (gì|những gì)", c):
        return None
    for _, mau, cau_dap in _NGOAI_PHAM_VI:
        if mau.search(c):
            return cau_dap[0]
    return None


def tra_loi_ngoai_pham_vi(cau: str, lan: int = 0) -> tuple[str, str]:
    """Câu model đã TỪ CHỐI -> (câu đáp, loại). `lan` = số lần đã đáp trong phiên (xoay vòng câu cảm ơn).
    Chỉ 6 loại xã giao rõ ràng có câu riêng; còn lại xin lỗi trung tính (không đoán câu là tán gẫu)."""
    c = cau.lower()
    for loai, mau, cau_dap in _NGOAI_PHAM_VI:
        if mau.search(c):
            return cau_dap[lan % len(cau_dap)], loai
    return TU_CHOI, "khong_lam_duoc"

TEN_KINH = {
    "front_left": "kính ghế lái",
    "front_right": "kính ghế phụ",
    "rear_left": "kính sau bên trái",
    "rear_right": "kính sau bên phải",
    "front": "hai kính trước",
    "rear": "hai kính sau",
    "all": "tất cả kính",
}
TEN_CHE_DO = {"cool": "làm lạnh", "heat": "sưởi", "auto": "tự động"}
TEN_CUA = {"driver": "cửa ghế lái", "passenger": "cửa ghế phụ", "rear_left": "cửa sau bên trái",
           "rear_right": "cửa sau bên phải"}
TEN_VUNG = {"driver": " cho ghế lái", "passenger": " cho ghế phụ", "rear": " cho hàng ghế sau", "all": ""}
TEN_SAY = {"front": "kính trước", "rear": "kính sau", "both": "cả kính trước lẫn kính sau"}
TEN_DEN = {
    "off": "tắt đèn pha",
    "auto": "chuyển đèn pha sang tự động",
    "low_beam": "bật đèn chiếu gần",
    "high_beam": "bật đèn chiếu xa",
    "parking": "bật đèn đỗ xe",
}
TEN_DIA_DIEM = {
    "gas_station": "cây xăng",
    "charging_station": "trạm sạc",
    "parking": "bãi đỗ xe",
    "restaurant": "nhà hàng",
    "hospital": "bệnh viện",
    "atm": "cây ATM",
    "coffee": "quán cà phê",
}
TEN_CUA_SO_TROI = {"open": "mở cửa sổ trời", "close": "đóng cửa sổ trời", "tilt": "hé cửa sổ trời"}


def _so(x: float) -> str:
    """Số kiểu Việt: 23450 -> '23.450', 264.6 -> '264,6'."""
    if float(x).is_integer():
        return f"{int(x):,}".replace(",", ".")
    return f"{x:.1f}".replace(".", ",")


def hanh_dong(ten: str, a: dict[str, Any], cau: str = "") -> str:
    """Cụm động từ mô tả một lời gọi — dùng trong câu hỏi lại ("Bạn có muốn tôi … không?").
    `cau` = câu gốc của người lái (để nói "bật sưởi" với câu than lạnh)."""
    if ten == "set_ac":
        if a.get("state") == "off":
            return "tắt điều hoà"
        if a.get("delta"):
            d = a["delta"]
            return f"{'giảm' if d < 0 else 'tăng'} nhiệt độ điều hoà {abs(d)} độ"
        nhiet = f" {a['temperature']} độ" if "temperature" in a else ""
        vung = TEN_VUNG.get(a.get("zone", "all"), "")
        if la_suoi(ten, a, cau):
            return f"bật sưởi{nhiet} cho ấm"
        if a.get("mode") == "cool":
            return f"bật làm lạnh{nhiet}{vung}"
        if a.get("mode") == "auto":
            return f"chuyển điều hoà sang tự động{nhiet}{vung}"
        return f"bật điều hoà{nhiet}{vung}"
    if ten == "set_volume" and a.get("delta"):
        return f"{'giảm' if a['delta'] < 0 else 'tăng'} âm lượng {abs(a['delta'])} mức"
    if ten == "set_fan_speed" and a.get("delta"):
        return f"{'giảm' if a['delta'] < 0 else 'tăng'} quạt gió {abs(a['delta'])} mức"
    if ten == "control_window":
        dong_tu = "hạ" if a.get("position") == "down" else "nâng"
        muc = f" {a['amount']}%" if a.get("amount", 100) < 100 else ""
        return f"{dong_tu} {TEN_KINH.get(a.get('window', 'all'), 'kính')}{muc}"
    return {
        "get_fuel_level": "kiểm tra mức xăng",
        "get_battery_range": "kiểm tra pin",
        "get_tire_pressure": "kiểm tra áp suất lốp",
        "get_vehicle_status": "kiểm tra tình trạng xe",
        "get_odometer": "xem số km xe đã đi",
        "get_range_estimate": "tính quãng đường còn đi được",
        "set_fan_speed": f"chỉnh quạt gió mức {a.get('level', '')}".strip(),
        "set_defrost": ("tắt" if a.get("state") == "off" else "bật") + " sấy kính",
        "lock_doors": "khoá cửa xe",
        "unlock_doors": "mở khoá cửa xe",
        "open_trunk": "mở cốp sau",
        "close_trunk": "đóng cốp sau",
        "open_charge_port": "mở nắp cổng sạc",
        "open_fuel_flap": "mở nắp bình xăng",
        "operate_sunroof": TEN_CUA_SO_TROI.get(a.get("action", ""), "điều chỉnh cửa sổ trời"),
        "set_headlights": TEN_DEN.get(a.get("mode", ""), "chỉnh đèn pha"),
        "set_hazard_lights": ("tắt" if a.get("state") == "off" else "bật") + " đèn khẩn cấp",
        "set_volume": "chỉnh âm lượng",
        "play_media": "phát nhạc",
        "navigate_to": f"dẫn đường tới {a.get('destination', '')}".strip(),
        "find_nearby": f"tìm {TEN_DIA_DIEM.get(a.get('category', ''), 'địa điểm')} gần đây",
        "get_weather": "xem thời tiết",
    }.get(ten, "làm việc đó")


def bao_cham_gioi_han(ten: str, a: dict[str, Any], truoc: dict[str, Any], sau: dict[str, Any]) -> str | None:
    """Lệnh tăng / giảm TƯƠNG ĐỐI (`delta`) mà giá trị không đổi vì đã chạm giới hạn -> nói rõ, thay vì
    "Đã tắt quạt gió" khi quạt vốn đã tắt (người dùng gặp 2026-10-06). `truoc` / `sau` = ảnh chụp xe."""
    d = a.get("delta")
    if not d:
        return None
    if ten == "set_fan_speed" and truoc["fan_level"] == sau["fan_level"]:
        return "Quạt gió đang tắt rồi." if d < 0 else "Quạt gió đã ở mức mạnh nhất (7) rồi."
    if ten == "set_volume" and truoc["volume"] == sau["volume"]:
        return "Âm lượng đã nhỏ nhất rồi." if d < 0 else "Âm lượng đã ở mức lớn nhất (40) rồi."
    dung_im = truoc["ac_temperature"] == sau["ac_temperature"]
    if ten == "set_ac" and truoc["ac_on"] and sau["ac_on"] and dung_im:
        return (f"Điều hoà đã ở mức thấp nhất {sau['ac_temperature']} độ rồi." if d < 0
                else f"Điều hoà đã ở mức cao nhất {sau['ac_temperature']} độ rồi.")
    return None


def bao_ket_qua(ten: str, a: dict[str, Any], k: dict[str, Any], cau: str = "") -> str:
    """Câu báo kết quả — chỉ dùng số có trong kết quả `k`."""
    if la_suoi(ten, a, cau) and k.get("ac_on"):
        return f"Đã bật sưởi {k['temperature']} độ cho ấm."
    if ten == "set_ac" and k.get("ac_on") and (h := _dau(a.get("delta", 0)) or huong_nhiet(cau)):
        bien = " (thấp nhất)" if k["temperature"] <= 16 else " (cao nhất)" if k["temperature"] >= 30 else ""
        huong = "giảm nhiệt độ điều hoà xuống" if h < 0 else "tăng nhiệt độ điều hoà lên"
        return f"Đã {huong} {k['temperature']} độ{bien}."
    if ten == "get_fuel_level":
        cau = f"Xăng còn {k['fuel_percent']}%, đi được khoảng {_so(k['range_km'])} km."
        return cau + (" Bạn nên đổ xăng sớm." if k["needs_refuel"] else "")
    if ten == "get_battery_range":
        return f"Pin còn {k['battery_percent']}%, chạy điện được khoảng {_so(k['range_km'])} km."
    if ten == "get_tire_pressure":
        if k["low"]:
            non = ", ".join(TEN_KINH.get(b, b).replace("kính", "bánh") for b in k["low"])
            return f"{non.capitalize()} đang non, nên bơm lên {_so(k['recommended_psi'])} PSI."
        return "Áp suất lốp đều ổn."
    if ten == "get_vehicle_status":
        khoa = "đã khoá" if k["doors_locked"] else "chưa khoá"
        cop = "đang mở" if k["trunk_open"] else "đang đóng"
        return f"Cửa {khoa}, cốp {cop}, xăng còn {k['fuel_percent']}%, pin còn {k['battery_percent']}%."
    if ten == "get_odometer":
        return f"Xe đã đi {_so(k['odometer'])} {k['unit']}."
    if ten == "get_range_estimate":
        tong = _so(k["total_range_km"])
        if "trip_km" not in a:
            return f"Xe còn đi được khoảng {tong} km."
        if k["reachable"]:
            return f"Xe còn đi được khoảng {tong} km, đủ cho chuyến {_so(a['trip_km'])} km."
        return (
            f"Xe chỉ còn đi được khoảng {tong} km, chưa đủ cho chuyến {_so(a['trip_km'])} km, "
            "bạn nên tiếp nhiên liệu."
        )
    if ten == "set_ac":
        if not k["ac_on"]:
            return "Đã tắt điều hoà."
        vung = TEN_VUNG.get(k["zone"], "")
        if a.get("mode") == "cool":
            return f"Đã bật làm lạnh {k['temperature']} độ{vung}."
        if a.get("mode") == "auto":
            return f"Đã chuyển điều hoà sang tự động, {k['temperature']} độ{vung}."
        return f"Đã bật điều hoà {k['temperature']} độ{vung}."
    if ten == "set_fan_speed":
        if k["fan_level"] == 0:
            return "Đã tắt quạt gió."  # từ mức >0 xuống 0; đã tắt sẵn thì `bao_cham_gioi_han` nói trước
        if a.get("delta"):
            bien = " (mạnh nhất)" if k["fan_level"] >= 7 else ""
            dong_tu = "giảm quạt gió xuống" if a["delta"] < 0 else "tăng quạt gió lên"
            return f"Đã {dong_tu} mức {k['fan_level']}{bien}."
        return f"Đã chỉnh quạt gió mức {k['fan_level']}."
    if ten == "set_defrost":
        return f"Đã bật sấy {TEN_SAY[k['zone']]}." if k["defrost_on"] else "Đã tắt sấy kính."
    if ten == "control_window":
        ten_kinh = TEN_KINH[k["window"]]
        if k["position"] == "down":
            return f"Đã hạ {ten_kinh}, đang mở {k['opening_percent']}%."
        return (
            f"Đã nâng {ten_kinh}."
            if k["opening_percent"] == 0
            else f"Đã nâng {ten_kinh}, còn mở {k['opening_percent']}%."
        )
    if ten == "operate_sunroof":
        if k["sunroof_state"] == "close":
            return "Đã đóng cửa sổ trời."
        return f"Đã {TEN_CUA_SO_TROI[k['sunroof_state']]} {k['intensity_percent']}%."
    if ten == "set_headlights":
        return f"Đã {TEN_DEN[k['headlights']]}."
    if ten == "set_hazard_lights":
        return "Đã bật đèn khẩn cấp." if k["hazards_on"] else "Đã tắt đèn khẩn cấp."
    if ten == "set_volume":
        return f"Âm lượng đang ở mức {k['volume']}."
    if ten == "play_media":
        return f"Đang phát {k['now_playing']}."
    if ten == "navigate_to":
        return (
            f"Đã tìm đường tới {k['destination']}: khoảng {_so(k['distance_km'])} km, "
            f"mất chừng {round(k['eta_min'])} phút."
        )
    if ten == "find_nearby":
        loai = TEN_DIA_DIEM.get(k["category"], "địa điểm")
        if not k["results"]:
            return f"Không tìm thấy {loai} nào gần đây."
        return f"Tìm thấy {len(k['results'])} {loai} gần đây, gần nhất: {k['results'][0]}."
    if ten == "get_weather":
        ngay = {0: "hôm nay", 1: "ngày mai"}.get(a.get("day", 0), f"{a.get('day')} ngày tới")
        return f"Thời tiết {k['location']} {ngay}: {k['description']}, khoảng {_so(k['temp_c'])} độ."
    return {
        "lock_doors": "Đã khoá tất cả cửa.",
        "unlock_doors": "Đã mở khoá cửa.",
        "open_trunk": "Đã mở cốp sau.",
        "close_trunk": "Đã đóng cốp sau.",
        "open_charge_port": "Đã mở nắp cổng sạc.",
        "open_fuel_flap": "Đã mở nắp bình xăng.",
    }.get(ten, "Đã xong.")


def trang_thai_lien_quan(ten: str, s: Any) -> str:
    """Một câu ngắn về trạng thái xe liên quan tới việc sắp đề xuất — để người lái quyết đúng hơn
    (PLAN.md mục U: "câu hỏi lại kèm trạng thái kính/đèn"). `s` = `TrangThaiXe`; rỗng nếu không có gì
    đáng nói."""
    if s is None:
        return ""
    if ten == "control_window":
        mo = [(w, p) for w, p in s.windows.items() if p > 0]
        if not mo:
            return "Các kính đều đang đóng."
        return "Đang mở: " + ", ".join(f"{TEN_KINH[w]} {p}%" for w, p in mo) + "."
    if ten == "operate_sunroof":
        return "Cửa sổ trời đang đóng." if s.sunroof_state == "close" else (
            f"Cửa sổ trời đang {'mở' if s.sunroof_state == 'open' else 'hé'} {s.sunroof_percent}%.")
    if ten == "set_headlights":
        return f"Đèn pha đang ở chế độ {TEN_DEN_HIEN_TAI.get(s.headlights, s.headlights)}."
    if ten == "set_ac":
        if not s.ac_on:
            return "Điều hoà đang tắt."
        che_do = f" ({TEN_CHE_DO[s.ac_mode]})" if s.ac_mode in ("cool", "heat") else ""
        return f"Điều hoà đang bật {s.ac_temperature} độ{che_do}."
    if ten == "set_fan_speed":
        return "Quạt gió đang tắt." if s.fan_level == 0 else f"Quạt gió đang ở mức {s.fan_level}."
    return ""


TEN_DEN_HIEN_TAI = {"off": "tắt", "auto": "tự động", "low_beam": "chiếu gần", "high_beam": "chiếu xa",
                    "parking": "đèn đỗ"}


def _kem_trang_thai(loi_goi: list[tuple[str, dict[str, Any]]], s: Any, cau_goc: str) -> str:
    # đề xuất "bật sưởi" thì không nói "điều hoà đang tắt" (dễ hiểu nhầm là bật điều hoà làm lạnh)
    cau = dict.fromkeys(trang_thai_lien_quan(ten, s) for ten, a in loi_goi if not la_suoi(ten, a, cau_goc))
    return " ".join(c for c in cau if c)


def hoi_de_xuat(loi_goi: list[tuple[str, dict[str, Any]]], cau: str = "", s: Any = None) -> str:
    truoc = _kem_trang_thai(loi_goi, s, cau)
    truoc = truoc + " " if truoc else ""
    if len(loi_goi) == 1:
        return f"{truoc}Bạn có muốn mình {hanh_dong(*loi_goi[0], cau)} không?"
    return f"{truoc}Bạn muốn mình {hanh_dong(*loi_goi[0], cau)} hay {hanh_dong(*loi_goi[1], cau)}?"


def hoi_chon(loi_goi: list[tuple[str, dict[str, Any]]], cau: str = "") -> str:
    return f"Bạn chọn giúp mình: {hanh_dong(*loi_goi[0], cau)} hay {hanh_dong(*loi_goi[1], cau)}?"


def hoi_an_toan(ten: str, a: dict[str, Any], ly_do: str) -> str:
    return f"{ly_do} Bạn chắc chắn muốn mình {hanh_dong(ten, a)} không?"


def hoi_thieu(ten: str, a: dict[str, Any], thieu: list[str]) -> str:
    t = thieu[0]
    if ten == "control_window":
        if t == "window":
            return f"Bạn muốn {'hạ' if a.get('position') == 'down' else 'nâng'} kính cửa nào?"
        return "Bạn muốn hạ hay nâng kính?"
    cau = {
        "state": f"Bạn muốn bật hay tắt {hanh_dong(ten, {}).split(' ', 1)[-1]}?",
        "destination": "Bạn muốn đi đâu?",
        "category": "Bạn muốn tìm gì gần đây: cây xăng, trạm sạc, bãi đỗ, quán ăn…?",
        "level": "Bạn muốn quạt gió mức mấy, từ 0 đến 7?",
        "mode": "Bạn muốn đèn pha chế độ nào: tự động, chiếu gần hay chiếu xa?",
        "action": "Bạn muốn mở, đóng hay hé cửa sổ trời?",
        "muc_am_luong": "Bạn muốn to hơn hay nhỏ hơn?",
    }
    return cau.get(t, CHUA_RO)
