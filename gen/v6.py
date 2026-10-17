"""Vòng dữ liệu v6 — prompt cho teacher + luật code (PLAN.md Phase 7, "KẾ HOẠCH v6"). Thuần Python, test
ở máy.

Vì sao có vòng này (đo 2026-10-06 trên train v5): 7.882 câu gọi hàm chỉ có 0,7% câu ≤ 3 chữ, 14/24 hàm không
có câu ngắn nào; "pha" không kèm "đèn" 9 câu, "kính ghế lái / phụ" 0 câu, "kiếng" 0 câu -> người dùng gõ
"bật pha", "hạ kính ghế phụ" là model từ chối. Teacher tự viết thì quen tay câu đầy đủ, lịch sự, kiểu văn
viết.

Cách làm: GIAO thuộc tính cho từng lô thay vì để teacher tự chọn (theo AttrPrompt, Yu et al. 2023 — cần kiểm
lại trích dẫn trước khi viết luận văn): hàm × cách gọi bộ phận × tầng độ dài × giọng. Từ vựng dưới đây là cách
người Việt gọi các bộ phận / việc trên xe — viết từ hiểu biết chung, KHÔNG lấy từ tập test nào.

Bốn khối teacher (K1–K4) + các phép biến đổi bằng code (nhiễu, quy ước `delta`) đều ở đây; phần gọi GPU ở
gen/modal_app.py::sinh_v6.
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from gen.teacher_prompts import VAI_NGUOI_NOI, TeacherResponseError, _extract_json, _kiem_tra_tham_so
from schema.loader import CATALOG

# ── K1. ĐA DẠNG CÁCH NÓI — cả 24 hàm ─────────────────────────────────────────
# "goi": cách gọi bộ phận / việc (mỗi lô GIAO một cách) · "dong_tu": động từ / cách hỏi hay dùng ·
# "gia_tri": cách nói giá trị tham số (gợi ý cho teacher trộn vào, không bắt buộc).
TU_VUNG: dict[str, dict[str, list[str]]] = {
    "get_fuel_level": {
        "goi": ["xăng", "nhiên liệu", "bình xăng", "kim xăng", "mức xăng"],
        "dong_tu": ["còn bao nhiêu", "xem", "kiểm tra", "còn không", "sắp hết chưa"],
        "gia_tri": [],
    },
    "get_battery_range": {
        "goi": ["pin", "mức pin", "bình điện", "phần trăm pin", "điện"],
        "dong_tu": ["còn bao nhiêu", "xem", "kiểm tra", "đi được bao xa", "còn mấy phần trăm"],
        "gia_tri": [],
    },
    "get_tire_pressure": {
        "goi": ["lốp", "vỏ xe", "bánh xe", "áp suất lốp", "hơi lốp"],
        "dong_tu": ["xem", "kiểm tra", "đủ hơi không", "có non không", "bao nhiêu"],
        "gia_tri": ["bánh trước trái", "bánh sau phải", "cả bốn bánh"],
    },
    "get_vehicle_status": {
        "goi": ["tình trạng xe", "xe", "tổng quan xe", "các thông số xe", "trạng thái xe"],
        "dong_tu": ["xem", "kiểm tra", "báo cáo", "có ổn không", "thế nào rồi"],
        "gia_tri": [],
    },
    "get_odometer": {
        "goi": ["số km", "odo", "công tơ mét", "đồng hồ cây số", "tổng quãng đường xe đã chạy"],
        "dong_tu": ["xem", "bao nhiêu", "chạy được bao nhiêu cây rồi", "báo"],
        "gia_tri": ["tính bằng km", "tính bằng dặm"],
    },
    "get_range_estimate": {
        "goi": [
            "quãng đường còn đi được", "tầm hoạt động", "số cây còn chạy được", "xăng với pin còn đi được",
        ],
        "dong_tu": ["còn đi được bao xa", "đủ đi không", "chạy thêm được bao nhiêu cây"],
        "gia_tri": ["đi thêm 50 cây", "chuyến 200 km", "về quê 120 cây số"],
    },
    "set_ac": {
        "goi": ["điều hoà", "điều hòa", "máy lạnh", "lạnh", "A/C", "máy điều hoà"],
        "dong_tu": ["bật", "mở", "tắt", "chỉnh", "để", "cho", "hạ", "tăng", "giảm"],
        "gia_tri": ["22 độ", "về 24", "cho ghế lái", "bên phụ", "hàng ghế sau", "cả xe"],
    },
    "set_fan_speed": {
        "goi": ["quạt", "quạt gió", "gió", "sức gió", "gió điều hoà"],
        "dong_tu": ["tăng", "giảm", "to lên", "nhỏ lại", "mạnh lên", "nhẹ thôi", "tắt", "để"],
        "gia_tri": ["số 3", "mức 5", "mạnh nhất", "nhẹ nhất", "số 1"],
    },
    "set_defrost": {
        "goi": ["sấy kính", "sưởi kính", "khử sương", "khử mờ kính", "sấy kính hậu"],
        "dong_tu": ["bật", "mở", "tắt", "cho"],
        "gia_tri": ["kính trước", "kính sau", "cả hai kính"],
    },
    "lock_doors": {
        "goi": ["cửa", "xe", "chốt cửa", "khoá cửa"],
        "dong_tu": ["khoá", "khóa", "chốt", "bấm khoá", "cài khoá"],
        "gia_tri": [],
    },
    "unlock_doors": {
        "goi": ["khoá cửa", "chốt cửa", "cửa", "khoá xe"],
        "dong_tu": ["mở khoá", "mở khóa", "nhả khoá", "mở chốt", "bỏ khoá"],
        "gia_tri": ["cửa lái", "cửa ghế phụ", "cửa sau bên trái", "tất cả cửa"],
    },
    "open_trunk": {
        "goi": ["cốp", "cốp sau", "cốp xe", "khoang hành lý"],
        "dong_tu": ["mở", "bật", "nhả", "mở giúp"],
        "gia_tri": [],
    },
    "close_trunk": {
        "goi": ["cốp", "cốp sau", "cốp xe", "khoang hành lý"],
        "dong_tu": ["đóng", "sập", "hạ", "khép", "đóng lại"],
        "gia_tri": [],
    },
    "open_charge_port": {
        "goi": ["nắp sạc", "cổng sạc", "cửa sạc", "chỗ cắm sạc"],
        "dong_tu": ["mở", "bật", "nhả", "mở ra"],
        "gia_tri": [],
    },
    "open_fuel_flap": {
        "goi": ["nắp xăng", "nắp bình xăng", "cửa xăng", "chỗ đổ xăng"],
        "dong_tu": ["mở", "bật", "nhả", "mở ra"],
        "gia_tri": [],
    },
    "control_window": {
        "goi": ["kính", "kiếng", "cửa kính", "cửa sổ", "kính xe"],
        "dong_tu": ["hạ", "mở", "kéo xuống", "thả", "nâng", "đóng", "kéo lên", "khép"],
        "gia_tri": [
            "ghế lái", "bên tài", "phía tôi", "ghế phụ", "bên phụ", "sau trái", "sau phải", "hai kính sau",
            "hai kính trước", "tất cả", "một nửa", "hé chút", "30%", "hết cỡ",
        ],
    },
    "operate_sunroof": {
        "goi": ["cửa sổ trời", "cửa nóc", "nóc xe", "cửa trời"],
        "dong_tu": ["mở", "đóng", "hé", "kéo", "lật lên"],
        "gia_tri": ["một nửa", "hé thôi", "mở hết"],
    },
    "set_headlights": {
        "goi": ["đèn pha", "pha", "đèn", "đèn trước", "đèn cốt", "cốt", "đèn chiếu xa", "đèn chiếu gần"],
        "dong_tu": ["bật", "mở", "tắt", "chuyển", "hạ", "nâng", "để"],
        "gia_tri": ["pha xa", "cốt", "tự động", "đèn đỗ"],
    },
    "set_hazard_lights": {
        "goi": ["đèn khẩn cấp", "đèn cảnh báo", "xi nhan đôi", "đèn nháy", "đèn ưu tiên"],
        "dong_tu": ["bật", "mở", "tắt", "nháy", "cho"],
        "gia_tri": [],
    },
    "set_volume": {
        "goi": ["âm lượng", "tiếng", "loa", "nhạc", "âm thanh"],
        "dong_tu": ["to lên", "nhỏ lại", "vặn", "tăng", "giảm", "bớt", "để"],
        "gia_tri": ["mức 15", "thêm 3", "nhỏ tí", "to hết cỡ", "về 10"],
    },
    "play_media": {
        "goi": ["nhạc", "bài hát", "đài", "radio", "FM", "nhạc trong điện thoại", "USB"],
        "dong_tu": ["mở", "bật", "phát", "cho nghe", "chơi"],
        "gia_tri": ["nhạc Sơn Tùng", "bài Lạc trôi", "nhạc bolero", "đài giao thông", "nhạc không lời"],
    },
    "navigate_to": {
        "goi": ["đường", "chỉ đường", "dẫn đường", "bản đồ"],
        "dong_tu": ["đi", "về", "chỉ đường tới", "dẫn tới", "đến", "ra"],
        "gia_tri": ["Hồ Gươm", "sân bay Nội Bài", "nhà", "công ty", "bến xe Miền Đông", "chợ Bến Thành"],
    },
    "find_nearby": {
        "goi": ["cây xăng", "trạm sạc", "chỗ đỗ xe", "quán ăn", "bệnh viện", "cây ATM", "quán cà phê"],
        "dong_tu": ["tìm", "gần đây có", "chỗ nào có", "quanh đây có", "đâu có"],
        "gia_tri": ["gần nhất", "trong 2 km"],
    },
    "get_weather": {
        "goi": ["thời tiết", "trời", "mưa", "nắng", "nhiệt độ ngoài trời"],
        "dong_tu": ["xem", "thế nào", "có mưa không", "dự báo", "hôm nay ra sao"],
        "gia_tri": ["hôm nay", "ngày mai", "ở Đà Lạt", "chiều nay"],
    },
}

DO_DAI = {
    "rat_ngan": "1–3 chữ (cộc lốc, kiểu buột miệng khi đang lái)",
    "ngan": "4–6 chữ",
    "vua": "7–14 chữ",
}
TI_LE_DO_DAI = {"rat_ngan": 0.4, "ngan": 0.35, "vua": 0.25}  # PLAN.md: ≈40 / 35 / 25%
GIONG = [
    "cộc lốc, ra lệnh thẳng",
    "lịch sự, có 'giúp', 'nhé', 'được không'",
    "khẩu ngữ, có từ đệm (đi, cái, hộ, giùm, coi, thử)",
    "giọng miền Nam (vd: 'mở', 'coi', 'giùm', 'nghen', 'đi mà')",
    "giọng miền Bắc (vd: 'bật', 'xem', 'hộ', 'cái', 'nhé')",
]


@dataclass(frozen=True)
class LoK1:
    ham: str
    goi: str
    do_dai: str
    giong: str
    n: int


def lap_lo_k1(so_lo_moi_ham: int, cau_moi_lo: int, seed: int) -> list[LoK1]:
    """Chia lô cho K1: mỗi hàm `so_lo_moi_ham` lô, cách gọi xoay vòng đều, tầng độ dài theo tỉ lệ, giọng
    ngẫu nhiên."""
    rng = random.Random(seed)
    tang = [t for t, p in TI_LE_DO_DAI.items() for _ in range(round(p * 20))]
    lo: list[LoK1] = []
    for ham, tv in TU_VUNG.items():
        for i in range(so_lo_moi_ham):
            goi = tv["goi"][i % len(tv["goi"])]
            lo.append(LoK1(ham, goi, tang[i % len(tang)], rng.choice(GIONG), cau_moi_lo))
    rng.shuffle(lo)
    return lo


def build_da_dang_prompt(lo: LoK1) -> str:
    fn = CATALOG.by_name[lo.ham]
    tv = TU_VUNG[lo.ham]
    dong_tu = ", ".join(f'"{d}"' for d in tv["dong_tu"])
    gia_tri = ""
    if tv["gia_tri"]:
        gia_tri = f"\n- Có thể kèm chi tiết như: {', '.join(tv['gia_tri'])} (tuỳ câu, không bắt buộc)."
    return f"""Bạn là người tạo dữ liệu huấn luyện cho một trợ lý giọng nói trên xe ô tô.

Việc cần nhờ trợ lý làm: {fn.description.strip()}

Hãy viết {lo.n} câu tiếng Việt mà người lái THẬT SỰ nói với trợ lý để nhờ ĐÚNG việc trên.

Yêu cầu bắt buộc:
- MỌI câu phải gọi thứ đó bằng đúng cụm "{lo.goi}" (không thay bằng cách gọi khác).
- Độ dài MỖI câu: {DO_DAI[lo.do_dai]}.
- Giọng: {lo.giong}.
- Đổi động từ / cách hỏi giữa các câu, ví dụ các cách: {dong_tu}.{gia_tri}
- Câu nói thật ngoài đời, đúng chính tả, có dấu; không đánh số; {lo.n} câu không trùng khuôn.
- Không chèn từ tiếng Anh (trừ tên gọi quen như A/C, FM, USB, ATM, odo).

Chỉ trả về đúng JSON, không thêm chữ nào khác:
{{"queries": ["câu 1", "câu 2", ...]}}"""


def dung_do_dai(query: str, do_dai: str) -> bool:
    """Câu có đúng tầng độ dài đã giao không (nới 1 chữ mỗi phía — teacher đếm chữ không chính xác)."""
    n = len(query.split())
    return {"rat_ngan": 1 <= n <= 4, "ngan": 3 <= n <= 7, "vua": 6 <= n <= 16}[do_dai]


# ── K2. TÍNH NĂNG MỚI CỦA SCHEMA v6 ─────────────────────────────────────────
# (khoá, hàm, mô tả cho teacher, số lô) — teacher viết câu RA LỆNH; nhãn do bước trích tham số (biết schema
# mới)
TINH_NANG_MOI: list[tuple[str, str, str, int]] = [
    ("delta_so", "set_ac", "tăng hoặc giảm nhiệt độ điều hoà MỘT KHOẢNG có nói rõ số độ (vd thêm 2 độ, bớt 3 "
                           "độ)", 20),
    ("delta_khong_so", "set_ac", "tăng hoặc giảm nhiệt độ điều hoà nhưng KHÔNG nói số (cho mát / ấm hơn "
                                  "chút, giảm nhiệt độ xuống tí)", 20),
    ("ve_so", "set_ac", "đặt điều hoà VỀ / XUỐNG / LÊN một nhiệt độ cụ thể có số (vd xuống 20 độ, về 24)", 8),
    ("suoi", "set_ac", "bật SƯỞI ẤM bằng điều hoà (có thể kèm số độ hoặc vùng ghế) — KHÔNG phải sưởi ghế, "
                       "sưởi vô lăng, sưởi kính", 12),
    ("lam_lanh", "set_ac", "bật chế độ LÀM LẠNH của điều hoà (có thể kèm số độ)", 8),
    ("tu_dong", "set_ac", "chuyển điều hoà sang chế độ TỰ ĐỘNG", 4),
    ("hai_kinh_sau", "control_window", "hạ hoặc nâng CẢ HAI kính phía sau (hàng ghế sau)", 12),
    ("hai_kinh_truoc", "control_window", "hạ hoặc nâng CẢ HAI kính phía trước", 8),
]


def build_tinh_nang_moi_prompt(mo_ta: str, giong: str, n: int) -> str:
    return f"""Bạn là người tạo dữ liệu huấn luyện cho một trợ lý giọng nói trên xe ô tô.

Hãy viết {n} câu tiếng Việt mà người trong xe nói với trợ lý để: {mo_ta}.

Yêu cầu:
- Câu RA LỆNH / NHỜ THẲNG, nói thật ngoài đời. Giọng: {giong}.
- Độ dài trộn: có câu 2–4 chữ, có câu 5–10 chữ, có câu dài hơn chút.
- Đổi động từ, đổi cách gọi (điều hoà / điều hòa / máy lạnh; kính / kiếng / cửa kính / cửa sổ...).
- Đúng chính tả, có dấu; không đánh số; {n} câu không trùng khuôn.

Chỉ trả về đúng JSON, không thêm chữ nào khác:
{{"queries": ["câu 1", "câu 2", ...]}}"""


# Quy ước nhiệt độ (PLAN.md, bảng tra hàm): "về / xuống / lên N độ" -> temperature; "tăng / giảm N độ" ->
# delta=±N; KHÔNG nói số -> delta=±2. Teacher hay lệch quy ước này -> code sửa / loại cho nhất quán.
# chữ số chỉ tính khi đi kèm đơn vị: "giảm MỘT chút" là không nói số (đọc mẫu chạy thật: lọt quy ước ±2)
_CO_SO = re.compile(
    r"\d|\b(một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|mươi)\s+(độ|nấc|mức|số|đơn vị|vạch|bậc|cấp)\b"
)
_VE_SO = re.compile(r"\b(về|xuống|lên|tới|đến|ở|để|chỉnh|đặt)\s+(mức\s+)?\d+")
_DO = re.compile(r"\d+\s*(độ|°)")


# Đọc mẫu dựng thử: "bật điều hòa lạnh nhé" -> teacher gán delta=-2 (tự hạ 2 độ, người nói không bảo) -> chỉ
# giữ
# `delta` khi câu có từ chỉ THAY ĐỔI; chỉ giữ `mode` khi câu nói rõ chế độ (đúng mô tả tham số trong schema)
_THAY_DOI = re.compile(r"tăng|giảm|hạ|nâng|hơn|thêm|bớt|xuống|lên|nữa|chút|tí|xíu|\bđộ\b")
_CHE_DO = {"heat": re.compile(r"sưởi|\bấm\b|nóng lên"), "cool": re.compile(r"lạnh|mát"),
           "auto": re.compile(r"tự động|\bauto\b")}


def ap_quy_uoc_nhiet(query: str, args: dict[str, Any]) -> dict[str, Any] | None:
    """Sửa nhãn `set_ac` theo quy ước nhiệt độ; None = mâu thuẫn, nên LOẠI mẫu."""
    q = query.lower()
    a = dict(args)
    if "delta" in a and not _THAY_DOI.search(q):
        a.pop("delta")
    if a.get("mode") in _CHE_DO and not _CHE_DO[a["mode"]].search(q):
        a.pop("mode")
    if "temperature" in a and "delta" in a:
        return None
    if "delta" in a and not _CO_SO.search(q):
        a["delta"] = 2 if a["delta"] > 0 else -2
    if "delta" in a and _VE_SO.search(q) and _DO.search(q):
        return None  # "xuống 20 độ" mà teacher ghi delta -> mơ hồ, loại
    if "temperature" in a and not _CO_SO.search(q):
        return None  # không nói số mà ghi nhiệt độ cụ thể -> bịa
    return a


def tham_so_v6(raw: dict[str, Any], ham: str) -> dict[str, Any]:
    """Kiểm tham số như mọi vòng (`_kiem_tra_tham_so`) + một vá: câu chỉnh nhiệt / chế độ mà teacher quên
    `state` bắt buộc (chạy thử: "Giảm nhiệt độ đi mà" -> {"delta": -2}) -> `state="on"` (chỉnh nhiệt độ thì
    điều hoà phải đang chạy). Ném TeacherResponseError như cũ nếu vẫn sai."""
    a = dict(raw)
    if ham == "set_ac" and "state" not in a and any(k in a for k in ("delta", "temperature", "mode")):
        a["state"] = "on"
    return _kiem_tra_tham_so(a, CATALOG.by_name[ham])


def parse_tham_so_v6(text: str, ham: str) -> dict[str, Any]:
    """Như `parse_call_gen_response` (JSON {"arguments": {...}}) nhưng kiểm bằng `tham_so_v6`."""
    data = _extract_json(text)
    if not isinstance(data, dict) or not isinstance(data.get("arguments"), dict):
        raise TeacherResponseError(f"thiếu khoá 'arguments' dạng dict: {data!r}")
    return tham_so_v6(data["arguments"], ham)


# Quy ước quạt gió (người dùng chốt 2026-10-06 khi duyệt nhãn test v6 — `set_fan_speed.delta`): câu tăng /
# giảm TƯƠNG ĐỐI -> delta (không nói số -> ±1; "thêm 2 nấc" -> +2); mức tuyệt đối ("mức 5", "số 3", "lên 3",
# "mạnh nhất", "tắt") -> level.
_QUAT_TUYET_DOI = re.compile(
    r"\b(mức|số|nấc)\s*(\d+|một|hai|ba|bốn|năm|sáu|bảy)\b|\b(về|ở|để)\s+(mức\s+|số\s+)?\d+|nhất\b|\bmax\b|"
    r"hết cỡ|\btắt\b|\bkhông\s+(mức|số)\b|\b(lên|xuống)\s+\d+\b(?!\s*(nấc|mức|cấp|số))"
)
_QUAT_GIAM = re.compile(
    r"\b(giảm|bớt|hạ|nhẹ lại|nhẹ bớt|nhẹ đi|nhẹ hơn|nhỏ lại|nhỏ bớt|yếu đi|yếu lại|êm lại|chậm lại)\b"
)
_QUAT_TANG = re.compile(
    r"\b(tăng|thêm|mạnh lên|mạnh hơn|mạnh thêm|to lên|to hơn|lớn lên|lớn hơn|nâng|nhanh lên)\b"
)
_SO_NAC = re.compile(r"\b(\d+|một|hai|ba|bốn|năm)\s+(nấc|mức|số|cấp)\b")
_SO_CHU = {"một": 1, "hai": 2, "ba": 3, "bốn": 4, "năm": 5}


def ap_quy_uoc_quat(query: str, args: dict[str, Any]) -> dict[str, Any]:
    """Nhãn `set_fan_speed` của câu RA LỆNH -> đúng quy ước level / delta. Câu không rõ hướng -> giữ
    nguyên."""
    q = query.lower()
    if "delta" in args:  # teacher đã ghi tương đối: không nói số -> ±1
        return args if _SO_NAC.search(q) else {"delta": 1 if args["delta"] > 0 else -1}
    if _QUAT_TUYET_DOI.search(q):
        return args
    huong = -1 if _QUAT_GIAM.search(q) else 1 if _QUAT_TANG.search(q) else 0
    if not huong:
        return args
    m = _SO_NAC.search(q)
    n = 1 if m is None else (int(m.group(1)) if m.group(1).isdigit() else _SO_CHU[m.group(1)])
    return {"delta": huong * n}


# Quy ước âm lượng (có từ vòng 3, `gen/sua_nhan.py`: "Vặn nhỏ tiếng xuống" -> delta=-3): delta KHÔNG nói số
# -> ±3
BUOC_AM_LUONG = 3


def ap_quy_uoc_am_luong(query: str, args: dict[str, Any]) -> dict[str, Any]:
    if "delta" in args and not _CO_SO.search(query.lower()):
        return {**args, "delta": BUOC_AM_LUONG if args["delta"] > 0 else -BUOC_AM_LUONG}
    return args


def ap_quy_uoc(ham: str, query: str, args: dict[str, Any]) -> dict[str, Any] | None:
    """Mọi luật code sau khi teacher ra nhãn; None = loại mẫu."""
    if ham == "set_ac":
        return ap_quy_uoc_nhiet(query, args)
    if ham == "set_headlights":
        return ap_quy_uoc_den(query, args)
    if ham == "set_fan_speed":
        return ap_quy_uoc_quat(query, args)
    if ham == "set_volume":
        return ap_quy_uoc_am_luong(query, args)
    return args


# Quy ước đèn (bảng tra hàm): "pha" / "chiếu xa" -> high_beam; "cốt" / "chiếu gần" -> low_beam. Chạy thử:
# teacher gán "bật đèn pha" lúc high_beam (trích tham số) lúc low_beam (gán nhãn hai lượt) -> ép theo chữ
# trong câu.
_DEN_GAN = re.compile(r"\bcốt\b|chiếu gần|\bgần\b")
_DEN_XA = re.compile(r"\bpha\b|chiếu xa|\bxa\b")


def ap_quy_uoc_den(query: str, args: dict[str, Any]) -> dict[str, Any]:
    q = query.lower()
    if args.get("mode") not in ("low_beam", "high_beam"):
        return args
    if _DEN_GAN.search(q):
        return {**args, "mode": "low_beam"}
    if _DEN_XA.search(q):
        return {**args, "mode": "high_beam"}
    return args


# ── K3. CẶP ĐỐI XỨNG với câu từ chối "sát nách" ─────────────────────────────
# (hàm làm ĐƯỢC, cụm phải dùng, việc KHÔNG làm được dễ nhầm). Lấy từ TINH_NANG_KHONG_HO_TRO (teacher_prompts —
# suy từ VehiclePropertyIds) và lỗi đã thấy khi người dùng thử demo ("hạ kính ghế phụ" bị hiểu thành chỉnh
# ghế).
CAP_DOI_XUNG: list[tuple[str, str, str]] = [
    ("control_window", "kính + ghế lái / ghế phụ / ghế sau (vd kính ghế phụ, kính chỗ ghế lái)",
     "chỉnh ghế, ngả ghế"),
    ("set_ac", "điều hoà / máy lạnh + ghế lái / ghế phụ / hàng ghế sau",
     "sưởi ghế, làm mát ghế, massage ghế"),
    ("set_ac", "sưởi / sưởi ấm bằng điều hoà", "sưởi ghế, sưởi vô lăng"),
    ("set_defrost", "sưởi kính / sấy kính", "sưởi gương, sưởi ghế"),
    ("set_headlights", "đèn pha / đèn trước / pha / cốt", "đèn trần, đèn trong xe, đèn viền"),
    ("open_trunk", "cốp sau / cốp", "cốp trước, nắp capo"),
    ("set_fan_speed", "quạt gió / gió điều hoà mạnh nhẹ", "hướng gió, cửa gió, lấy gió trong / ngoài"),
    ("play_media", "mở nhạc / bài hát / đài", "chuyển bài tiếp, tua nhạc, gọi điện"),
    ("navigate_to", "dẫn đường / đi tới một nơi", "huỷ lộ trình, hỏi mấy giờ tới"),
    ("operate_sunroof", "cửa sổ trời", "rèm che nắng, rèm cửa sổ trời"),
]


def build_doi_xung_prompt(ham: str, cum: str, de_nham: str, giong: str, n: int) -> str:
    fn = CATALOG.by_name[ham]
    return f"""Bạn là người tạo dữ liệu huấn luyện cho một trợ lý giọng nói trên xe ô tô.

Việc trợ lý LÀM ĐƯỢC: {fn.description.strip()}
Việc trợ lý KHÔNG làm được và rất dễ nhầm với việc trên: {de_nham}.

Hãy viết {n} câu RA LỆNH nhờ đúng việc LÀM ĐƯỢC ở trên, trong đó MỖI câu phải có cụm: {cum}.
Mục đích: câu nghe gần giống việc KHÔNG làm được nhưng thật ra là việc LÀM ĐƯỢC.

Yêu cầu:
- Giọng: {giong}. Độ dài trộn: có câu 3–5 chữ, có câu 6–12 chữ.
- TUYỆT ĐỐI không viết câu nhờ việc KHÔNG làm được.
- Đúng chính tả, có dấu; không đánh số; {n} câu không trùng khuôn.

Chỉ trả về đúng JSON, không thêm chữ nào khác:
{{"queries": ["câu 1", "câu 2", ...]}}"""


# ── K4. HAI LƯỢT (ngữ cảnh lượt trước) ──────────────────────────────────────
# Định dạng prompt cho student (CATALOG.render_call_prompt(..., luot_truoc=...)):
# "Lượt trước: <nexa_i>(...)\n<câu>".
KIEU_NOI_TIEP: dict[str, str] = {
    "doi_chieu": "bảo làm NGƯỢC LẠI việc vừa làm, nói trống không, KHÔNG nhắc lại tên bộ phận (kiểu: tắt đi, "
                 "đóng lại, mở lại, thôi tắt)",
    "them_nua": "bảo làm THÊM / MẠNH HƠN / NHIỀU HƠN theo đúng hướng vừa làm, KHÔNG nhắc tên bộ phận (kiểu: "
                "nữa đi, thêm chút nữa, to nữa)",
    "bo_phan_khac": "bảo làm việc tương tự cho một vị trí / bộ phận KHÁC cùng loại, nói tắt dựa vào câu "
                    "trước (kiểu: cả bên kia nữa, phía sau thì sao, ghế phụ cũng thế)",
    "lenh_moi": "một lệnh MỚI, nói ĐẦY ĐỦ, KHÔNG liên quan câu trước (nhờ một việc xe làm được khác hẳn)",
    "ngoai_pham_vi": "nhờ một việc mà xe / trợ lý KHÔNG làm được, hoặc chỉ tán gẫu",
}
# kiểu nối tiếp hợp lệ cho từng hàm lượt 1 (giống tập test v6: "làm thêm" sau "mở cốp" là vô lý)
HAM_LUOT_1: dict[str, list[str]] = {
    "set_ac": ["doi_chieu", "them_nua", "bo_phan_khac", "lenh_moi", "ngoai_pham_vi"],
    "set_fan_speed": ["doi_chieu", "them_nua", "lenh_moi", "ngoai_pham_vi"],
    "set_defrost": ["doi_chieu", "bo_phan_khac", "lenh_moi", "ngoai_pham_vi"],
    "control_window": ["doi_chieu", "them_nua", "bo_phan_khac", "lenh_moi", "ngoai_pham_vi"],
    "operate_sunroof": ["doi_chieu", "them_nua", "lenh_moi", "ngoai_pham_vi"],
    "open_trunk": ["doi_chieu", "lenh_moi", "ngoai_pham_vi"],
    "close_trunk": ["doi_chieu", "lenh_moi", "ngoai_pham_vi"],
    "lock_doors": ["doi_chieu", "lenh_moi", "ngoai_pham_vi"],
    "unlock_doors": ["doi_chieu", "lenh_moi", "ngoai_pham_vi"],
    "set_headlights": ["doi_chieu", "lenh_moi", "ngoai_pham_vi"],
    "set_hazard_lights": ["doi_chieu", "lenh_moi", "ngoai_pham_vi"],
    "set_volume": ["doi_chieu", "them_nua", "lenh_moi", "ngoai_pham_vi"],
    "play_media": ["lenh_moi", "ngoai_pham_vi"],
    "navigate_to": ["lenh_moi", "ngoai_pham_vi"],
}


def build_noi_tiep_prompt(cau_1: str, goi_1: str, kieu: str, giong: str, n: int) -> str:
    return f"""Bạn là người tạo dữ liệu huấn luyện cho một trợ lý giọng nói trên xe ô tô.

Người lái vừa nói: "{cau_1}"
Trợ lý đã làm: {goi_1}

Hãy viết {n} câu khác nhau mà người lái có thể nói NGAY SAU đó, mỗi câu: {KIEU_NOI_TIEP[kieu]}.

Yêu cầu:
- Câu phải tự nhiên khi đứng ngay sau câu trước. Giọng: {giong}. Ngắn gọn như nói thật (thường 1–8 chữ).
- Đúng chính tả, có dấu; không đánh số; không trùng khuôn.

Chỉ trả về đúng JSON, không thêm chữ nào khác:
{{"queries": ["câu 1", "câu 2", ...]}}"""


def _catalog_doc() -> str:
    return "\n\n".join(f.teacher_doc() for f in CATALOG.functions)


def build_nhan_noi_tiep_prompt(cau_1: str, goi_1: str, cau_2: str) -> str:
    """Teacher gán nhãn câu lượt 2 THEO NGỮ CẢNH lượt 1 — KHÔNG biết kiểu nối tiếp đã giao (để tự hiểu)."""
    return f"""Bạn là trợ lý điều khiển xe ô tô. Dưới đây là các chức năng bạn có:

{_catalog_doc()}

Hội thoại:
- Người lái (lượt 1): "{cau_1}"
- Bạn đã làm: {goi_1}
- Người lái (lượt 2): "{cau_2}"

Câu lượt 2 cần gọi hàm nào, với tham số gì? HIỂU THEO NGỮ CẢNH lượt 1 (vd lượt 1 mở cốp, lượt 2 "đóng đi"
-> close_trunk). Nếu lượt 2 là việc không có trong danh sách chức năng hoặc chỉ tán gẫu ->
irrelevant_function.
Chỉ điền tham số suy ra chắc chắn; tham số bắt buộc phải đủ. Nhiệt độ không nói số -> delta=-2 hoặc delta=2.

Chỉ trả về đúng JSON, không thêm chữ nào khác:
{{"ham": "tên_hàm", "tham_so": {{"tên": giá_trị, ...}}}}"""


def parse_nhan_noi_tiep(text: str) -> tuple[str, dict[str, Any]]:
    data = _extract_json(text)
    if not isinstance(data, dict) or not isinstance(data.get("ham"), str):
        raise TeacherResponseError(f"thiếu 'ham': {data!r}")
    ten = data["ham"].strip()
    if ten not in CATALOG.by_name:
        raise TeacherResponseError(f"hàm không tồn tại: {ten!r}")
    tham_so = data.get("tham_so") or {}
    if not isinstance(tham_so, dict):
        raise TeacherResponseError(f"'tham_so' không phải dict: {tham_so!r}")
    return ten, tham_so


def build_verify_noi_tiep_prompt(cau_1: str, goi_1: str, cau_2: str, goi_2: str) -> str:
    return f"""Bạn là người kiểm duyệt dữ liệu huấn luyện cho trợ lý điều khiển xe ô tô.

Hội thoại:
- Người lái (lượt 1): "{cau_1}"
- Trợ lý đã làm: {goi_1}
- Người lái (lượt 2): "{cau_2}"
- Trợ lý định làm cho lượt 2: {goi_2}

Việc trợ lý định làm cho lượt 2 có ĐÚNG ý người lái không, khi hiểu theo ngữ cảnh lượt 1? Tham số có sai,
có bịa thêm điều người lái không nói không? (Tham số tuỳ chọn bỏ trống là đúng. irrelevant_function = từ
chối, đúng khi lượt 2 nhờ việc xe không làm được hoặc chỉ tán gẫu.)

Chỉ trả về đúng JSON, không thêm chữ nào khác:
{{"valid": true hoặc false, "reason": "lý do ngắn gọn, tiếng Việt"}}"""


def chon_giong(rng: random.Random) -> str:
    """Giọng cho K2–K4: trộn giọng K1 với vai người nói của các vòng trước."""
    return rng.choice(GIONG + VAI_NGUOI_NOI)


# ── K5. NHIỄU BẰNG CODE (không dấu, lỗi gõ) ─────────────────────────────────
def bo_dau(s: str) -> str:
    """'Mở cốp giùm' -> 'Mo cop gium' (cả đ -> d) — kiểu gõ không dấu."""
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


_LOI_GO = ("lap", "thieu", "doi_cho", "bo_mot_dau")


def loi_go(s: str, rng: random.Random) -> str:
    """Một lỗi gõ thường gặp ở MỘT chữ dài ≥ 3 ký tự: lặp chữ cái, thiếu chữ cái, đảo hai chữ cái, mất dấu
    một chữ."""
    tu = s.split()
    vi_tri = [i for i, t in enumerate(tu) if len(t) >= 3 and t.isalpha()]
    if not vi_tri:
        return s
    i = rng.choice(vi_tri)
    t = tu[i]
    kieu = rng.choice(_LOI_GO)
    j = rng.randrange(1, len(t) - 1)
    if kieu == "lap":
        t = t[: j + 1] + t[j] + t[j + 1 :]
    elif kieu == "thieu":
        t = t[:j] + t[j + 1 :]
    elif kieu == "doi_cho":
        t = t[:j] + t[j + 1] + t[j] + t[j + 2 :]
    else:
        t = bo_dau(t)
    tu[i] = t
    return " ".join(tu)


def them_nhieu(s: str, rng: random.Random) -> tuple[str, str]:
    """Trả về (câu đã thêm nhiễu, kiểu nhiễu): ~50% bỏ dấu cả câu, ~35% một lỗi gõ, ~15% cả hai."""
    r = rng.random()
    if r < 0.5:
        return bo_dau(s), "bo_dau"
    if r < 0.85:
        return loi_go(s, rng), "loi_go"
    return bo_dau(loi_go(s, rng)), "bo_dau_loi_go"


