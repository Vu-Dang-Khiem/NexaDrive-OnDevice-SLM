"""Bộ lọc chương trình cho dữ liệu sinh ra — chạy local, không cần GPU.

Hai lỗi có thật đã tìm thấy trong dataset đầu tiên (full2, 27.327 mẫu):

1. **Lẫn chữ Trung/Nhật/Hàn** (119 mẫu): Qwen là model Trung Quốc nên thỉnh thoảng
   chèn chữ Hán vào câu tiếng Việt — "Đóng cốp sau xe lại đi,谢谢.",
   "Dẫn tôi đến bệnh viện Từ Dũ, tránh cao速.", và tệ nhất là lọt vào cả THAM SỐ:
   `navigate_to(destination="火車站")`. Train thẳng thì model học cách chèn chữ lạ.

2. **Lộ tên hàm** (target leakage): nếu câu người dùng chứa luôn tên hàm ("bật set_ac
   lên") thì bài toán trở nên dễ giả tạo — model chỉ cần khớp chuỗi. Cách kiểm này lấy
   từ Jafarzadeh Asl et al. 2026 (§A.7): họ loại hoặc sinh lại các câu bị lộ, và áp dụng
   cùng tiêu chí đó trước khi chấm điểm.

Nguyên tắc chung (cũng theo paper trên): thứ gì CODE kiểm được thì đừng nhờ model chấm.
"""

from __future__ import annotations

import re
from typing import Any

from schema.loader import CATALOG

# CJK thống nhất, Hiragana/Katakana, Hangul, và các khối mở rộng hay gặp.
_CHU_DONG_A = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿ｦ-ﾟ]")


def co_chu_dong_a(text: str) -> bool:
    """True nếu chuỗi có ký tự Trung/Nhật/Hàn — dataset tiếng Việt không được phép có."""
    return bool(_CHU_DONG_A.search(text))


# Hư từ tiếng Anh — CHỈ những từ không trùng âm tiết tiếng Việt ("an" trong "an toàn",
# "to" = to lớn, "can" đều bị bỏ ra). Chỉ hư từ mới là dấu hiệu câu lai ngữ pháp tiếng Anh;
# từ mượn (km, USB, playlist, Bluetooth, sunroof, radio) là cách người Việt nói THẬT nên
# không được đụng vào. Đo trên 20.170 mẫu: 2.219 câu có từ không phải âm tiết Việt, gần
# hết là từ mượn; câu lai thật chỉ ~21, toàn khuôn "bạn có thể telling me about your...".
_HU_TU_ANH = {
    "the", "is", "are", "was", "my", "your", "you", "about", "this", "that", "it", "of",
    "and", "for", "with", "please", "what", "how", "turn", "air", "cold", "hot", "telling",
    "tell", "want", "open", "close", "me",
}
_TRONG_NGOAC = re.compile(r"[\"'“”‘’«][^\"'“”‘’»]*[\"'“”‘’»]")
_TU = re.compile(r"[A-Za-zÀ-ỹđĐ]+")
# Tên bài hát, địa danh, tên quán thường là tiếng Anh mà không có ngoặc ("You Are My Sunshine").
HAM_MIEN_LAI_TIENG_ANH = {"play_media", "navigate_to", "find_nearby"}


def lai_tieng_anh(text: str, toi_thieu: int = 2) -> bool:
    """True nếu câu có từ `toi_thieu` hư từ tiếng Anh KHÁC NHAU trở lên (ngoài ngoặc kép).

    Cần ≥2 chứ không phải 1: tên sách/tên riêng ("War and Peace", "The Iliad") có đúng 1
    hư từ và là câu hợp lệ; câu lai thật luôn có nhiều ("telling me about your").
    """
    bo_ngoac = _TRONG_NGOAC.sub(" ", text)
    tu = {w.lower() for w in _TU.findall(bo_ngoac) if w.isascii()}
    return len(tu & _HU_TU_ANH) >= toi_thieu


# Âm tiết tiếng Việt viết không dấu vẫn là chữ ASCII ("con", "sao", "quanh", "nghe") — nên KHÔNG
# thể coi mọi từ ASCII là tiếng Anh. Thay vào đó kiểm cấu trúc âm tiết: phụ âm đầu + 1–3 nguyên âm +
# phụ âm cuối hợp lệ. Từ tiếng Anh gần như luôn vỡ khuôn này ("meetings", "forget", "water", "mood").
_AM_TIET_VIET = re.compile(
    r"^(ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|[bcdghklmnprstvx])?[aeiouy]{1,3}(ng|nh|ch|[cmnpt])?$"
)
# Từ mượn người Việt nói thật, giữ lại (cùng tinh thần `_HU_TU_ANH`: không đụng vào từ mượn quen).
TU_MUON_QUEN = {
    "bluetooth", "usb", "atm", "km", "gps", "radio", "wifi", "ok", "gara", "taxi", "cafe", "video",
    "camera", "playlist", "album", "app", "pin", "tivi", "led", "ac",
}


def tu_tieng_anh(text: str) -> list[str]:
    """Các từ ASCII trong câu KHÔNG phải âm tiết tiếng Việt và không phải từ mượn quen.

    Dùng cho câu HỎI LẠI kể hoàn cảnh (PLAN.md mục U): chạy thử thấy lọt "Đi meetings từ sáng",
    "lỡ Forget water vô xe", "mood nó ủ rũ" — `lai_tieng_anh` (đòi ≥ 2 hư từ) không bắt được.
    KHÔNG áp cho câu gọi hàm (tên bài hát, địa danh tiếng Anh là hợp lệ ở đó).
    """
    bo_ngoac = _TRONG_NGOAC.sub(" ", text)
    return [
        w for w in _TU.findall(bo_ngoac)
        if w.isascii() and w.lower() not in TU_MUON_QUEN and not _AM_TIET_VIET.match(w.lower())
    ]


def _cac_ten_ham() -> list[str]:
    return [f.name for f in CATALOG.functions]


def lo_ten_ham(text: str) -> str | None:
    """Trả về tên hàm bị lộ trong câu người dùng, hoặc None nếu sạch.

    Chỉ bắt tên hàm dạng mã (`set_ac`, `open_trunk`) và token `<nexa_`, KHÔNG bắt từ
    tiếng Việt bình thường — người dùng nói "bật điều hoà" là hoàn toàn hợp lệ và không
    phải lộ nhãn.
    """
    thap = text.lower()
    for ten in _cac_ten_ham():
        if ten in thap:
            return ten
    if "<nexa_" in thap:
        return "<nexa_"
    return None


def cau_hop_le(query: str) -> tuple[bool, str]:
    """Kiểm 1 câu người dùng trước khi đưa vào dataset. Trả về (hợp lệ, lý do nếu loại)."""
    if not query.strip():
        return False, "câu rỗng"
    if co_chu_dong_a(query):
        return False, "lẫn chữ Trung/Nhật/Hàn"
    ten = lo_ten_ham(query)
    if ten is not None:
        return False, f"lộ tên hàm: {ten}"
    return True, ""


def tham_so_hop_le(args: dict[str, Any]) -> tuple[bool, str]:
    """Kiểm giá trị tham số — đã gặp thật: destination=\"火車站\"."""
    for k, v in args.items():
        if isinstance(v, str) and co_chu_dong_a(v):
            return False, f"tham số {k} lẫn chữ Trung/Nhật/Hàn"
    return True, ""


_KHOANG_CACH = re.compile(r"([\d.]+)\s*km")


def ket_qua_nhat_quan(row: dict[str, Any]) -> tuple[bool, str]:
    """Kiểm mẫu `reply`: kết quả hàm có mâu thuẫn với chính yêu cầu của người dùng không.

    Đã gặp thật (thẩm định chéo bắt được): "tìm quán cà phê trong bán kính 1km" nhưng kết
    quả mô phỏng trả về địa điểm 4.9 km — câu trả lời sinh từ đó dạy model nói ngược ý
    người dùng. Lỗi gốc đã sửa trong gen/simulate.py; hàm này dọn nốt mẫu cũ.
    """
    if row["meta"].get("type") != "reply" or row["meta"].get("fn") != "find_nearby":
        return True, ""
    from gen.records import result_from_reply_prompt
    from schema.call_format import parse_calls

    try:
        call = parse_calls(row["prompt"].split("<|im_start|>assistant\n")[1])[0]
        ket_qua = result_from_reply_prompt(row["prompt"])
    except (ValueError, IndexError):
        return True, ""
    ban_kinh = call.args.get("radius_km")
    if ban_kinh is None:
        return True, ""
    xa = [float(x) for x in _KHOANG_CACH.findall(str(ket_qua.get("results", "")))]
    if any(x > float(ban_kinh) for x in xa):
        return False, f"kết quả {xa} vượt bán kính {ban_kinh} km người dùng xin"
    return True, ""


def tra_loi_khong_bia_so(row: dict[str, Any]) -> tuple[bool, str]:
    """Mẫu `reply` không được chứa số mà kết quả hàm lẫn câu hỏi đều không có.

    Model bịa số trên xe rất nguy hiểm ("xăng còn 60%" trong khi thực tế 12%), nên mẫu
    dạy nó cũng không được phép sai. Luật nới lỏng (làm tròn, ngăn nghìn, dấu âm, số lấy
    từ câu hỏi) nằm trong eval/metrics.khong_bia_so.
    """
    if row["meta"].get("type") != "reply":
        return True, ""
    from eval.metrics import khong_bia_so
    from gen.records import query_from_call_prompt, result_from_reply_prompt

    try:
        ket_qua = result_from_reply_prompt(row["prompt"])
        query = query_from_call_prompt(row["prompt"])
    except ValueError:
        return True, ""
    tra_loi = row["completion"].replace("<|im_end|>", "")
    if not khong_bia_so(tra_loi, ket_qua, query):
        return False, "câu trả lời có số không truy được về kết quả hàm (bịa số)"
    return True, ""


_NOI_KHAC = r"(?!\s+(?:nhà|phòng|công ty|văn phòng|chính|ngoài))"
_LENH_HAM_THAT = re.compile(
    rf"\b(?:khoá|khóa|mở khoá|mở khóa)\s+(?:hết\s+|kỹ\s+)?cửa{_NOI_KHAC}(?:\s+xe)?\b"
    r"|\b(?:mở|đóng)\s+cốp"
    rf"|\b(?:mở|hạ|nâng|đóng|kéo)\s+(?:hết\s+)?(?:kính|cửa sổ)(?!\s+trời){_NOI_KHAC}"
    r"|\b(?:mở|đóng|hé)\s+cửa sổ trời"
    r"|\b(?:bật|tắt|mở)\s+(?:cái\s+)?"
    rf"(?:điều hoà|điều hòa|máy lạnh|đèn pha|đèn khẩn cấp|sấy kính|quạt gió){_NOI_KHAC}"
    r"|\bmở\s+nắp\s+(?:sạc|bình xăng|cổng sạc)"
    r"|\b(?:tăng|giảm|vặn)\s+(?:nhỏ\s+|to\s+)?(?:âm lượng|loa)"
    r"|\bdẫn đường\b|\b(?:phát|mở|bật)\s+(?:nhạc|bài|đài)",
    re.IGNORECASE,
)


def co_lenh_ham_that(cau: str) -> str | None:
    """Câu TỪ CHỐI mà chứa cụm lệnh việc tác tử LÀM ĐƯỢC -> câu ghép, nhãn "từ chối" là SAI một nửa.

    Đã thấy ở lần thử 2 câu từ chối kể hoàn cảnh: "Đóng cốp sau lại, …, đặt hộ tôi suất phở" —
    gán `irrelevant_function` là dạy model từ chối "đóng cốp". Bắt thô theo cụm động từ + đối
    tượng; thà bỏ oan vài câu kể chuyện ("vừa mở nắp sạc xong") còn hơn dạy từ chối nhầm.
    Loại trừ đồ vật CHỖ KHÁC ("khoá cửa nhà", "tắt điều hoà phòng") — đó đúng là câu từ chối loại B.
    """
    m = _LENH_HAM_THAT.search(cau)
    return m.group(0) if m else None


_THOI_TIET = re.compile(
    r"thời tiết|dự báo|nhiệt độ\b.{0,30}\b(ngoài trời|bên ngoài|hôm nay|ngày mai)|có mưa không", re.IGNORECASE
)
_GAN_DAY = re.compile(r"gần đây|gần nhất|quanh đây|lân cận|xung quanh đây", re.IGNORECASE)


def co_viec_lam_duoc_an(cau: str) -> str | None:
    """Câu TỪ CHỐI nhưng lời nhờ thật ra là xem thời tiết / tìm chỗ gần đây — tác tử LÀM ĐƯỢC.

    Đã thấy ở lần thử 3 (PLAN.md mục P): nhóm "hỏi kiến thức" trượt sang "nhiệt độ ngoài trời bao nhiêu"
    (get_weather), "chỗ nào ngon gần đây" (find_nearby) và giám khảo vẫn giữ. Gán từ chối = dạy TỪ CHỐI NHẦM.
    "Thời tiết" chỉ xét VẾ CUỐI (lời nhờ): phần hoàn cảnh nhắc thời tiết là đúng BẪY TỪ KHOÁ cố ý tạo ("Thời
    tiết đẹp quá…, chuyển sang chế độ thể thao") — bản đầu xét cả câu đã bỏ oan đúng loại câu cần nhất.
    "Gần đây / gần nhất" xét CẢ CÂU: gần như không bao giờ là bẫy, và hay nằm giữa câu ("Google Map nói gần
    đây có quán cà phê, em quên tên, anh biết không?" — lọt khi chỉ xét vế cuối, đọc mẫu sinh thật thấy).
    Bắt thô — chấp nhận bỏ oan vài câu ("giá rửa xe ở tiệm gần nhất").
    """
    if (m := _GAN_DAY.search(cau)) is not None:
        return m.group(0)
    ve = [v for v in re.split(r"[,.;!?]", cau) if v.strip()]
    m = _THOI_TIET.search(ve[-1]) if ve else None
    return m.group(0) if m else None


_LOI_NHO = re.compile(
    r"\b(bật|tắt|mở|đóng|hạ|nâng|kéo|khoá|khóa|tăng|giảm|vặn|chỉnh|đặt|tìm|dẫn|chỉ đường|phát|mở nhạc|"
    r"kiểm tra|xem|coi|báo|cho biết|sấy|đổi|chuyển)\b|\?",
    re.IGNORECASE,
)


def co_loi_nho(cau: str) -> bool:
    """Câu GỌI HÀM phải có lời nhờ (động từ hành động hoặc câu hỏi). Câu chỉ than thở ("nóng quá, khó chịu
    ghê") mà gán nhãn gọi hàm thì mâu thuẫn với dữ liệu HỎI LẠI — câu than thở đúng ra phải hỏi lại."""
    return bool(_LOI_NHO.search(cau))


NGUONG_CUNG_KHUON = 0.3


def _tap_tu(s: str) -> set[str]:
    return set(re.findall(r"\w+", s.lower()))


def bo_cung_khuon(cau: list[str], nguong: float = NGUONG_CUNG_KHUON) -> list[int]:
    """Trả chỉ số các câu GIỮ LẠI: câu nào trùng TẬP TỪ >= `nguong` với một câu đã giữ thì bỏ.

    Khác lọc gần trùng của `gen/chia_tap.py` (MinHash, bắt câu gần như trùng chữ — dùng cho kiểm
    nhiễm). Ở đây bắt câu CÙNG KHUÔN đổi vài chữ, loại mà teacher hay viết liền trong một lô:
    "vừa đỗ xe, trời nắng, kéo rèm che nắng…" × 8. Ngưỡng 0,3 chọn trên 64 câu chạy thử (PLAN.md
    mục P): hai câu KHÁC lô trùng cao nhất 0,27 -> 0/1.792 cặp gộp nhầm; câu cùng lô gộp 75/224 cặp.
    Tham lam theo thứ tự, O(n²) — đủ cho vài nghìn câu.
    """
    giu: list[int] = []
    tap_giu: list[set[str]] = []
    for i, c in enumerate(cau):
        t = _tap_tu(c)
        if any(len(t & g) / max(1, len(t | g)) >= nguong for g in tap_giu):
            continue
        giu.append(i)
        tap_giu.append(t)
    return giu


def loc_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Lọc dataset đã sinh (dạng row {prompt, completion, meta}).

    Trả về (rows sạch, rows bị loại kèm khoá `ly_do_loai`). Kiểm cả phần prompt (câu
    người dùng) lẫn phần completion (nhãn + câu trả lời của lượt 2).
    """
    from gen.records import query_from_call_prompt

    sach: list[dict[str, Any]] = []
    bo: list[dict[str, Any]] = []
    for r in rows:
        try:
            query = query_from_call_prompt(r["prompt"])
        except ValueError:
            bo.append({**r, "ly_do_loai": "không đọc được câu người dùng"})
            continue
        ok, ly_do = cau_hop_le(query)
        if not ok:
            bo.append({**r, "ly_do_loai": ly_do})
            continue
        if r["meta"]["fn"] not in HAM_MIEN_LAI_TIENG_ANH and lai_tieng_anh(query):
            bo.append({**r, "ly_do_loai": "câu lai ngữ pháp tiếng Anh"})
            continue
        if co_chu_dong_a(r["completion"]):
            bo.append({**r, "ly_do_loai": "nhãn/câu trả lời lẫn chữ Trung/Nhật/Hàn"})
            continue
        ok, ly_do = ket_qua_nhat_quan(r)
        if not ok:
            bo.append({**r, "ly_do_loai": ly_do})
            continue
        ok, ly_do = tra_loi_khong_bia_so(r)
        if not ok:
            bo.append({**r, "ly_do_loai": ly_do})
            continue
        sach.append(r)
    return sach, bo


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Lọc dataset đã sinh (chữ Đông Á, lộ tên hàm)")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--bo-ra", type=Path, default=None, help="file ghi các mẫu bị loại để soi lại")
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    sach, bo = loc_rows(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in sach:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if args.bo_ra:
        with open(args.bo_ra, "w", encoding="utf-8") as f:
            for r in bo:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"vào  : {len(rows)} mẫu")
    print(f"giữ  : {len(sach)} mẫu")
    print(f"loại : {len(bo)} mẫu")
    theo_ly_do: dict[str, int] = {}
    for r in bo:
        theo_ly_do[r["ly_do_loai"]] = theo_ly_do.get(r["ly_do_loai"], 0) + 1
    for ly_do, n in sorted(theo_ly_do.items(), key=lambda x: -x[1]):
        print(f"   {n:>4}  {ly_do}")


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
