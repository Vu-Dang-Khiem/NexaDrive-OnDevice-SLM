"""Đổi SỐ VIẾT BẰNG CHỮ -> CHỮ SỐ cho câu do STT viết ra (PLAN.md Phase 6).

STT (Parakeet, Whisper) viết "hai mươi hai độ", "năm mươi phần trăm", "bảy phẩy ba ki lô mét"; student được
train trên câu viết "22 độ", "50%", "7,3 km". Nhiều chữ số tiếng Việt cũng là từ thường ("năm" = năm tháng,
"ba" = bố, "tư", "một chút") nên luật THẬN TRỌNG:
  • cụm 2 chữ số trở lên hoặc có "mười / mươi / trăm / nghìn / ngàn" -> luôn đổi ("hai mươi hai" -> 22);
  • MỘT chữ số đứng riêng -> chỉ đổi khi ngay sau là đơn vị ("độ", "phần trăm", "km", "phút"...) hoặc ngay
    trước là từ gợi ý ("mức", "số", "nấc", "lên", "xuống", "còn"...): "quạt mức ba" -> "quạt mức 3", nhưng
    "ba tôi", "năm nay", "xuống một chút" giữ nguyên;
  • "<số> phẩy <số>" -> thập phân kiểu Việt "7,3"; "phần trăm" -> "%"; "ki lô mét / cây số" -> "km".
Sửa sau bộ B giọng thật (2026-10-07, PLAN.md Phase 6 "KẾT QUẢ BỘ B"):
  • số NÓI TẮT hai chữ số ("hai hai độ" = 22, "hai lăm" = 25, "ba tư" = 34) — chỉ khi cạnh đơn vị / từ gợi ý
    (từng CỘNG thành 4, 7); hai chữ số rời khác ("hai ba ngày") giữ nguyên chữ;
  • cụm bắt đầu bằng "trăm / nghìn / triệu" không có số đứng trước -> chỉ là số khi ngay sau là đơn vị ("hơn
    trăm cây số" = 100 km), còn lại là tên riêng ("bà Triệu"), không đổi;
  • "không" đứng riêng là từ phủ định / hỏi ("về không") — chỉ đổi sau "mức / số / nấc"; "không" cuối cụm sát
    cuối câu là tiểu từ hỏi ("còn hai mươi không?") -> tách ra, không nhập vào số.
"""

from __future__ import annotations

import re

_DON_VI_SO = {
    "không": 0, "một": 1, "mốt": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4, "năm": 5, "lăm": 5, "nhăm": 5,
    "sáu": 6, "bảy": 7, "bẩy": 7, "tám": 8, "chín": 9,
}
_LINH = {"linh", "lẻ"}
_CHUC = {"mười", "mươi"}
_LON = {"trăm": 100, "nghìn": 1000, "ngàn": 1000, "triệu": 1_000_000}
_TU_SO = set(_DON_VI_SO) | _LINH | _CHUC | set(_LON)
_DAU_HIEU_CUM = _CHUC | set(_LON)

# một chữ số đứng riêng chỉ đổi khi cạnh các từ này
_DON_VI_SAU = {"độ", "phần", "%", "km", "ki", "kilômét", "cây", "phút", "giây", "giờ", "tiếng", "lần", "mét",
               "psi", "bar", "nấc", "mức", "số", "tầng", "người"}
_GOI_Y_TRUOC = {"mức", "số", "nấc", "lên", "xuống", "còn", "về", "thành", "bằng", "tới", "đến", "khoảng",
                "chừng", "tầm", "cỡ", "volume", "kênh", "bài", "quận", "phường"}
_KHONG_LA_SO_SAU = {"mức", "số", "nấc"}  # "không" đứng riêng chỉ là số 0 sau các từ này ("quạt mức không")
# "xuống một chút", "một nửa", "thêm tí" — chữ số đứng riêng trước các từ này là từ thường, không đổi
_KHONG_DOI_SAU = {"chút", "tí", "xíu", "nửa", "ít", "lát", "cái", "khi", "mình", "lúc"}

_TU = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def doc_cum(tu: list[str]) -> int | None:
    """Đọc một cụm chữ số -> giá trị (None nếu không phải cụm hợp lệ). Hỗ trợ tới hàng triệu."""
    tong, nhom, co_so = 0, 0, False  # co_so: nhóm đã có chữ số ("không trăm" = 0, khác "trăm" = 100)
    don_vi = False  # vừa đọc chữ số HÀNG ĐƠN VỊ — thêm một chữ số nữa ngay sau là cụm sai ("hai hai" ≠ 4)
    i = 0
    while i < len(tu):
        t = tu[i]
        if t in _LINH:  # "một trăm linh năm": hàng chục = 0
            pass
        elif t in _DON_VI_SO:
            if don_vi:
                return None
            if i + 1 < len(tu) and tu[i + 1] == "mươi":
                nhom += _DON_VI_SO[t] * 10
                i += 1
            else:
                nhom += _DON_VI_SO[t]
            co_so = True
            don_vi = tu[i] != "mươi"
            i += 1
            continue
        elif t == "mười":
            nhom += 10
            co_so = True
        elif t == "mươi":
            return None  # "mươi" không có số đứng trước
        elif t == "trăm":
            if nhom >= 10:  # "hai mươi trăm" (STT nuốt chữ "phần") không phải số
                return None
            nhom = nhom * 100 if co_so else 100
        elif t in ("nghìn", "ngàn", "triệu"):
            tong += (nhom if co_so else 1) * _LON[t]
            nhom, co_so = 0, False
        else:
            return None
        don_vi = False
        i += 1
    return tong + nhom


def _so_noi_tat(cum: list[str]) -> int | None:
    """Số NÓI TẮT hai chữ số: "hai hai" = 22, "hai lăm" = 25, "ba mốt" = 31 (chữ số đầu 2–9, sau 1–9)."""
    if len(cum) != 2 or not all(t in _DON_VI_SO for t in cum):
        return None
    chuc, dv = _DON_VI_SO[cum[0]], _DON_VI_SO[cum[1]]
    return chuc * 10 + dv if chuc >= 2 and dv >= 1 else None


def _nen_doi(cum: list[str], truoc: str, sau: str) -> bool:
    if len(cum) == 1 and sau in _KHONG_DOI_SAU:
        return False
    if cum == ["không"]:  # phủ định / tiểu từ hỏi ("chạy về không") — chỉ là số 0 sau "mức / số / nấc"
        return truoc in _KHONG_LA_SO_SAU
    if cum[0] in _LON:  # "trăm / nghìn / triệu" không có số trước: là số chỉ khi có đơn vị
        return sau in _DON_VI_SAU  # ("hơn trăm cây số"), còn lại là tên riêng ("bà Triệu")
    return len(cum) >= 2 or bool(set(cum) & _DAU_HIEU_CUM) or sau in _DON_VI_SAU or truoc in _GOI_Y_TRUOC


def _phan_le(le: list[str]) -> str | None:
    """Phần sau "phẩy": đọc từng chữ số ("ba" -> 3, "không năm" -> 05) hoặc cả cụm ("hai mươi" -> 20)."""
    if all(t in _DON_VI_SO for t in le):
        return "".join(str(_DON_VI_SO[t]) for t in le)
    v = doc_cum(le)
    return None if v is None else str(v)


def doi_so(cau: str) -> str:
    """Đổi các cụm số viết bằng chữ trong câu sang chữ số (giữ nguyên phần còn lại)."""
    tok = _TU.findall(cau)
    thap = [t.lower() for t in tok]
    ra: list[str] = []
    i = 0
    while i < len(tok):
        if thap[i] not in _TU_SO:
            ra.append(tok[i])
            i += 1
            continue
        j = i
        while j < len(tok) and thap[j] in _TU_SO:
            j += 1
        # "không" cuối cụm, sát cuối câu / dấu câu = tiểu từ hỏi ("còn hai mươi không?") -> tách khỏi cụm số
        if j - i > 1 and thap[j - 1] == "không" and (j == len(thap) or not re.fullmatch(r"\w+", thap[j])):
            j -= 1
        cum, truoc = thap[i:j], (thap[i - 1] if i > 0 else "")
        sau = thap[j] if j < len(thap) else ""
        gia_tri = None
        if not (cum[0] == "trăm" and truoc == "phần") and _nen_doi(cum, truoc, sau):  # "phần trăm" ≠ 100
            gia_tri = doc_cum(cum)
            if gia_tri is None and (sau in _DON_VI_SAU or truoc in _GOI_Y_TRUOC):  # "hai hai độ" = 22
                gia_tri = _so_noi_tat(cum)
        if gia_tri is None:
            ra.extend(tok[i:j])
            i = j
            continue
        so = str(gia_tri)
        if sau == "phẩy" and j + 1 < len(thap) and thap[j + 1] in _TU_SO:  # thập phân "<số> phẩy <số>"
            k = j + 1
            while k < len(thap) and thap[k] in _TU_SO:
                k += 1
            le = _phan_le(thap[j + 1 : k])
            if le:
                so, j = f"{so},{le}", k
        ra.append(so)
        i = j
    out = _ghep(ra)
    out = re.sub(r"(\d)\s*phần\s+trăm", r"\1%", out, flags=re.IGNORECASE)
    return re.sub(r"(\d)\s*(ki\s*-?\s*lô\s*-?\s*mét|kilômét|cây số)", r"\1 km", out, flags=re.IGNORECASE)


def chuan_cau_stt(cau: str) -> str:
    """Chuẩn hoá câu STT về đúng HÌNH THỨC câu student đã train: đổi số chữ -> chữ số, bỏ dấu câu cuối,
    viết hoa chữ đầu. Lý do (đo trên bộ A, PLAN.md Phase 6): đưa CHỮ GỐC vào student, chỉ đổi chữ đầu thành
    thường -> từ chối nhầm 3 -> 21 câu / 230; thêm "?" cuối -> 12. Câu train / test viết hoa đầu, không dấu
    câu cuối; STT luôn thêm dấu câu, PhoWhisper còn viết thường."""
    c = re.sub(r"[\s.?!…,;:]+$", "", doi_so(cau).strip())
    return c[:1].upper() + c[1:]


def _ghep(tok: list[str]) -> str:
    """Ghép token lại thành câu: không chèn dấu cách trước dấu câu."""
    s = ""
    for t in tok:
        if s and not re.fullmatch(r"[^\w\s]", t):
            s += " "
        s += t
    return s
