"""Sửa TỪ KHOÁ XE bị STT nghe nhầm theo các cặp âm đầu người Việt nhiều vùng hay lẫn (PLAN.md "cách (b)").

Từ khoá = mọi cách gọi bộ phận ≥ 2 âm tiết trong `gen.v6.TU_VUNG` (viết ở vòng v6, TRƯỚC khi có
các bộ ghi âm) — không thêm từ nhặt từ lỗi đã thấy. Hai cụm "khớp" khi giống nhau sau khi bỏ dấu
thanh + dấu chữ (GIỮ "đ") và gộp âm đầu l/n, s/x, ch/tr, d/gi/r — vd "lắp bình sang" -> "nắp bình
xăng". Chỉ sửa cụm ≥ 2 âm tiết: từ đơn ("sang", "cúp") là từ thật, sửa sẽ nhầm. Loại từ khoá trùng
cụm THẬT hay gặp (đo trên câu có dấu của train,
`eval/stt/kiem_sua_nham.py`): "bản đồ" ↔ "bán đồ".

Đo 2026-10-12: không thêm câu đúng cho PhoWhisper-small (+2 cho Zipformer) -> CHƯA dùng trong demo.
"""

from __future__ import annotations

import re
import unicodedata

from gen.v6 import TU_VUNG

# "gi" đứng trước "d" / "r" để gộp cả ba về "d"
_AM_DAU = (("gi", "d"), ("ch", "tr"), ("l", "n"), ("s", "x"), ("r", "d"))
LOAI = {"bản đồ"}  # trùng cụm thật "bán đồ" (chỉ khác dấu thanh) — 116 câu train bị sửa nhầm

# chỗ đặt dấu kiểu cũ -> kiểu mới ("hòa" -> "hoà"): khác chỗ đặt dấu KHÔNG phải nghe nhầm, không sửa
_DAU_CU = {"òa": "oà", "óa": "oá", "ỏa": "oả", "õa": "oã", "ọa": "oạ", "òe": "oè", "óe": "oé", "ỏe": "oẻ",
           "õe": "oẽ", "ọe": "oẹ", "ùy": "uỳ", "úy": "uý", "ủy": "uỷ", "ũy": "uỹ", "ụy": "uỵ"}


def _khong_dau(s: str) -> str:
    """Bỏ dấu thanh + dấu chữ. "đ" không tách dấu khi NFD nên tự được giữ — đ là âm khác hẳn d / gi / r
    (gộp vào d làm "độ" khớp "gió")."""
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def _goc_am_tiet(am: str) -> str:
    am = _khong_dau(am)
    for a, b in _AM_DAU:
        if am.startswith(a) and not (a == "gi" and am == "gi"):
            return b + am[len(a):]
    return am


def _goc(cum: list[str]) -> tuple[str, ...]:
    return tuple(_goc_am_tiet(a) for a in cum)


def _dat_dau(s: str) -> str:
    s = unicodedata.normalize("NFC", s.lower())
    for a, b in _DAU_CU.items():
        s = s.replace(a, b)
    return s


def _bang() -> tuple[dict[tuple[str, ...], str], set[str], int]:
    bang: dict[tuple[str, ...], str] = {}
    dung: set[str] = set()
    for tv in TU_VUNG.values():
        for cum in tv["goi"]:
            am = cum.lower().split()
            if len(am) < 2 or not all(re.fullmatch(r"\w+", a) for a in am) or " ".join(am) in LOAI:
                continue
            dung.add(_dat_dau(" ".join(am)))
            bang.setdefault(_goc(am), unicodedata.normalize("NFC", " ".join(am)))
    return bang, dung, max(len(k) for k in bang)


BANG, CACH_VIET_DUNG, DAI_NHAT = _bang()


def sua(cau: str) -> str:
    """Thay cụm nghe nhầm bằng từ khoá đúng; giữ nguyên dấu câu / khoảng trắng; cụm viết đúng không đụng."""
    manh = re.findall(r"\w+|\W+", unicodedata.normalize("NFC", cau))
    tu = [i for i, m in enumerate(manh) if re.fullmatch(r"\w+", m)]
    ra = list(manh)
    k = 0
    while k < len(tu):
        for n in range(min(DAI_NHAT, len(tu) - k), 1, -1):
            chon = tu[k:k + n]
            if any(re.search(r"[^\s]", manh[j]) for j in range(chon[0] + 1, chon[-1]) if j not in chon):
                continue  # có dấu câu chen giữa -> không coi là một cụm
            cum = [manh[j] for j in chon]
            dung = BANG.get(_goc(cum))
            if dung is None:
                continue
            if _dat_dau(" ".join(cum)) not in CACH_VIET_DUNG:  # nghe nhầm -> thay; viết đúng -> giữ
                thay = dung.split()
                if cum[0][:1].isupper():
                    thay[0] = thay[0][:1].upper() + thay[0][1:]
                for j, t in zip(chon, thay, strict=True):
                    ra[j] = t
            k += n  # cụm đã xét xong (thay hay giữ)
            break
        else:
            k += 1
    return "".join(ra)
