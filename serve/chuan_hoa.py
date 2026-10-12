"""Chuẩn hoá HÌNH THỨC câu gõ tay trước khi đưa vào model (PLAN.md Phase 7, "chữ hoa / thường").

Vì sao cần (đo 2026-10-07 trên v6.1 s42 q4): tokenizer coi "Hỏng" và "hỏng" là hai token khác nhau, dữ
liệu train (do teacher sinh) LUÔN viết hoa chữ đầu -> "hỏng xe giữa đường rồi" bị từ chối (68%), "Hỏng xe
giữa đường rồi" hỏi bật đèn cảnh báo (99%). Bước 1 (người dùng chọn): chỉ VIẾT HOA CHỮ CÁI ĐẦU — đúng hình
thức câu train; phần còn lại giữ nguyên như người dùng gõ (tên riêng không bị đổi). Đường giọng nói đã làm
việc này trong `so_chu.chuan_cau_stt`.

Thêm (nhóm A, 2026-10-07): gõ lặp chữ "mở cốpp" -> model đề xuất nắp sạc. Tiếng Việt không có từ nào tận
cùng bằng phụ âm đôi -> gộp phụ âm cuối lặp ("cốpp" -> "cốp"); chữ cái lặp từ 3 lần ("alooo") -> một.
Chữ số không đụng ("1000" giữ nguyên).
"""

from __future__ import annotations

import re

_PHU_AM_CUOI_LAP = re.compile(r"([bcdđghklmnpqrstvx])\1+\b", re.IGNORECASE)
_CHU_LAP_3 = re.compile(r"([^\W\d_])\1{2,}")
_CHU_CAI = re.compile(r"[^\W\d_]")


def la_toan_in_hoa(c: str) -> bool:
    """Câu gõ (gần như) TOÀN chữ in hoa: ≥ 3 chữ cái và ≥ 80% là chữ hoa ("NÓNG QUÁ", "MỞ CỐP ĐI")."""
    chu = _CHU_CAI.findall(c)
    return len(chu) >= 3 and sum(x.isupper() for x in chu) >= 0.8 * len(chu)


def chuan_hoa_chu(cau: str) -> str:
    """Gộp khoảng trắng thừa, gộp chữ gõ lặp, câu toàn in hoa -> chữ thường, viết hoa chữ cái đầu; câu
    viết bình thường giữ nguyên phần sau chữ đầu (tên riêng không đổi). Đo 2026-10-07: "NÓNG QUÁ" -> model
    đề xuất chỉnh âm lượng (đúng: điều hoà) — câu toàn in hoa không mang tên riêng nên đổi về thường."""
    c = " ".join(cau.split())
    c = _CHU_LAP_3.sub(r"\1", _PHU_AM_CUOI_LAP.sub(r"\1", c))
    if la_toan_in_hoa(c):
        c = c.lower()
    return c[:1].upper() + c[1:]
