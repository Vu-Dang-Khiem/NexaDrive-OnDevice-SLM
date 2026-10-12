"""Tách câu NHIỀU LỆNH nối bằng từ nối rõ ràng (PLAN.md — "Câu nhiều lệnh", người dùng chọn hướng A
2026-10-08).

Model v6.1 chưa từng học gọi nhiều hàm một lượt (0 mẫu làm luôn nhiều việc trong dữ liệu train) -> câu "bật
đèn pha xong kiểm tra xăng xong đó mở cốp" chỉ làm một việc. Tầng chạy thật tách câu theo từ nối rồi gọi
model LẦN LƯỢT từng đoạn (đúng thứ tự người lái nói); mỗi đoạn vẫn qua ngưỡng tự tin + luật an toàn như câu
đơn.

Chỉ tách ở từ nối CHẮC CHẮN: "xong (đó / rồi / thì)", "sau đó", "tiếp theo", "tiếp đó", "với cả", ", rồi" /
"rồi thì". KHÔNG tách ở "và" ("bật điều hoà và quạt" là một ý) hay "rồi" đứng một mình ("hết xăng rồi").
Mỗi đoạn phải có ≥ 2 chữ ("sạc xong chưa" không bị tách).
"""

from __future__ import annotations

import re

_TU_NOI = "xong đó|xong rồi thì|xong rồi|xong thì|xong|sau đó|tiếp theo|tiếp đó|với cả|rồi thì"
_NOI = re.compile(rf"\s*[,;]?\s*\b(?:{_TU_NOI})\b\s*,?\s*|\s*[,;]\s*rồi\s+", re.IGNORECASE)
_IT_NHAT = 2  # số chữ tối thiểu mỗi đoạn


def tach_lenh(cau: str) -> list[str]:
    """Danh sách đoạn lệnh theo thứ tự; câu không có từ nối (hoặc tách ra đoạn quá ngắn) -> [câu]."""
    doan = [d.strip(" ,.;") for d in _NOI.split(cau)]
    doan = [d for d in doan if d]
    if len(doan) < 2 or any(len(d.split()) < _IT_NHAT for d in doan):
        return [cau]
    return doan
