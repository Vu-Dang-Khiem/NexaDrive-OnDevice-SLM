"""Tạo hồ sơ TẬP HIỆU CHỈNH cho ngưỡng có cam kết thống kê (cách 2, người dùng chọn 2026-10-06). Chạy ở máy, $0.

Chỉ câu NGOÀI PHẠM VI (nhãn luôn là từ chối): ngưỡng chỉ cần chọn theo tỉ lệ câu ngoài phạm vi bị làm bừa — ngưỡng càng thấp
model càng tự làm nhiều lệnh, nên chọn ngưỡng THẤP NHẤT mà cận trên (90%) của tỉ lệ làm bừa ≤ 10%. Câu do ChatGPT viết (cùng
kiểu với tập test) — conformal chỉ cam kết khi tập hiệu chỉnh cùng phân phối với lúc dùng thật. KHÔNG dùng tập test.

    python -m eval.tao_hieu_chinh_v6 --ra data/hieu_chinh_v6
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

SEED = 2026_10_07
LOAI = [
    ("ngan_tinh_nang_xe", 50, "lệnh NGẮN 1–5 chữ, ra lệnh thẳng cho một tính năng CÓ trên xe thật nhưng trợ lý KHÔNG làm được"),
    ("tinh_nang_xe", 40, "lệnh nói bình thường (5–15 chữ) cho một tính năng CÓ trên xe thật nhưng trợ lý KHÔNG làm được"),
    ("viec_ngoai_xe", 40, "nhờ một việc KHÔNG liên quan tới xe (nhắn tin, gọi đồ ăn, hỏi kiến thức, lịch hẹn, đồ trong nhà...)"),
    ("tan_gau", 30, "nói chuyện phiếm / cảm thán, không nhờ trợ lý làm gì"),
    ("ke_hoan_canh", 40, "kể một hoàn cảnh trong xe rồi nhờ một việc mà trợ lý KHÔNG làm được"),
]


def tao(ra: Path) -> dict[str, int]:
    dong = [(loai, yc) for loai, n, yc in LOAI for _ in range(n)]
    random.Random(SEED).shuffle(dong)
    ra.mkdir(parents=True, exist_ok=True)
    with open(ra / "1_viet_cau.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "loai", "yeu_cau", "cau", "ghi_chu"])
        w.writeheader()
        for i, (loai, yc) in enumerate(dong, 1):
            w.writerow({"id": i, "loai": loai, "yeu_cau": yc, "cau": "", "ghi_chu": ""})
    return {loai: n for loai, n, _ in LOAI}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ra", type=Path, default=Path("data/hieu_chinh_v6"))
    print(tao(ap.parse_args().ra))


if __name__ == "__main__":
    main()
