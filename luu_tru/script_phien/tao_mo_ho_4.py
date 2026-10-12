"""Tạo file trống cho bộ mơ hồ thứ 4 (130 câu = 13 hàm × 10), xáo bằng seed 2032 (khác mo_ho_3: 2031)."""
import csv
import random
import shutil
from pathlib import Path

HAM = {
    "set_ac": "bật / tắt điều hoà, chỉnh nhiệt độ",
    "set_fan_speed": "chỉnh quạt gió mạnh lên / yếu đi (KHÔNG phải bật điều hoà)",
    "set_defrost": "sấy kính trước hoặc kính sau cho hết mờ",
    "lock_doors": "khoá cửa xe",
    "close_trunk": "đóng cốp sau",
    "control_window": "hạ / nâng kính cửa sổ",
    "operate_sunroof": "mở / đóng / hé cửa sổ trời",
    "set_headlights": "bật đèn pha / đổi chế độ đèn pha",
    "set_hazard_lights": "bật / tắt đèn khẩn cấp (nháy cả 4 góc, khi CHÍNH xe mình gặp sự cố hoặc phải dừng gấp)",
    "set_volume": "chỉnh âm lượng to lên / nhỏ đi",
    "play_media": "mở nhạc / đài",
    "navigate_to": "dẫn đường tới một nơi cụ thể",
    "find_nearby": "tìm quanh đây: cây xăng, trạm sạc, bãi đỗ, nhà hàng, bệnh viện, cây ATM, quán cà phê",
}
rows = [(ham, mo_ta) for ham, mo_ta in HAM.items() for _ in range(10)]
random.Random(2032).shuffle(rows)
ra = Path("data/mo_ho_4")
ra.mkdir(parents=True, exist_ok=True)
with open(ra / "1_viet_cau.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "chuc_nang", "cau", "ghi_chu"])
    for i, (_, mo_ta) in enumerate(rows, 1):
        w.writerow([i, mo_ta, "", ""])
with open(ra / "khoa_id.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "ham_dich"])
    for i, (ham, _) in enumerate(rows, 1):
        w.writerow([i, ham])
shutil.copy("data/mo_ho_3/BANG_TRA_HAM.md", ra / "BANG_TRA_HAM.md")
shutil.copy("data/mo_ho_3/HUONG_DAN_gan_nhan.md", ra / "HUONG_DAN_gan_nhan.md")
print("xong", len(rows))
