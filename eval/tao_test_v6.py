"""Tạo hồ sơ TẬP TEST v6 cho người dùng làm với ChatGPT (PLAN.md Phase 7 — v6). Chạy ở máy, miễn phí.

Cùng quy trình mo_ho_4: ChatGPT chỉ thấy MÔ TẢ chức năng bằng tiếng Việt (không thấy tên hàm, không thấy
câu mẫu nào của dữ liệu train); khoá đáp án (`khoa_id.csv`) giữ ở máy. Test làm và ĐÓNG BĂNG TRƯỚC khi sinh
dữ liệu train v6 — tránh việc tôi nhìn dữ liệu train rồi mới đặt đề.

Bốn phần (đo đúng các lỗi người dùng gặp khi thử demo 2026-10-06):
  A ngan        lệnh thẳng NGẮN 1–4 chữ, cả 24 hàm ("bật pha", "mở kiếng")
  B cach_goi    lệnh thẳng bình thường nhưng gọi bộ phận / giá trị theo cách khác thường (gọi tắt, địa phương)
  C moi         tính năng schema v6: chỉnh nhiệt tương đối, sưởi / làm lạnh / tự động, cả hai kính trước / sau
  D hai_luot    hai lượt: lượt 2 nối tiếp lượt 1 (đổi chiều, làm thêm, sang bộ phận khác), lệnh mới,
                ngoài phạm vi

    python -m eval.tao_test_v6 --ra data/test_v6
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from schema.loader import CATALOG

SEED = 2026_10_06

# mô tả chức năng cho người / ChatGPT viết câu — KHÔNG lộ tên hàm, KHÔNG gợi ý cách nói tắt
MO_TA = {
    "get_fuel_level": "hỏi còn bao nhiêu xăng",
    "get_battery_range": "hỏi pin còn bao nhiêu / chạy điện được bao xa",
    "get_tire_pressure": "hỏi áp suất lốp",
    "get_vehicle_status": "hỏi tình trạng chung của xe",
    "get_odometer": "hỏi xe đã đi tổng cộng bao nhiêu km",
    "get_range_estimate": "hỏi xe còn đi được bao xa / có đủ cho một quãng đường không",
    "set_ac": "bật / tắt / chỉnh điều hoà",
    "set_fan_speed": "chỉnh quạt gió điều hoà mạnh / nhẹ / tắt",
    "set_defrost": "bật / tắt sấy kính (khử sương kính)",
    "lock_doors": "khoá cửa xe",
    "unlock_doors": "mở khoá cửa xe",
    "open_trunk": "mở cốp sau",
    "close_trunk": "đóng cốp sau",
    "open_charge_port": "mở nắp cổng sạc",
    "open_fuel_flap": "mở nắp bình xăng",
    "control_window": "hạ / nâng kính cửa sổ",
    "operate_sunroof": "mở / đóng / hé cửa sổ trời",
    "set_headlights": "bật / tắt / đổi chế độ đèn pha (chiếu xa, chiếu gần, tự động)",
    "set_hazard_lights": "bật / tắt đèn khẩn cấp (xe mình gặp sự cố)",
    "set_volume": "chỉnh âm lượng to / nhỏ",
    "play_media": "mở nhạc / đài",
    "navigate_to": "dẫn đường tới một nơi có tên",
    "find_nearby": "tìm quanh đây: cây xăng, trạm sạc, bãi đỗ, nhà hàng, bệnh viện, ATM, quán cà phê",
    "get_weather": "hỏi thời tiết",
}

# Phần C — tính năng mới của schema v6
PHAN_C = [
    *[("set_ac", "giảm hoặc tăng nhiệt độ điều hoà MỘT KHOẢNG có nói số độ (vd thêm / bớt mấy độ)")] * 6,
    *[("set_ac", "giảm hoặc tăng nhiệt độ điều hoà, KHÔNG nói số (kiểu cho mát / ấm hơn chút)")] * 6,
    *[("set_ac", "đặt điều hoà VỀ một nhiệt độ cụ thể (có số độ)")] * 3,
    *[("set_ac", "bật sưởi ấm (có thể kèm số độ)")] * 4,
    *[("set_ac", "bật chế độ làm lạnh (có thể kèm số độ)")] * 3,
    *[("set_ac", "chuyển điều hoà sang chế độ tự động")] * 1,
    *[("control_window", "hạ hoặc nâng CẢ HAI kính phía sau")] * 4,
    *[("control_window", "hạ hoặc nâng CẢ HAI kính phía trước")] * 3,
]

# Phần D — lượt 1 là các chức năng điều khiển (có trạng thái để nối tiếp)
LUOT_1 = [
    "set_ac", "set_fan_speed", "set_defrost", "control_window", "operate_sunroof", "open_trunk",
    "lock_doors", "set_headlights", "set_hazard_lights", "set_volume", "play_media", "close_trunk",
]
_DOI_CHIEU = ("lượt 2 bảo làm NGƯỢC LẠI việc vừa làm, nói trống không, không nhắc lại tên bộ phận "
              "(kiểu bảo tắt / đóng / mở lại đi)")
_THEM_NUA = "lượt 2 bảo làm THÊM / MẠNH HƠN / NHIỀU HƠN theo đúng hướng vừa làm, không nhắc tên bộ phận"
_BO_PHAN_KHAC = ("lượt 2 bảo làm việc tương tự cho một vị trí / bộ phận KHÁC cùng loại, nói tắt dựa vào "
                 "lượt 1 (kiểu: cả bên kia nữa, phía sau thì sao)")
# (kiểu, yêu cầu, số câu, các chức năng lượt 1 HỢP LỆ cho kiểu đó) — "làm thêm" sau "mở cốp" là vô lý
KIEU_D = [
    ("doi_chieu", _DOI_CHIEU, 15, [h for h in LUOT_1 if h != "play_media"]),  # không có hàm tắt nhạc
    ("them_nua", _THEM_NUA, 10,
     ["set_ac", "set_fan_speed", "set_volume", "control_window", "operate_sunroof"]),
    ("bo_phan_khac", _BO_PHAN_KHAC, 15, ["set_ac", "set_defrost", "control_window"]),
    ("lenh_moi", "lượt 2 là một lệnh MỚI, nói đầy đủ, không liên quan lượt 1", 10, LUOT_1),
    ("ngoai_pham_vi", "lượt 2 nhờ một việc xe KHÔNG làm được hoặc chỉ tán gẫu", 10, LUOT_1),
]
_YC_A = "lệnh nói THẲNG, NGẮN 1–4 chữ, kiểu buột miệng khi đang lái"
_YC_B = ("lệnh nói thẳng 4–12 chữ, gọi bộ phận / giá trị theo cách KHÁC cách gọi chuẩn (gọi tắt, từ địa "
         "phương Bắc / Trung / Nam, khẩu ngữ)")


def tao(ra: Path) -> dict[str, int]:
    rng = random.Random(SEED)
    ham = [f.name for f in CATALOG.functions if f.name != "irrelevant_function"]
    assert set(ham) == set(MO_TA), "MO_TA phải phủ đúng các hàm trong schema"
    dong: list[dict[str, str]] = []
    khoa: list[dict[str, str]] = []

    def them(phan: str, ham_dich: str, kieu: str, chuc_nang: str, yeu_cau: str) -> None:
        i = str(len(dong) + 1)
        dong.append({"id": i, "phan": phan, "chuc_nang": chuc_nang, "yeu_cau": yeu_cau, "luot_1": "",
                     "cau": "", "ghi_chu": ""})
        khoa.append({"id": i, "phan": phan, "ham_dich": ham_dich, "kieu": kieu})

    for _ in range(3):
        for h in rng.sample(ham, len(ham)):
            them("A", h, "ngan", MO_TA[h], _YC_A)
    for _ in range(2):
        for h in rng.sample(ham, len(ham)):
            them("B", h, "cach_goi", MO_TA[h], _YC_B)
    for h, yc in rng.sample(PHAN_C, len(PHAN_C)):
        them("C", h, "moi", MO_TA[h], yc)
    hai_luot = [(kieu, yc, ds[k % len(ds)]) for kieu, yc, n, ds in KIEU_D for k in range(n)]
    for kieu, yc, h1 in rng.sample(hai_luot, len(hai_luot)):
        them("D", h1, kieu, f"lượt 1: {MO_TA[h1]}", yc)

    ra.mkdir(parents=True, exist_ok=True)
    for ten, rows, cot in (
        ("1_viet_cau.csv", dong, ["id", "phan", "chuc_nang", "yeu_cau", "luot_1", "cau", "ghi_chu"]),
        ("khoa_id.csv", khoa, ["id", "phan", "ham_dich", "kieu"]),
    ):
        with open(ra / ten, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cot)
            w.writeheader()
            w.writerows(rows)
    dem: dict[str, int] = {}
    for d in dong:
        dem[d["phan"]] = dem.get(d["phan"], 0) + 1
    return dem


def main() -> None:
    ap = argparse.ArgumentParser(description="Tạo hồ sơ tập test v6")
    ap.add_argument("--ra", type=Path, default=Path("data/test_v6"))
    print(tao(ap.parse_args().ra))


if __name__ == "__main__":
    main()
