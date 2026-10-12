"""Tạo hồ sơ TẬP ĐÁNH GIÁ GRPO cho người dùng làm với ChatGPT (PLAN.md Phase 7 — GRPO). Chạy ở máy, $0.

Vì sao cần tập MỚI: tập test v6 / test sạch đã dùng để so v5 · v6 · v6.1 nhiều lần; so thêm GRPO trên đó là
chọn đi chọn lại trên cùng bộ đề. Cùng quy trình test v6 / mo_ho_4: ChatGPT chỉ thấy MÔ TẢ chức năng (không
tên hàm, không câu mẫu train), người dùng duyệt nhãn, Claude kiểm trùng với mọi dữ liệu train rồi ĐÓNG BĂNG
TRƯỚC khi chạy GRPO. Khoá đáp án (`khoa_id.csv`) giữ ở máy.

Bốn phần, nhắm đúng chỗ GRPO cần chứng minh + phần "không được tụt":
  R ro_rang      48  lệnh thẳng, cả 24 chức năng × 2 (kiểm câu rõ ràng không tụt)
  M mo_ho        64  than phiền / nói bóng gió / kể hoàn cảnh -> trợ lý nên HỎI LẠI (điểm yếu chính)
  N ngoai        48  tán gẫu, việc ngoài xe, tính năng xe không hỗ trợ, hoàn cảnh + nhờ việc không làm được
  H hai_luot     40  lượt 2 nối tiếp lượt 1 · lượt 2 là câu bóng gió KHÔNG liên quan lượt 1 · lượt 2 ngoài
                     phạm vi (đo đúng lỗi "ngữ cảnh làm lật câu" người dùng gặp 2026-10-07)
Bản "chữ đầu viết thường" do code sinh từ chính tập đã đóng băng (không phải viết thêm).

    python -m eval.tao_test_grpo --ra data/test_grpo
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from eval.tao_test_v6 import MO_TA
from schema.loader import CATALOG

SEED = 2026_10_08

# Phần M: chỉ các chức năng schema cho phép HỎI LẠI (có implicit_examples; hàm nhạy cảm như mở cốp / mở khoá
# không bao giờ được đề xuất từ câu than thở)
HAM_HOI_LAI = [f.name for f in CATALOG.functions if f.implicit_examples]
# mô tả KIỂU câu, không đưa câu ví dụ cụ thể (ChatGPT dễ chép lại; nhất là câu người dùng từng báo lỗi)
_YC_M = [
    ("than_phien", 24, "than phiền một cảm giác / vấn đề trong xe, NÓI RÕ cảm giác nhưng KHÔNG ra lệnh"),
    ("bong_gio", 24, "nói BÓNG GIÓ, cảm thán nhẹ một câu, gợi ý mơ hồ — người nghe tinh ý mới hiểu là muốn "
                     "dùng chức năng này"),
    ("hoan_canh", 16, "KỂ một hoàn cảnh dài 12–25 chữ, không ra lệnh, mà ai nghe cũng thấy nên dùng chức "
                      "năng này"),
]
_YC_N = [
    ("tan_gau", 12, "tán gẫu / cảm thán về chuyện KHÔNG liên quan tới xe hay việc trong xe, không nhờ gì"),
    ("viec_ngoai_xe", 12, "nhờ một việc KHÔNG liên quan tới xe (nhắn tin, đặt đồ ăn, hỏi kiến thức, đồ "
                          "trong nhà...)"),
    ("tinh_nang_khong_co", 12, "ra lệnh một tính năng CÓ trên xe thật nhưng trợ lý KHÔNG làm được (bộ phận / "
                               "tính năng ngoài danh sách chức năng của trợ lý)"),
    ("hoan_canh_nho", 12, "kể hoàn cảnh trong xe rồi nhờ một việc trợ lý KHÔNG làm được"),
]
# Phần H — lượt 1 là lệnh thẳng (có trạng thái để nối tiếp)
LUOT_1 = [
    "set_ac", "set_fan_speed", "set_defrost", "control_window", "operate_sunroof", "open_trunk",
    "lock_doors", "set_headlights", "set_hazard_lights", "set_volume", "play_media", "close_trunk",
]
_YC_H = [
    ("noi_tiep", 20, "lượt 2 nói TIẾP lượt 1, nói tắt dựa vào lượt 1, không nhắc tên bộ phận (làm ngược "
                     "lại / làm thêm / làm cho vị trí khác)"),
    ("bong_gio_khac", 12, "lượt 2 là câu than phiền / nói BÓNG GIÓ về một việc KHÁC, KHÔNG liên quan lượt 1 "
                          "(trợ lý nên hỏi lại về việc khác đó)"),
    ("ngoai_pham_vi", 8, "lượt 2 tán gẫu hoặc nhờ việc xe KHÔNG làm được, không liên quan lượt 1"),
]
_YC_R = "lệnh nói THẲNG, 2–12 chữ, như người thật nói trong xe (có thể cộc, nói tắt, có từ đệm)"


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

    for _ in range(2):
        for h in rng.sample(ham, len(ham)):
            them("R", h, "ro_rang", MO_TA[h], _YC_R)
    m = [(k, yc, HAM_HOI_LAI[j % len(HAM_HOI_LAI)]) for k, n, yc in _YC_M for j in range(n)]
    for k, yc, h in rng.sample(m, len(m)):
        them("M", h, k, MO_TA[h], yc)
    for k, n, yc in _YC_N:
        for _ in range(n):
            them("N", "irrelevant_function", k, "(không thuộc chức năng nào của trợ lý)", yc)
    hl = [(k, yc, LUOT_1[j % len(LUOT_1)]) for k, n, yc in _YC_H for j in range(n)]
    for k, yc, h1 in rng.sample(hl, len(hl)):
        them("H", h1, k, f"lượt 1: {MO_TA[h1]}", yc)

    # xáo thứ tự dòng (giữ id theo thứ tự mới) để ChatGPT không viết theo cụm cùng kiểu
    cap = list(zip(dong, khoa, strict=True))
    rng.shuffle(cap)
    for i, (d, k) in enumerate(cap, 1):
        d["id"] = k["id"] = str(i)
    ra.mkdir(parents=True, exist_ok=True)
    cot_viet = ["id", "phan", "chuc_nang", "yeu_cau", "luot_1", "cau", "ghi_chu"]
    for ten, rows, cot in (
        ("1_viet_cau.csv", [d for d, _ in cap], cot_viet),
        ("khoa_id.csv", [k for _, k in cap], ["id", "phan", "ham_dich", "kieu"]),
    ):
        with open(ra / ten, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cot)
            w.writeheader()
            w.writerows(rows)
    dem: dict[str, int] = {}
    for d, _ in cap:
        dem[d["phan"]] = dem.get(d["phan"], 0) + 1
    return dem


def main() -> None:
    ap = argparse.ArgumentParser(description="Tạo hồ sơ tập đánh giá GRPO")
    ap.add_argument("--ra", type=Path, default=Path("data/test_grpo"))
    print(tao(ap.parse_args().ra))


if __name__ == "__main__":
    main()
