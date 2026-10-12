"""Rà dữ liệu sinh ra để tìm chỗ model SẼ học nhầm — chạy TRƯỚC khi train, miễn phí.

VÌ SAO CÓ FILE NÀY. Vòng train đầu tiên ra kết quả rất chênh lệch giữa hai tập:

    tập val (teacher sinh)   : rõ ràng 97,8%  |  ngoài phạm vi 100%
    tập test (người viết tay): rõ ràng 71,2%  |  ngoài phạm vi 62,3%

Val không phát hiện được gì, vì val do CÙNG teacher sinh bằng CÙNG bộ prompt — chỗ mù
của dữ liệu train cũng là chỗ mù của dữ liệu val, nó không tự soi ra lỗi của chính nó.
Mổ xẻ các câu sai thì cả ba nguyên nhân đều nhìn thấy được **chỉ bằng thống kê trên
dữ liệu train**, không cần đụng tới tập test. File này tự động hoá đúng ba phép soi đó,
để lần sau không phải trả tiền train rồi mới biết.

(Quan trọng về phương pháp: nếu sửa dữ liệu dựa trên các câu test đã chấm thì tập test
hết còn là "dữ liệu chưa từng thấy", và điểm vòng sau đẹp lên một phần chỉ vì đã vá
đúng chỗ nó hỏng. Nên mọi phép rà ở đây chỉ đọc file train.)

BỐN PHÉP RÀ:

  1. HÀM BỊ KHOÁ TỪ VỰNG — một từ phủ gần 100% số câu của hàm, không có từ đồng nghĩa.
     Ca thật: `control_window` có 267/267 câu dùng "cửa sổ", 0 câu dùng "kính", trong
     khi `set_defrost` dùng "kính" ở 100% số câu. Model học được luật sạch không một
     phản ví dụ: "kính" ⟹ sấy kính. Người thật nói "hạ kính bên tài" -> gọi nhầm hàm.
     Gốc rễ nằm trong schema: `examples` của control_window cũng chỉ viết "cửa sổ",
     teacher diễn giải từ ví dụ mồi nên cả 267 câu thừa hưởng đúng một cách gọi.

  2. DẠNG CÂU LỆCH GIỮA DƯƠNG VÀ ÂM — nếu câu mệnh lệnh hầu như chỉ xuất hiện ở mẫu
     gọi hàm, model học tắt "mệnh lệnh ⟹ gọi hàm" thay vì học ranh giới ngữ nghĩa.
     Ca thật: mẫu từ chối chỉ 2,1% là mệnh lệnh, mẫu gọi hàm 41,9%. Đến khi gặp
     "Bật sưởi ghế lái lên" (xe không có tính năng đó) là model gọi bừa `set_ac`.

  3. MẪU HỎI LẠI LỆCH HÀM — `<nexa_ask>` đề xuất hàm nào, lệch bao nhiêu.
     Ca thật: 298/803 mẫu ask đề xuất `set_ac`, chỉ 34 đề xuất `control_window`.
     Kết quả: hễ người dùng than phiền gì model cũng đề nghị bật điều hoà.

  4. ĐÓNG KHUÔN CÂU — bao nhiêu phần trăm mẫu dùng chung vài chữ mở đầu.
     Ca thật: 26,5% mẫu từ chối mở đầu bằng "bạn có thể giúp...".

Máy chỉ nêu ứng viên, NGƯỜI quyết. Với mỗi hàm bị khoá, câu phải tự hỏi là "tiếng Việt
còn cách nào khác gọi thứ này không?" — máy không trả lời hộ được.

    python -m gen.ra_soat_tu_vung --input data/processed/final/train_vi.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from gen.records import query_from_call_prompt
from schema.loader import CATALOG

NEGATIVE_FN = "irrelevant_function"

# Từ chức năng + từ lịch sự: có mặt khắp nơi nên không phân biệt được hàm nào với hàm nào.
TU_BO = {
    "là", "của", "cho", "với", "và", "thì", "mà", "ở", "trong", "ngoài", "ra", "vào",
    "lên", "xuống", "đi", "rồi", "nhé", "nhá", "ạ", "à", "ơi", "này", "kia", "đó",
    "tôi", "mình", "bạn", "em", "anh", "chị", "giúp", "hộ", "giùm", "được", "không",
    "có", "làm", "ơn", "xin", "vui", "lòng", "hãy", "thử", "xem", "cái", "một", "các",
    "những", "cả", "hết", "luôn", "ngay", "đang", "đã", "sẽ", "bị", "quá", "lắm",
    "thế", "sao", "nào", "gì", "bao", "nhiêu", "chút", "xíu", "tí", "hơi", "khá",
    "nữa", "còn", "chưa", "đừng", "nên", "cần", "muốn", "chỉ", "tầm", "khoảng",
    "tới", "đến", "từ", "về", "cứ", "tự", "nhiên", "ok", "oke", "ê", "đây", "xe",
}

# Động từ điều khiển — câu mở đầu bằng một trong số này là câu ra lệnh.
DONG_TU_LENH = {
    "bật", "mở", "tắt", "đóng", "hạ", "kéo", "nâng", "gập", "chuyển", "khóa", "khoá",
    "vặn", "tăng", "giảm", "chỉnh", "đặt", "bấm", "kích", "cài", "huỷ", "hủy", "dừng",
}

_DAU = re.compile(r"[^\wÀ-ỹ\s]", re.UNICODE)


def tach_tu(cau: str) -> list[str]:
    return [t for t in _DAU.sub(" ", cau.lower()).split() if t]


def cum_tu(cau: str, toi_da: int = 3) -> set[str]:
    """Tiếng Việt ghép nhiều âm tiết ("cửa sổ", "khoang hành lý") nên phải lấy cả cụm."""
    tu = tach_tu(cau)
    ra: set[str] = set()
    for n in range(1, toi_da + 1):
        for i in range(len(tu) - n + 1):
            cum = tu[i : i + n]
            if all(t in TU_BO for t in cum):
                continue
            ra.add(" ".join(cum))
    return ra


def la_menh_lenh(cau: str) -> bool:
    tu = tach_tu(cau)
    return bool(tu) and tu[0] in DONG_TU_LENH


def gom_cau(rows: list[dict[str, Any]], loai: tuple[str, ...] = ("call",)) -> dict[str, list[str]]:
    """Gom câu người dùng theo hàm.

    Mặc định CHỈ lấy `call`: đó là câu ra lệnh trực tiếp, nơi tên bộ phận được nói rõ.
    Trộn thêm `ask` sẽ làm loãng tỉ lệ vì câu ngụ ý ("Mùi xe khó chịu quá") vốn không
    chứa tên bộ phận — chính chỗ này đã làm phép rà bỏ sót `control_window` lần đầu.
    """
    theo_ham: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        if r["meta"]["type"] not in loai:
            continue
        try:
            theo_ham[r["meta"]["fn"]].append(query_from_call_prompt(r["prompt"]))
        except ValueError:
            continue
    return theo_ham


# ── 1. hàm bị khoá từ vựng ──────────────────────────────────────────────────
def ham_bi_khoa(
    theo_ham: dict[str, list[str]], nguong_phu: float = 0.95, toi_thieu_cau: int = 20
) -> list[dict[str, Any]]:
    """Hàm mà một từ phủ gần hết số câu của nó -> mù với mọi cách gọi khác."""
    dem_theo_ham = {
        fn: Counter(t for c in cau for t in cum_tu(c)) for fn, cau in theo_ham.items()
    }
    ket_qua = []
    for fn, cau in sorted(theo_ham.items()):
        if len(cau) < toi_thieu_cau or fn == NEGATIVE_FN:
            continue
        phu = [(t, n / len(cau)) for t, n in dem_theo_ham[fn].items() if n / len(cau) >= nguong_phu]
        if not phu:
            continue
        phu.sort(key=lambda x: (-x[1], -len(x[0])))
        goc = phu[0][0]
        # Hàm nào KHÁC cũng hay dùng từ này -> đó là hàm dễ bị model nhầm sang
        khac = [(g, d[goc]) for g, d in dem_theo_ham.items() if g != fn and d[goc]]
        khac.sort(key=lambda x: -x[1])
        ket_qua.append(
            {
                "ham": fn,
                "so_cau": len(cau),
                "tu_khoa": goc,
                "do_phu": round(phu[0][1] * 100, 1),
                "ham_khac_cung_dung": khac[:3],
                "tu_phu_cao_khac": [t for t, _ in phu[1:5]],
            }
        )
    return ket_qua


# ── 2. dạng câu lệch giữa mẫu dương và mẫu âm ───────────────────────────────
def lech_dang_cau(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mệnh lệnh chỉ có ở mẫu dương -> model học tắt "mệnh lệnh ⟹ gọi hàm"."""
    duong, am = [], []
    for r in rows:
        if r["meta"]["type"] not in ("call", "ask"):
            continue
        try:
            q = query_from_call_prompt(r["prompt"])
        except ValueError:
            continue
        (am if r["meta"]["fn"] == NEGATIVE_FN else duong).append(q)

    ml_duong = sum(map(la_menh_lenh, duong))
    ml_am = sum(map(la_menh_lenh, am))
    tong_ml = ml_duong + ml_am
    return {
        "so_mau_duong": len(duong),
        "so_mau_am": len(am),
        "menh_lenh_duong_%": round(ml_duong / len(duong) * 100, 1) if duong else 0.0,
        "menh_lenh_am_%": round(ml_am / len(am) * 100, 1) if am else 0.0,
        # Model đoán "mệnh lệnh -> gọi hàm" thì đúng bao nhiêu phần trăm trên tập train?
        "goi_ham_dung_neu_doan_theo_menh_lenh_%": round(ml_duong / tong_ml * 100, 1) if tong_ml else 0.0,
    }


# ── 3. mẫu hỏi lại lệch hàm ─────────────────────────────────────────────────
def lech_ham_hoi_lai(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """`<nexa_ask>` đề xuất hàm nào — lệch thì model đề nghị mãi một thứ."""
    ten = {f.index: f.name for f in CATALOG.functions}
    dem: Counter = Counter()
    so_mau = 0
    for r in rows:
        if r["meta"]["type"] != "ask":
            continue
        so_mau += 1
        for m in re.findall(r"<nexa_(\d+)>", r["completion"]):
            dem[ten.get(int(m), f"?{m}")] += 1
    tong = sum(dem.values()) or 1
    ra = [
        {"ham": fn, "so_lan": n, "ty_le_%": round(n / tong * 100, 1)}
        for fn, n in dem.most_common()
    ]
    # hàm KHÔNG BAO GIỜ được đề xuất cũng là thông tin
    for f in CATALOG.functions:
        if f.name not in dem and f.name != NEGATIVE_FN:
            ra.append({"ham": f.name, "so_lan": 0, "ty_le_%": 0.0})
    return ra


# ── 4. đóng khuôn câu ───────────────────────────────────────────────────────
def dong_khuon(theo_ham: dict[str, list[str]], so_tu_dau: int = 4) -> list[dict[str, Any]]:
    """Bao nhiêu % câu của một hàm dùng chung đúng mấy chữ mở đầu."""
    ra = []
    for fn, cau in theo_ham.items():
        if len(cau) < 20:
            continue
        dem = Counter(" ".join(tach_tu(c)[:so_tu_dau]) for c in cau)
        khuon, n = dem.most_common(1)[0]
        ra.append(
            {
                "ham": fn,
                "so_cau": len(cau),
                "khuon_hay_nhat": khuon,
                "ty_le_%": round(n / len(cau) * 100, 1),
                "so_cau_khac_nhau": len(set(cau)),
            }
        )
    ra.sort(key=lambda x: -x["ty_le_%"])
    return ra


def main() -> None:
    ap = argparse.ArgumentParser(description="Rà dữ liệu train tìm chỗ model sẽ học nhầm")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--nguong-phu", type=float, default=0.95)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    theo_ham_call = gom_cau(rows, ("call",))
    theo_ham_all = gom_cau(rows, ("call", "ask"))
    khoa = ham_bi_khoa(theo_ham_call, args.nguong_phu)
    dang = lech_dang_cau(rows)
    hoi = lech_ham_hoi_lai(rows)
    khuon = dong_khuon(theo_ham_all)

    print(f"Đã đọc {len(rows)} mẫu · {len(theo_ham_call)}/{len(CATALOG.functions)} hàm có câu `call`\n")

    print(f"=== 1. HÀM BỊ KHOÁ TỪ VỰNG (một từ phủ ≥{args.nguong_phu * 100:.0f}% số câu) ===")
    print(f"   {'hàm':22}{'câu':>6}{'từ khoá':>16}{'phủ':>7}   hàm khác cũng dùng từ này")
    for k in khoa:
        khac = ", ".join(f"{g}({n})" for g, n in k["ham_khac_cung_dung"]) or "— KHÔNG AI —"
        print(f"   {k['ham']:22}{k['so_cau']:>6}{k['tu_khoa']:>16}{k['do_phu']:>6}%   {khac}")
    print(f"   -> {len(khoa)}/{len(theo_ham_call) - 1} hàm chỉ có MỘT cách gọi tên bộ phận.")

    print("\n=== 2. DẠNG CÂU: mệnh lệnh nằm ở đâu ===")
    print(f"   mẫu GỌI HÀM ({dang['so_mau_duong']:>5} câu): {dang['menh_lenh_duong_%']:>5}% là câu mệnh lệnh")
    print(f"   mẫu TỪ CHỐI ({dang['so_mau_am']:>5} câu): {dang['menh_lenh_am_%']:>5}% là câu mệnh lệnh")
    print(f"   -> đoán bừa \"mệnh lệnh thì gọi hàm\" đúng {dang['goi_ham_dung_neu_doan_theo_menh_lenh_%']}% "
          "trên train.")
    print("      Càng gần 100% thì model càng có động cơ học tắt theo DẠNG CÂU thay vì theo nghĩa.")

    print("\n=== 3. MẪU HỎI LẠI đề xuất hàm nào ===")
    for h in hoi[:6]:
        print(f"   {h['ham']:22}{h['so_lan']:>6} lần  {h['ty_le_%']:>5}%")
    im = [h["ham"] for h in hoi if h["so_lan"] == 0]
    thap = [h["ham"] for h in hoi if 0 < h["so_lan"] <= 40]
    print(f"   ít được đề xuất (≤40 lần): {', '.join(thap) if thap else '(không)'}")
    print(f"   KHÔNG BAO GIỜ được đề xuất: {', '.join(im) if im else '(không)'}")

    print("\n=== 4. ĐÓNG KHUÔN CÂU (5 hàm khuôn nặng nhất) ===")
    for k in khuon[:5]:
        print(f"   {k['ham']:22}{k['ty_le_%']:>6}% câu mở đầu bằng \"{k['khuon_hay_nhat']}...\"")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {"khoa_tu_vung": khoa, "dang_cau": dang, "hoi_lai": hoi, "dong_khuon": khuon},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nĐã ghi -> {args.json}")


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
