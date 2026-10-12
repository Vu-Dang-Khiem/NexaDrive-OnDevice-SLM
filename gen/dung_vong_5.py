"""Vòng 5 — cặp đối chứng thứ ba: câu HỎI LẠI kể hoàn cảnh (PLAN.md mục U). Chạy ở máy, miễn phí.

Bước "cứu câu cùng khuôn": `sinh_hoi_lai_hoan_canh` lọc cùng khuôn bằng `bo_cung_khuon` (ngưỡng 0,3, so với
MỌI câu đã giữ — kể cả khác hàm). Ngưỡng đó chọn trên 64 câu chạy thử; ở quy mô ~4.700 câu nó loại 2.962 câu.
Soi 400 câu bị loại: 38% có độ giống cao nhất (cùng hàm) < 0,35 — bị loại oan vì câu tiếng Việt ngắn hay chung
"quá", "mình", "xe", "này"; 52% ở 0,35–0,5 (phần lớn là diễn đạt lại cùng ý); 9% > 0,5. Lọc lại: chỉ so TRONG
CÙNG HÀM, ngưỡng 0,45 -> bỏ câu gần trùng thật, cứu câu chỉ chung vài chữ. Câu cứu được đi lại ĐÚNG quy trình
(teacher gán nhãn không biết hàm đích, giữ câu khớp, rồi giám khảo).

    python -m gen.dung_vong_5 --cuu data/processed/v5/v5_hoi_lai_hc_chi_tiet.jsonl \\
        --ra data/processed/v5/v5_cuu_cung_khuon.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from gen.dung_vong_2 import SEED, P, doc, ghi
from gen.loc import bo_cung_khuon
from gen.records import IM_END, query_from_call_prompt
from schema.call_format import parse_calls, render_completion

NGUONG_CUU = 0.45
# lý do loại ở bước code mà KHÔNG được cứu (câu hỏng thật)
_LOAI_CODE_THAT = {"loc_code", "tieng_anh", "qua_ngan", "ra_lenh", "trung"}


def cuu_cung_khuon(chi_tiet: list[dict[str, Any]], nguong: float = NGUONG_CUU) -> list[dict[str, Any]]:
    """Trả về các câu bị loại 'cung_khuon' nay được cứu: không gần trùng (≥ `nguong`) với câu nào CÙNG
    HÀM đã qua bước code, cũng không gần trùng nhau. Câu đã qua bước code (kể cả bị teacher loại sau đó)
    đứng trước trong thứ tự xét, nên luôn được ưu tiên giữ."""
    theo_ham_da_qua: dict[str, list[str]] = defaultdict(list)
    theo_ham_cuu: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in chi_tiet:
        if r["loai_vi"] == "cung_khuon":
            theo_ham_cuu[r["fn_dich"]].append(r)
        elif r["loai_vi"] not in _LOAI_CODE_THAT:
            theo_ham_da_qua[r["fn_dich"]].append(r["query"])
    ra: list[dict[str, Any]] = []
    for ham, ung_vien in theo_ham_cuu.items():
        cu = theo_ham_da_qua.get(ham, [])
        giu = bo_cung_khuon(cu + [r["query"] for r in ung_vien], nguong=nguong)
        ra += [ung_vien[i - len(cu)] for i in giu if i >= len(cu)]
    return ra


# ── Lỗi có hệ thống tìm ra khi người dùng kiểm tay 52 câu (90,4% đúng; PLAN.md mục U) ─────────────────────
# Mỗi luật bắt đúng KIỂU lỗi người kiểm thấy, áp cho MỌI mẫu hỏi lại (mới lẫn cũ) — không sửa riêng 52 câu.
_ON = re.compile(r"ồn|inh|ầm|to quá|nhức")
_NGOAI = re.compile(r"bên ngoài|ngoài đường|ngoài kia|ngoài trời|hàng xóm|xe cộ|công trình|khoan")
_KINH_MO = re.compile(r"mờ|đọng|hơi nước|hấp hơi|sương|nhòe|nhoè|đổ mồ hôi|bám hơi")
_DEN_DA_BAT = re.compile(
    r"(đèn pha|đèn xe|đèn (của )?(mình|tao|tôi|em)|đèn nhà mình)[^.,]{0,20}(không sáng|yếu|mờ|tối|chưa đủ)"
    r"|không sáng bằng|đủ sáng không|có đủ sáng"
)


def sua_loi_he_thong(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Một mẫu `<nexa_ask>` -> (mẫu sau sửa | None nếu loại, mã lý do; "" = giữ nguyên).

    - `on_ngoai_am_luong` (loại): ồn từ BÊN NGOÀI mà gợi ý âm lượng — luật của chính bài: ồn bên ngoài
      -> đóng kính (`build_verify_ask_prompt`). Người kiểm: dòng 21, 29.
    - `say_kinh_khong_mo` (loại): sấy kính mà câu không có dấu hiệu kính MỜ (bám từ "kính": "nắng gắt,
      kính như lò sưởi" -> sấy kính). Dòng 47.
    - `den_da_bat_yeu` (loại): đèn ĐÃ bật mà yếu -> gợi ý đèn gần vô nghĩa (hệ quả của luật đổi đèn xa
      -> đèn gần khi gợi ý). Dòng 49. Không bắt "đèn ĐƯỜNG yếu" (đèn phố — gợi ý bật đèn xe là đúng).
    - `quat_tat_thanh_giam` (sửa): gợi ý TẮT quạt (`level=0`) cho câu gió mạnh / ồn -> GIẢM (`level=2`),
      như quy ước các tập test; tắt hẳn là quá tay (cùng tinh thần đổi đèn xa -> đèn gần). Dòng 36.
    """
    cau = query_from_call_prompt(row["prompt"]).lower()
    calls = [(c.name, dict(c.args)) for c in parse_calls(row["completion"].replace(IM_END, ""))]
    ten = [n for n, _ in calls]
    if ten[0] == "set_volume" and _NGOAI.search(cau) and _ON.search(cau):
        return None, "on_ngoai_am_luong"
    if "set_defrost" in ten and not _KINH_MO.search(cau):
        return None, "say_kinh_khong_mo"
    if "set_headlights" in ten and _DEN_DA_BAT.search(cau):
        return None, "den_da_bat_yeu"
    if any(n == "set_fan_speed" and a.get("level") == 0 for n, a in calls):
        moi = [
            (n, {**a, "level": 2} if n == "set_fan_speed" and a.get("level") == 0 else a) for n, a in calls
        ]
        return {**row, "completion": render_completion(moi, ask=True) + IM_END}, "quat_tat_thanh_giam"
    return row, ""


NEN_V5 = P / "final_v4_doi_chung"  # bản đối chứng — model chính thức hiện tại
HOI_LAI_HC = P / "v5" / "v5_hoi_lai_hc_gop.jsonl"


def dung_v5() -> dict[str, Any]:
    """Train v5 = bản đối chứng + câu hỏi lại kể hoàn cảnh (đã qua giám khảo); MỌI mẫu hỏi lại (cũ và mới)
    qua `sua_loi_he_thong`. Val giữ y hệt bản đối chứng (chọn checkpoint theo nhóm rõ ràng — so được với
    vòng trước)."""
    nen = doc(NEN_V5 / "train_vi.jsonl")
    moi = doc(HOI_LAI_HC)
    da_co = {r["prompt"] for r in nen}
    dem: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for nguon, ds in (("cu", nen), ("moi", moi)):
        for r in ds:
            if nguon == "moi" and r["prompt"] in da_co:
                dem["moi_trung_train_cu"] += 1
                continue
            if r["meta"]["type"] != "ask":
                rows.append(r)
                continue
            r2, ly_do = sua_loi_he_thong(r)
            if ly_do:
                dem[f"{nguon}:{ly_do}"] += 1
            if r2 is not None:
                nhan_nguon = "hoi_lai_hc" if nguon == "moi" else "cu"
                rows.append({**r2, "meta": {**r2["meta"], "nguon_v5": nhan_nguon}})
    random.Random(SEED).shuffle(rows)
    ra = P / "final_v5"
    ghi(ra / "train_vi.jsonl", rows)
    for tap in ("val_vi", "val_mau"):
        ghi(ra / f"{tap}.jsonl", doc(NEN_V5 / f"{tap}.jsonl"))
    hoi_lai = [r for r in rows if r["meta"]["type"] == "ask"]
    bao_cao = {
        "train": len(rows),
        "theo_loai": dict(Counter(r["meta"]["type"] for r in rows)),
        "hoi_lai_cu": sum(r["meta"]["nguon_v5"] == "cu" for r in hoi_lai),
        "hoi_lai_hc_moi": sum(r["meta"]["nguon_v5"] == "hoi_lai_hc" for r in hoi_lai),
        "hoi_lai_dai_hon_10_chu": sum(len(query_from_call_prompt(r["prompt"]).split()) > 10 for r in hoi_lai),
        "sua_loai": dict(dem),
    }
    (ra / "bao_cao_dung.json").write_text(json.dumps(bao_cao, ensure_ascii=False, indent=2), encoding="utf-8")
    return bao_cao


def main() -> None:
    ap = argparse.ArgumentParser(description="Vòng 5: cứu câu cùng khuôn (--cuu) hoặc dựng train v5 (--dung)")
    ap.add_argument("--dung", action="store_true", help="dựng data/processed/final_v5")
    ap.add_argument("--cuu", type=Path)
    ap.add_argument("--ra", type=Path)
    ap.add_argument("--nguong", type=float, default=NGUONG_CUU)
    a = ap.parse_args()
    if a.dung:
        print(json.dumps(dung_v5(), ensure_ascii=False, indent=2))
        return
    chi_tiet = [json.loads(x) for x in open(a.cuu, encoding="utf-8") if x.strip()]
    cuu = cuu_cung_khuon(chi_tiet, a.nguong)
    with open(a.ra, "w", encoding="utf-8") as f:
        for r in cuu:
            dong = {k: r[k] for k in ("query", "fn_dich", "vai", "bay")}
            f.write(json.dumps(dong, ensure_ascii=False) + "\n")
    dem: dict[str, int] = defaultdict(int)
    for r in cuu:
        dem[r["fn_dich"]] += 1
    tong_ck = sum(1 for r in chi_tiet if r["loai_vi"] == "cung_khuon")
    print(f"cứu {len(cuu)}/{tong_ck} câu cùng khuôn (ngưỡng {a.nguong}, chỉ so cùng hàm)")
    print(" ", dict(sorted(dem.items())))


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
