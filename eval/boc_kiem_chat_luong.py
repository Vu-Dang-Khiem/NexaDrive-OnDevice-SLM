"""Bốc ~200 mẫu GỌI HÀM + TỪ CHỐI từ dữ liệu train vòng 2 để NGƯỜI kiểm chất lượng.

Vì sao: 400 câu người đã gán chỉ kiểm được mẫu HỎI LẠI (~5% dữ liệu train). 95% còn lại (gọi hàm,
trả lời lượt 2, từ chối) mới có Llama đồng ý 95,2% — máy với máy, không phải chất lượng. Các paper
làm dataset gọi hàm đều báo con số người kiểm: APIGen 600 mẫu (95,3%), Huawei 2026 §A.7 217 mẫu phân
tầng (100% đúng). Xem PLAN.md.

Bốc PHÂN TẦNG (như Huawei): mỗi hàm vài mẫu gọi hàm, vài mẫu trả lời lượt 2, và hai nhóm từ chối
(dễ / khó dạng mệnh lệnh) — để hàm hiếm và loại hiếm đều có mặt. Người chỉ đánh ĐÚNG / SAI, không
phải tự nghĩ đáp án (khác đợt 400 câu).

    python -m eval.boc_kiem_chat_luong
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from gen.records import query_from_call_prompt, result_from_reply_prompt
from schema.call_format import parse_calls, render_readable

NGUON = Path("data/processed/final_v2/train_vi.jsonl")
RA_CSV = Path("data/nhan_tay/kiem_200.csv")
RA_KHOA = Path("data/nhan_tay/kiem_200_khoa.jsonl")

GOI_HAM_MOI_HAM = 5
TRA_LOI_MOI_HAM = 1
TU_CHOI_DE = 25
TU_CHOI_KHO = 25
SEED = 2027

_LUOT_1 = re.compile(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", re.S)


def de_doc(completion: str) -> str:
    """Chuỗi functional token -> dạng người đọc được: 'control_window(window="all", position="down")'."""
    calls = parse_calls(completion.replace("<|im_end|>", ""))
    if any(c.name == "irrelevant_function" for c in calls):
        return "TỪ CHỐI (ngoài khả năng của xe)"
    return " ; ".join(render_readable(c.name, dict(c.args)) for c in calls)


def tang(r: dict[str, Any]) -> str:
    m = r["meta"]
    if m["fn"] == "irrelevant_function":
        if m["type"] == "reply":
            return "tra_loi|tu_choi"
        return "tu_choi_kho" if m.get("nhom_phu") == "tu_choi_kho" else "tu_choi_de"
    return f"{m['type']}|{m['fn']}"


def boc(seed: int = SEED) -> list[dict[str, Any]]:
    rows = [json.loads(x) for x in NGUON.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [r for r in rows if r["meta"]["type"] != "ask"]  # hỏi lại đã kiểm ở đợt 400 câu
    theo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        theo[tang(r)].append(r)
    rng = random.Random(seed)
    ra: list[dict[str, Any]] = []
    for k in sorted(theo):
        n = (TU_CHOI_DE if k == "tu_choi_de" else TU_CHOI_KHO if k == "tu_choi_kho"
             else GOI_HAM_MOI_HAM if k.startswith("call|") else TRA_LOI_MOI_HAM)
        ra += rng.sample(theo[k], min(n, len(theo[k])))
    rng.shuffle(ra)
    return ra


def dong(stt: int, r: dict[str, Any]) -> dict[str, str]:
    cau = query_from_call_prompt(r["prompt"])
    if r["meta"]["type"] == "reply":
        m = _LUOT_1.search(r["prompt"])
        return {
            "stt": str(stt), "loai": "trả lời lượt 2", "cau": cau,
            "dap_an": de_doc(m.group(1)) if m else "?",
            "ket_qua_ham": json.dumps(result_from_reply_prompt(r["prompt"]), ensure_ascii=False),
            "cau_tra_loi": r["completion"].replace("<|im_end|>", "").strip(),
            "danh_gia": "", "ghi_chu": "",
        }
    loai = "từ chối" if r["meta"]["fn"] == "irrelevant_function" else "gọi hàm"
    return {"stt": str(stt), "loai": loai, "cau": cau, "dap_an": de_doc(r["completion"]),
            "ket_qua_ham": "", "cau_tra_loi": "", "danh_gia": "", "ghi_chu": ""}


def main() -> None:
    ra = boc()
    RA_CSV.parent.mkdir(parents=True, exist_ok=True)
    cot = ["stt", "loai", "cau", "dap_an", "ket_qua_ham", "cau_tra_loi", "danh_gia", "ghi_chu"]
    with open(RA_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cot)
        w.writeheader()
        for i, r in enumerate(ra, 1):
            w.writerow(dong(i, r))
    with open(RA_KHOA, "w", encoding="utf-8") as f:
        for i, r in enumerate(ra, 1):
            f.write(json.dumps({"stt": i, "tang": tang(r), **r}, ensure_ascii=False) + "\n")
    dem: dict[str, int] = defaultdict(int)
    for r in ra:
        dem[tang(r).split("|")[0]] += 1
    print(f"Đã bốc {len(ra)} mẫu -> {RA_CSV}")
    print("   theo loại:", dict(dem))


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
