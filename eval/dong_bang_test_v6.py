"""Tập test v6 — ĐÓNG BĂNG: nhãn ChatGPT (`2_da_gan_nhan_210.csv`) + sửa sau duyệt (`4_sua_sau_duyet.csv`) ->
jsonl.

    python -m eval.dong_bang_test_v6

Ra hai file trong data/processed/final/:
  test_v6_mot_luot.jsonl  phần A / B / C (meta.nhom_phu = v6_A / v6_B / v6_C)
  test_v6_hai_luot.jsonl  phần D — prompt ĐỊNH DẠNG v6 ("Lượt trước: <lời gọi lượt 1>\\n<câu lượt 2>");
                          meta giữ câu và lời gọi lượt 1 + kiểu nối tiếp để chấm được cả v5 (không nhớ /
                          luật bộ nhớ) lẫn v6.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from eval.chuyen_test import dong_thanh_row
from eval.kiem_test_v6 import doc_csv
from eval.metrics import doc_output
from schema.call_format import render_call
from schema.loader import CATALOG

THU_MUC = Path("data/test_v6")
RA = Path("data/processed/final")


def dung(nhan: list[dict[str, str]], sua: list[dict[str, str]], khoa: dict[str, dict[str, str]]
         ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    theo_id = {r["id"]: dict(r) for r in nhan}
    for s in sua:
        theo_id[s["id"]][s["cot"]] = s["nhan_moi"]
    mot, hai = [], []
    for i, r in sorted(theo_id.items(), key=lambda x: int(x[0])):
        k = khoa[i]
        row = dong_thanh_row(r["cau"], r["nhan"], nhom_phu=f"v6_{k['phan']}")
        row["meta"].update(nguon="chatgpt_nguoi_duyet", id_v6=int(i), ham_de=k["ham_dich"], kieu_de=k["kieu"])
        if k["phan"] != "D":
            mot.append(row)
            continue
        r1 = dong_thanh_row(r["luot_1"], r["nhan_luot_1"])
        ten1, args1 = r1["meta"]["fn"], dict(doc_output(r1["completion"]).calls[0][1])
        goi_1 = render_call(ten1, args1)
        row["prompt"] = CATALOG.render_call_prompt(r["cau"], goi_1)
        row["meta"].update(cau_1=r["luot_1"], fn_1=ten1, args_1=args1, goi_1=goi_1, cau=r["cau"])
        hai.append(row)
    return mot, hai


def ghi(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    nhan = doc_csv(THU_MUC / "2_da_gan_nhan_210.csv")
    with open(THU_MUC / "4_sua_sau_duyet.csv", encoding="utf-8-sig", newline="") as f:
        sua = list(csv.DictReader(f))
    khoa = {r["id"]: r for r in doc_csv(THU_MUC / "khoa_id.csv")}
    mot, hai = dung(nhan, sua, khoa)
    ghi(RA / "test_v6_mot_luot.jsonl", mot)
    ghi(RA / "test_v6_hai_luot.jsonl", hai)
    print(f"một lượt: {len(mot)} {dict(Counter(r['meta']['nhom_phu'] for r in mot))}")
    print(f"hai lượt: {len(hai)} {dict(Counter(r['meta']['kieu_de'] for r in hai))}")
    print("loại nhãn:", dict(Counter(r["meta"]["type"] for r in mot + hai)),
          "từ chối:", sum(r["meta"]["fn"] == "irrelevant_function" for r in mot + hai))


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
