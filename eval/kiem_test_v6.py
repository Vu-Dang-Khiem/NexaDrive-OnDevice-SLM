"""Tập test v6 — bước 1 -> bước 2: kiểm file câu ChatGPT viết (người dùng đã sửa tay) rồi tạo file cho bước
gán nhãn (bỏ cột chức năng / yêu cầu / phần — người gán nhãn không được thấy "đáp án định sẵn").

    python -m eval.kiem_test_v6 --vao data/test_v6/1_viet_cau_210_da_viet.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def doc_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [{k.strip(): (v or "").strip() for k, v in r.items() if k} for r in csv.DictReader(f)]


def kiem(goc: list[dict[str, str]], da_viet: list[dict[str, str]]) -> list[str]:
    """Danh sách vấn đề (rỗng = ổn)."""
    loi: list[str] = []
    theo_id = {r["id"]: r for r in da_viet}
    if len(da_viet) != len(goc) or set(theo_id) != {r["id"] for r in goc}:
        thieu = sorted({r["id"] for r in goc} - set(theo_id))[:10]
        loi.append(f"số dòng / id lệch: {len(da_viet)} dòng, thiếu {thieu}")
    for g in goc:
        r = theo_id.get(g["id"])
        if r is None:
            continue
        for cot in ("phan", "chuc_nang", "yeu_cau"):
            if r.get(cot, "") != g[cot]:
                loi.append(f"id {g['id']}: cột {cot} bị đổi")
        if not r.get("cau"):
            loi.append(f"id {g['id']}: thiếu câu")
        if g["phan"] == "A" and len(r.get("cau", "").split()) > 4:
            loi.append(f"id {g['id']} (A): {len(r['cau'].split())} chữ > 4 — \"{r['cau']}\"")
        if g["phan"] == "D" and not r.get("luot_1"):
            loi.append(f"id {g['id']} (D): thiếu câu lượt 1")
        if g["phan"] != "D" and r.get("luot_1"):
            loi.append(f"id {g['id']}: phần {g['phan']} mà có câu lượt 1")
    dem = Counter(" ".join(r.get("cau", "").lower().split()) for r in da_viet if r.get("cau"))
    for c, n in dem.items():
        if n > 1:
            loi.append(f"câu trùng {n} lần: \"{c}\"")
    return loi


def tao_file_gan_nhan(da_viet: list[dict[str, str]], ra: Path) -> None:
    with open(ra, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "luot_1", "cau", "nhan_luot_1", "nhan", "ghi_chu"])
        w.writeheader()
        for r in da_viet:
            w.writerow({"id": r["id"], "luot_1": r.get("luot_1", ""), "cau": r["cau"], "nhan_luot_1": "",
                        "nhan": "", "ghi_chu": ""})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vao", type=Path, required=True)
    ap.add_argument("--goc", type=Path, default=Path("data/test_v6/1_viet_cau.csv"))
    ap.add_argument("--ra", type=Path, default=Path("data/test_v6/2_cho_gan_nhan.csv"))
    a = ap.parse_args()
    goc, da_viet = doc_csv(a.goc), doc_csv(a.vao)
    loi = kiem(goc, da_viet)
    print(f"{len(da_viet)} dòng; theo phần: {dict(Counter(r.get('phan') for r in da_viet))}")
    print("\n".join(loi) if loi else "không có vấn đề")
    if not loi:
        tao_file_gan_nhan(da_viet, a.ra)
        print(f"đã tạo {a.ra}")


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
