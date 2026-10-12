"""Ngưỡng tin cậy cho v5 vs v6.1 (người dùng chọn "đo ngưỡng" 2026-10-06). Luật chọn ngưỡng Y HỆT mục V
(`eval.so_muc_nen.chon_nguong`: ngưỡng cao nhất trên VAL mà rõ ràng đúng giảm ≤ 1% và gọi bừa giảm ≥ 1); luật áp ngưỡng
`kq_nguong` (hạng 1 là hàm và p < ngưỡng -> hạng 2 từ chối thì từ chối, khác thì hỏi lại).

    python -m eval.nguong_v6 --buoc val     # tải độ tự tin trên val, chọn + in ngưỡng từng model (KHOÁ trước khi đo test)
    python -m eval.nguong_v6 --buoc test    # áp ngưỡng đã khoá lên test (tiêu chí như eval/cham_v6, mức TRIỂN KHAI)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from eval.cham_v6 import CU_CHI, mcnemar
from eval.so_muc_nen import chon_nguong, dem, in_bang_nguong, kq_nguong

RUNS = Path("runs/v6/tu_tin")
MODEL = ("v5s42", "v5s7", "v61s42", "v61s7")
# KHOÁ sau bước val (`--buoc val` -> 0,85 cho cả 4 model), ghi PLAN.md TRƯỚC khi chạy `--buoc test`
NGUONG: dict[str, float] = dict.fromkeys(MODEL, 0.85)


def tai(ten_file: str, thu_muc: str = "final_v6_1") -> list[dict[str, Any]]:
    import modal

    vol = modal.Volume.from_name("nexa-drive")
    RUNS.mkdir(parents=True, exist_ok=True)
    can = next(e.size for e in vol.listdir(f"/data/vi/{thu_muc}") if e.path.endswith(f"/{ten_file}"))
    p = RUNS / ten_file
    if not (p.exists() and p.stat().st_size == can):
        for _ in range(6):  # đường tải Modal -> máy này hay cắt cụt âm thầm (PLAN.md)
            du = b"".join(vol.read_file(f"/data/vi/{thu_muc}/{ten_file}"))
            if len(du) == can:
                break
        else:
            raise RuntimeError(f"tải {ten_file} không đủ")
        p.write_bytes(du)
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def buoc_val() -> None:
    for m in MODEL:
        R = tai(f"tu_tin_val_{m}.jsonl")
        print(f"\n== {m}: {len(R)} câu val; không ngưỡng {dem(R)}; NGƯỠNG THEO LUẬT MỤC V: {chon_nguong(R)}")
        in_bang_nguong(m, R)


def dung(R: list[dict[str, Any]], t: float) -> list[bool]:
    return [kq_nguong(d, t) == "dung" for d in R]


def ty_le(ds: list[bool]) -> float:
    return round(100 * sum(ds) / len(ds), 1) if ds else 0.0


def so_sanh(a: list[bool], b: list[bool]) -> str:
    hon = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    kem = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    return f"{ty_le(a):5.1f} vs {ty_le(b):5.1f} (mới hơn {hon}, cũ hơn {kem}, p={mcnemar(hon, kem):.4f})"


def buoc_test() -> None:
    from eval.cham_v6 import TEST, doc
    from schema.call_format import render_completion
    from serve import bo_nho, tra_loi
    from serve.xe_mo_phong import TrangThaiXe

    hai_luot = doc(TEST / "test_v6_hai_luot.jsonl")
    tb: dict[str, list[tuple[float, float]]] = {}
    for s in ("s42", "s7"):
        R5 = tai(f"tu_tin_test_v5{s}.jsonl", "final_v6")
        R6 = tai(f"tu_tin_test_v61{s}.jsonl", "final_v6")
        t5, t6 = NGUONG[f"v5{s}"], NGUONG[f"v61{s}"]

        def theo(R: list[dict[str, Any]], tap: str, dk=lambda d: True) -> list[dict[str, Any]]:
            return [d for d in R if d["tap"] == tap and dk(d)]

        print(f"\n===== hạt giống {s} — CÓ NGƯỠNG (v6.1 {t6} / v5 {t5}) =====")
        for ten, dk in (("A+B", lambda d: d["nhom_phu"] in ("v6_A", "v6_B")), ("A", lambda d: d["nhom_phu"] == "v6_A"),
                        ("B", lambda d: d["nhom_phu"] == "v6_B"), ("C", lambda d: d["nhom_phu"] == "v6_C")):
            print(f"  ({'1' if ten == 'A+B' else '3' if ten == 'C' else ' '}) {ten:4s}",
                  so_sanh(dung(theo(R6, "test_v6_mot_luot", dk), t6), dung(theo(R5, "test_v6_mot_luot", dk), t5)))
        # (2) hai lượt: v6.1 có lượt trước; v5 + luật (luật kích hoạt thì dùng luật, không thì v5 không nhớ CÓ NGƯỠNG)
        d6 = dung(theo(R6, "test_v6_hai_luot"), t6)
        d5_khong = dung(theo(R5, "test_v6_hai_luot_khong_nho"), t5)
        luat = []
        for r, k in zip(hai_luot, d5_khong, strict=True):
            m = r["meta"]
            h = tra_loi.huong_nhiet(m["cau_1"]) or bo_nho.huong_cau(m["cau_1"], None)
            tiep = bo_nho.noi_tiep(m["cau"], bo_nho.LuotTruoc(m["fn_1"], m["args_1"],
                                                               bo_nho.huong_cua(m["fn_1"], m["args_1"], h)), TrangThaiXe())
            if tiep is None:
                luat.append(k)
            else:
                from eval.metrics import DUNG, cham_mot_cau

                luat.append(cham_mot_cau(render_completion([tiep]), r["completion"])[0] == DUNG)
        print("  (2) hai lượt: v6.1 vs v5+luật", so_sanh(d6, luat), "| v5 không nhớ", ty_le(d5_khong))
        print("  (4) không tụt:")
        for ten, (tap, nhom) in CU_CHI.items():
            dk = (lambda d, n=nhom: d["nhom"] == n) if nhom else (lambda d: True)
            a, b = dung(theo(R6, tap, dk), t6), dung(theo(R5, tap, dk), t5)
            tb.setdefault(ten, []).append((ty_le(a), ty_le(b)))
            print(f"      {ten:10s} {so_sanh(a, b)}")
        print("  (5) nhiễu", so_sanh(dung(theo(R6, "test_v6_nhieu"), t6), dung(theo(R5, "test_v6_nhieu"), t5)))
    print("\n===== (4) CÓ NGƯỠNG, trung bình 2 hạt giống — ngưỡng: rõ ràng / ngoài pv / mo_ho_4 không thấp hơn v5 quá 3 =====")
    for ten, ds in tb.items():
        a, b = sum(x for x, _ in ds) / 2, sum(y for _, y in ds) / 2
        print(f"  {ten:10s} v6.1 {a:5.1f} | v5 {b:5.1f} | chênh {a - b:+5.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buoc", choices=("val", "test"), required=True)
    a = ap.parse_args()
    if a.buoc == "val":
        buoc_val()
    else:
        buoc_test()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
