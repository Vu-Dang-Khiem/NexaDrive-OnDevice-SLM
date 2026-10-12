"""Chấm model mới (v6 / v6.1) vs v5 theo 5 TIÊU CHÍ ĐÃ KHOÁ TRƯỚC (PLAN.md "KẾ HOẠCH v6", 2026-10-06). Model
HF đã gộp, KHÔNG ngưỡng (như các số HF của v5 trong PLAN). Chấm luật C (`eval.metrics.cham_mot_cau`), nhãn test
KHÔNG sửa.

    python -m eval.cham_v6 --moi v6 --tai     # tải dự đoán từ volume (kiểm kích thước, tải lại tới khi đủ) rồi chấm
    python -m eval.cham_v6 --moi v61          # chỉ chấm (đã có runs/v6/*.jsonl)

Checkpoint (luật cũ trên val mẫu, hoà -> lấy sớm như v5): v6 s42 / s7 ck3596; v5 s42 ck3624, v5 s7 ck2416 (v5 gộp
lại trên tài khoản Modal mới từ adapter cũ — bản s7 trùng mã băm từng byte với bản đã triển khai).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from eval.metrics import DUNG, cham_mot_cau, doc_output, nhom_cua_dap_an
from schema.call_format import render_completion
from schema.loader import CATALOG
from serve import bo_nho, tra_loi
from serve.xe_mo_phong import TrangThaiXe

TEST = Path("data/processed/final")
RUNS = Path("runs/v6")
HAT = ("s42", "s7")
TAP = ("test_vi_sach_v2", "test_vi_bo_sung", "test_vi_mo_ho_4", "test_v6_mot_luot", "test_v6_hai_luot",
       "test_v6_hai_luot_khong_nho", "test_v6_nhieu")
CU_CHI = {  # (tập, nhóm) của tiêu chí (4); None = cả tập
    "rõ ràng": ("test_vi_sach_v2", "ro_rang"), "mơ hồ": ("test_vi_sach_v2", "mo_ho"),
    "ngoài pv": ("test_vi_sach_v2", "ngoai_pham_vi"), "bổ sung": ("test_vi_bo_sung", None),
    "mo_ho_4": ("test_vi_mo_ho_4", None),
}


def mcnemar(b: int, c: int) -> float:  # chính xác, hai phía (giống eval/bao_cao)
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def tai(moi: str) -> None:
    import modal

    vol = modal.Volume.from_name("nexa-drive")
    RUNS.mkdir(parents=True, exist_ok=True)
    kich_thuoc = {e.path.rsplit("/", 1)[-1]: e.size for e in vol.listdir("/data/vi/final_v6")}
    for ban in ("v5", moi):
        for s in HAT:
            for t in TAP:
                if ban == "v5" and t == "test_v6_hai_luot":
                    continue  # v5 chưa học dòng "Lượt trước" -> chấm v5 trên bản không nhớ + luật
                if ban != "v5" and t == "test_v6_hai_luot_khong_nho" and f"du_doan_{ban}{s}_{t}.jsonl" not in kich_thuoc:
                    continue  # model mới chấm trên bản CÓ lượt trước
                ten = f"du_doan_{ban}{s}_{t}.jsonl"
                if (RUNS / ten).exists() and (RUNS / ten).stat().st_size == kich_thuoc[ten]:
                    continue
                for _ in range(6):  # đường tải Modal -> máy này hay cắt cụt âm thầm (PLAN.md)
                    du = b"".join(vol.read_file(f"/data/vi/final_v6/{ten}"))
                    if len(du) == kich_thuoc[ten]:
                        break
                else:
                    raise RuntimeError(f"tải {ten} không đủ sau 6 lần")
                (RUNS / ten).write_bytes(du)


def doc(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]


def dung_tung_cau(tap: str, ban_hat: str) -> list[bool]:
    dap_an = doc(TEST / f"{tap}.jsonl")
    du = {r["prompt"]: r["output"] for r in doc(RUNS / f"du_doan_{ban_hat}_{tap}.jsonl")}
    assert len(du) == len(dap_an), f"{ban_hat} {tap}: {len(du)} dự đoán / {len(dap_an)} câu"
    return [cham_mot_cau(du[r["prompt"]], r["completion"])[0] == DUNG for r in dap_an]


def v5_cong_luat(hat: str) -> list[bool]:
    """v5 + luật bộ nhớ (serve/bo_nho.py): câu nối tiếp CHỈ gồm một động từ -> suy từ lượt 1; còn lại dùng dự đoán
    v5 KHÔNG nhớ (prompt không có lượt trước) — đúng như runtime đang chạy."""
    khong_nho = {r["prompt"]: r["output"]
                 for r in doc(RUNS / f"du_doan_v5{hat}_test_v6_hai_luot_khong_nho.jsonl")}
    kq = []
    for r in doc(TEST / "test_v6_hai_luot.jsonl"):
        m = r["meta"]
        h = tra_loi.huong_nhiet(m["cau_1"]) or bo_nho.huong_cau(m["cau_1"], None)
        truoc = bo_nho.LuotTruoc(m["fn_1"], m["args_1"], bo_nho.huong_cua(m["fn_1"], m["args_1"], h))
        tiep = bo_nho.noi_tiep(m["cau"], truoc, TrangThaiXe())
        out = render_completion([tiep]) if tiep is not None else khong_nho[CATALOG.render_call_prompt(m["cau"])]
        kq.append(cham_mot_cau(out, r["completion"])[0] == DUNG)
    return kq


def ty_le(ds: list[bool]) -> float:
    return round(100 * sum(ds) / len(ds), 1) if ds else 0.0


def so_sanh(a: list[bool], b: list[bool]) -> str:
    """a = mới, b = cũ: tỉ lệ + McNemar."""
    hon = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    kem = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    return f"{ty_le(a):5.1f} vs {ty_le(b):5.1f} (mới hơn {hon}, cũ hơn {kem}, p={mcnemar(hon, kem):.4f})"


def chi_so(tap: str, dk) -> list[int]:
    return [i for i, r in enumerate(doc(TEST / f"{tap}.jsonl")) if dk(r)]


def chon(ds: list[bool], idx: list[int]) -> list[bool]:
    return [ds[i] for i in idx]


def cham(moi: str) -> None:
    nhom = {t: [nhom_cua_dap_an(doc_output(r["completion"])) for r in doc(TEST / f"{t}.jsonl")]
            for t in ("test_vi_sach_v2", "test_vi_bo_sung", "test_vi_mo_ho_4")}
    phan = {p: chi_so("test_v6_mot_luot", lambda r, p=p: r["meta"]["nhom_phu"] in p)
            for p in (("v6_A", "v6_B"), ("v6_A",), ("v6_B",), ("v6_C",))}
    tb: dict[str, list[tuple[float, float]]] = {}
    for s in HAT:
        v5 = {t: dung_tung_cau(t, f"v5{s}") for t in TAP if t != "test_v6_hai_luot"}
        n6 = {t: dung_tung_cau(t, f"{moi}{s}") for t in TAP if t != "test_v6_hai_luot_khong_nho"}
        m1, c1 = n6["test_v6_mot_luot"], v5["test_v6_mot_luot"]
        print(f"\n===== hạt giống {s}  ({moi} vs v5) =====")
        print("(1) CHÍNH — A+B (120):", so_sanh(chon(m1, phan[("v6_A", "v6_B")]), chon(c1, phan[("v6_A", "v6_B")])))
        print("      A ngắn:", so_sanh(chon(m1, phan[("v6_A",)]), chon(c1, phan[("v6_A",)])))
        print("      B cách gọi:", so_sanh(chon(m1, phan[("v6_B",)]), chon(c1, phan[("v6_B",)])))
        luat, khong = v5_cong_luat(s), v5["test_v6_hai_luot_khong_nho"]
        hl = n6["test_v6_hai_luot"]
        print(f"(2) HAI LƯỢT (60): {moi} {ty_le(hl)} | v5 + luật {ty_le(luat)} | v5 không nhớ {ty_le(khong)}")
        print(f"      {moi} vs v5+luật:", so_sanh(hl, luat))
        for k in ("doi_chieu", "them_nua", "bo_phan_khac", "lenh_moi", "ngoai_pham_vi"):
            idx = chi_so("test_v6_hai_luot", lambda r, k=k: r["meta"]["kieu_de"] == k)
            print(f"      {k:14s} {moi} {ty_le(chon(hl, idx)):5.1f} | v5+luật {ty_le(chon(luat, idx)):5.1f} | "
                  f"v5 không nhớ {ty_le(chon(khong, idx)):5.1f}")
        print(f"(3) Tính năng mới C (30): {moi} {ty_le(chon(m1, phan[('v6_C',)]))} "
              f"(v5 tham khảo {ty_le(chon(c1, phan[('v6_C',)]))})")
        print("(4) KHÔNG TỤT trên test cũ:")
        for ten, (t, g) in CU_CHI.items():
            idx = [i for i, x in enumerate(nhom[t]) if g is None or x == g]
            a, b = chon(n6[t], idx), chon(v5[t], idx)
            tb.setdefault(ten, []).append((ty_le(a), ty_le(b)))
            print(f"      {ten:10s} {so_sanh(a, b)}")
        print("(5) Nhiễu E (104):", so_sanh(n6["test_v6_nhieu"], v5["test_v6_nhieu"]))
    print("\n===== (4) trung bình 2 hạt giống — ngưỡng: rõ ràng / ngoài pv / mo_ho_4 không thấp hơn v5 quá 3 điểm =====")
    for ten, ds in tb.items():
        a, b = sum(x for x, _ in ds) / len(ds), sum(y for _, y in ds) / len(ds)
        print(f"  {ten:10s} {moi} {a:5.1f} | v5 {b:5.1f} | chênh {a - b:+5.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--moi", default="v6", help="tên model mới trong tên file dự đoán (v6 / v61)")
    ap.add_argument("--tai", action="store_true")
    a = ap.parse_args()
    if a.tai:
        tai(a.moi)
    cham(a.moi)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
