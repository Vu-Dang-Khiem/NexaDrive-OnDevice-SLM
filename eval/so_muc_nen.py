"""So bản GGUF với mốc HF bf16 theo LUẬT ĐÃ KHOÁ (PLAN.md mục W, bước 3) + áp luật chọn ngưỡng (mục V).

    python -m eval.so_muc_nen --moc runs/tu_tin/val_lorav5_s7.jsonl \
        --ban f16=runs/gguf/val_f16.jsonl q4_k_m=runs/gguf/val_q4_k_m.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Any

from schema.loader import CATALOG

TU_CHOI = CATALOG.by_name["irrelevant_function"].nexa_token
ASK = CATALOG.nexa_ask
LUOI = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.97)
# luật khoá: rõ ràng đúng giảm ≤ 0,5% · gọi bừa tăng ≤ 2 · mơ hồ đúng giảm ≤ 2
GIAM_RO_RANG_TOI_DA = 0.5
TANG_GOI_BUA_TOI_DA = 2
GIAM_MO_HO_TOI_DA = 2


def doc(p: str) -> list[dict[str, Any]]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def kq_nguong(d: dict[str, Any], t: float) -> str:
    """Luật ngưỡng mục S: hạng 1 là hàm và p < t -> hạng 2 từ chối thì từ chối, khác thì hỏi lại."""
    (t1, p1), (t2, _) = d["top_buoc_1"][0], d["top_buoc_1"][1]
    if t1 in (TU_CHOI, ASK) or p1 >= t:
        return d["ket_qua"]
    qd = "tu_choi" if t2 == TU_CHOI else "hoi_lai"
    if d["nhom"] == "ngoai_pham_vi":
        return "dung" if qd == "tu_choi" else "hoi_lai_khi_phai_tu_choi"
    if d["nhom"] == "ro_rang":
        return "tu_choi_nham" if qd == "tu_choi" else "hoi_lai_thua"
    return "mo_ho_" + qd


def dem(R: list[dict[str, Any]], t: float = 0.0) -> dict[str, int]:
    k = [(d["nhom"], kq_nguong(d, t)) for d in R]
    return {
        "ro_rang_dung": sum(n == "ro_rang" and x == "dung" for n, x in k),
        "goi_bua": sum(n == "ngoai_pham_vi" and x == "goi_bua" for n, x in k),
        "mo_ho_dung": sum(n == "mo_ho" and x == "dung" for n, x in k),
    }


def dat_luat(moc: dict[str, int], ban: dict[str, int]) -> bool:
    giam_rr = 100 * (moc["ro_rang_dung"] - ban["ro_rang_dung"]) / moc["ro_rang_dung"]
    return (giam_rr <= GIAM_RO_RANG_TOI_DA
            and ban["goi_bua"] - moc["goi_bua"] <= TANG_GOI_BUA_TOI_DA
            and moc["mo_ho_dung"] - ban["mo_ho_dung"] <= GIAM_MO_HO_TOI_DA)


def chon_nguong(R: list[dict[str, Any]]) -> float | None:
    """Ngưỡng cao nhất mà rõ ràng đúng giảm ≤ 1% và gọi bừa giảm ≥ 1 (luật mục V)."""
    goc = dem(R)
    hop = [t for t in LUOI
           if 100 * (goc["ro_rang_dung"] - dem(R, t)["ro_rang_dung"]) / goc["ro_rang_dung"] <= 1.0
           and dem(R, t)["goi_bua"] <= goc["goi_bua"] - 1]
    return max(hop) if hop else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--moc", required=True)
    ap.add_argument("--ban", nargs="+", required=True, help="ten=duong_dan")
    a = ap.parse_args()
    M = doc(a.moc)
    moc_theo_cau = {(d["tap"], d["cau"]): d for d in M}
    cm = dem(M)
    print(f"MỐC HF bf16 ({len(M)} câu): {cm}")
    for muc in a.ban:
        ten, p = muc.split("=", 1)
        R = doc(p)
        cb = dem(R)
        cap = [(d, moc_theo_cau[(d["tap"], d["cau"])]) for d in R if (d["tap"], d["cau"]) in moc_theo_cau]
        trung_out = sum(d["output"] == m["output"] for d, m in cap)
        trung_tok = sum(d["top_buoc_1"][0][0] == m["top_buoc_1"][0][0] for d, m in cap)
        lech_p = [abs(d["p_buoc_1"] - m["p_buoc_1"]) for d, m in cap
                  if d["top_buoc_1"][0][0] == m["top_buoc_1"][0][0]]
        ms = sorted(d["ms_doc_prompt"] + d["ms_sinh"] for d in R)
        print(f"\n== {ten} ({len(R)} câu, ghép được {len(cap)} với mốc)")
        print(f"  {cb}  | chênh: rõ ràng {cb['ro_rang_dung'] - cm['ro_rang_dung']:+d} "
              f"({100 * (cb['ro_rang_dung'] - cm['ro_rang_dung']) / cm['ro_rang_dung']:+.2f}%), "
              f"gọi bừa {cb['goi_bua'] - cm['goi_bua']:+d}, mơ hồ {cb['mo_ho_dung'] - cm['mo_ho_dung']:+d}")
        print(f"  output trùng mốc {trung_out}/{len(cap)} · token đầu trùng {trung_tok}/{len(cap)} · "
              f"|Δp| tb {statistics.mean(lech_p):.4f}, lớn nhất {max(lech_p):.4f}")
        print(f"  độ trễ/câu (ms): p50 {ms[len(ms) // 2]:.0f} · p95 {ms[int(0.95 * len(ms))]:.0f} · "
              f"lớn nhất {ms[-1]:.0f}")
        ket_luan = "ĐẠT" if dat_luat(cm, cb) else "KHÔNG ĐẠT"
        print(f"  ==> LUẬT KHOÁ: {ket_luan} · ngưỡng theo luật mục V: {chon_nguong(R)}")
        in_bang_nguong(ten, R)
    in_bang_nguong("MỐC HF bf16", M)


def in_bang_nguong(ten: str, R: list[dict[str, Any]]) -> None:
    g = dem(R)
    print(f"  bảng ngưỡng {ten}: ngưỡng | rõ ràng đúng (giảm %) | gọi bừa | mơ hồ đúng")
    for t in (0.0,) + LUOI:
        c = dem(R, t)
        giam = 100 * (g["ro_rang_dung"] - c["ro_rang_dung"]) / g["ro_rang_dung"]
        nhan = t or "không"
        print(f"    {nhan:>6} | {c['ro_rang_dung']} ({giam:.2f}) | {c['goi_bua']} | {c['mo_ho_dung']}")


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
