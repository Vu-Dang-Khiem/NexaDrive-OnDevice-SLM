"""GHÉP HAI STT (PLAN.md "GHÉP HAI STT … cách (a)", khoá 2026-10-12): mỗi câu có chữ của PhoWhisper-small và
Zipformer; đưa cả hai qua model gọi hàm TRIỂN KHAI (v6.1 q4 + ngưỡng 0,80) rồi chọn bản model tự tin hơn.

  V1: p token đầu cao hơn (bằng -> PhoWhisper)
  V2: như V1, nhưng bản RA LỆNH (gọi hàm / hỏi lại) có p ≥ 0,80 thắng bản TỪ CHỐI

    conda run -n vdk1 python -m eval.stt.ghep_stt <pho_b.jsonl> <zip_b.jsonl> <pho_cd.jsonl> <zip_cd.jsonl>
"""
from __future__ import annotations

import json
import math
import sys

from eval.metrics import DUNG, cham_mot_cau
from schema.loader import CATALOG
from serve.so_chu import chuan_cau_stt

GGUF = "models/gguf/lora-v61-s42-q4_k_m.gguf"
TU_CHOI = CATALOG.by_name["irrelevant_function"].nexa_token


def doc(p: str) -> dict[str, dict]:
    return {r["id"]: r for r in (json.loads(x) for x in open(p, encoding="utf-8-sig") if x.strip())}


def mcnemar(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def chon(a: dict, b: dict, bien_the: str) -> dict:
    """a = PhoWhisper, b = Zipformer: mỗi bản {out, p, tu_choi}."""
    if bien_the == "V2":
        lenh_a = not a["tu_choi"] and a["p"] >= 0.80
        lenh_b = not b["tu_choi"] and b["p"] >= 0.80
        if lenh_a != lenh_b:
            return a if lenh_a else b
    return b if b["p"] > a["p"] else a


def chay(m, pho: dict, zip_: dict) -> dict[str, list]:
    kq: dict[str, list] = {"pho": [], "zip": [], "V1": [], "V2": [], "tot_nhat": []}
    for k, r in pho.items():
        ban = {}
        for ten, row in (("pho", r), ("zip", zip_[k])):
            out = m.goi_ham(chuan_cau_stt(row["chu_stt"]))  # y như eval.stt.cham_chuoi
            ban[ten] = {"out": out, "p": m.lan_cuoi.p, "tu_choi": m.lan_cuoi.top[0][0] == TU_CHOI,
                        "dung": cham_mot_cau(out, r["completion"])[0] == DUNG}
        kq["pho"].append(ban["pho"]["dung"])
        kq["zip"].append(ban["zip"]["dung"])
        for v in ("V1", "V2"):
            kq[v].append(chon(ban["pho"], ban["zip"], v)["dung"])
        kq["tot_nhat"].append(ban["pho"]["dung"] or ban["zip"]["dung"])
    return kq


def bao(ten: str, kq: dict[str, list]) -> None:
    n = len(kq["pho"])
    print(f"\n== {ten} ({n} câu)")
    for v in ("pho", "zip", "V1", "V2", "tot_nhat"):
        dong = f"  {v:9} đúng {sum(kq[v]):2d}/{n}"
        if v in ("zip", "V1", "V2"):
            x = sum(i and not j for i, j in zip(kq[v], kq["pho"], strict=True))
            y = sum(j and not i for i, j in zip(kq[v], kq["pho"], strict=True))
            dong += f"   so PhoWhisper: hơn {x}, kém {y}, p {mcnemar(x, y):.3f}"
        print(dong)


def main(pb: str, zb: str, pcd: str, zcd: str) -> None:
    from serve.mo_hinh_gguf import MayChuLlama, MoHinhGGUF

    with MayChuLlama(GGUF, port=8091) as may:
        m = MoHinhGGUF(url=may.url)
        bao("BỘ B — chọn V1 / V2", chay(m, doc(pb), doc(zb)))
        bao("BỘ C + D — xác nhận (lần dùng lại thứ hai)", chay(m, doc(pcd), doc(zcd)))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(*sys.argv[1:5])
