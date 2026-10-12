"""Hai bộ ghi âm MỚI cho fine-tune STT (PLAN.md "FINE-TUNE STT", 2026-10-12) — bốc xong là khoá.

• Bộ E (KIỂM TRA CUỐI, 40 câu): như bộ B / C / D — tập test đóng băng (test sạch + test v6 một lượt),
  KHÔNG trùng B / C / D, xáo seed 2030; tỉ lệ bộ B: 20 rõ ràng (tối đa 10 có số) · 10 mơ hồ · 10 ngoài pv.
• Bộ FT (giọng thật để TRAIN STT, 150 câu): từ dữ liệu train của model gọi hàm (`final_v6_1/train_vi`,
  KHÔNG phải test), bỏ câu nhiễu (bỏ dấu / lỗi gõ), hai lượt, "lượt trước thừa"; xáo seed 2031: 24 chức
  năng × 5 câu (ưu tiên 2 câu có số) + 15 hỏi lại + 15 từ chối. Kiểm nhiễm: không trùng / gần trùng câu
  nào của mọi tập test đóng băng và bộ B / C / D / E.

  conda run -n vdk1 python -m eval.stt.chon_bo_e_ft
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from eval.kiem_nhiem import tim_nhiem
from eval.metrics import doc_output, nhom_cua_dap_an
from eval.stt.chon_bo_c import NGUON, co_so
from gen.records import query_from_call_prompt

STT = Path("data/stt")
DA_CO = [STT / f"bo_{b}/danh_sach.jsonl" for b in "bcd"]
RA_E, RA_FT = STT / "bo_e/danh_sach.jsonl", STT / "bo_ft/danh_sach.jsonl"
TEST = sorted(Path("data/processed/final").glob("test_*.jsonl")) + [
    Path("data/processed/final/hieu_chinh_v6.jsonl")
]


def doc(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.open(encoding="utf-8") if x.strip()]


def ghi(ra: Path, tien_to: str, chon: list[dict]) -> None:
    ra.parent.mkdir(parents=True, exist_ok=True)
    with ra.open("w", encoding="utf-8") as f:
        for i, r in enumerate(chon):
            f.write(
                json.dumps(
                    {
                        "id": f"{tien_to}{i:03d}",
                        "tep": f"{tien_to}{i:03d}.wav",
                        "cau": r["cau"],
                        "nhom": r["nhom"],
                        "nguon": r["nguon"],
                        "co_so": co_so(r["cau"]),
                        "prompt": r["prompt"],
                        "completion": r["completion"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def bo_e() -> list[dict]:
    da_dung = {json.loads(x)["cau"] for p in DA_CO for x in p.open(encoding="utf-8")}
    rows = []
    for ten, p in NGUON.items():
        for r in doc(p):
            r["cau"] = query_from_call_prompt(r["prompt"])
            if r["cau"] not in da_dung:
                r["nguon"], r["nhom"] = ten, nhom_cua_dap_an(doc_output(r["completion"]))
                rows.append(r)
    random.Random(2030).shuffle(rows)
    rr = [r for r in rows if r["nhom"] == "ro_rang"]
    so = [r for r in rr if co_so(r["cau"])][:10]
    khong_so = [r for r in rr if not co_so(r["cau"])][: 20 - len(so)]
    return (
        so
        + khong_so
        + [r for r in rows if r["nhom"] == "mo_ho"][:10]
        + [r for r in rows if r["nhom"] == "ngoai_pham_vi"][:10]
    )


def bo_ft(cam: list[dict]) -> list[dict]:
    rows = []
    for r in doc(Path("data/processed/final_v6_1/train_vi.jsonl")):
        m = r["meta"]
        if (
            m["type"] not in ("call", "ask")
            or "fn_1" in m
            or str(m.get("v6", "")).startswith(("nhieu", "luot"))
        ):
            continue
        r["cau"] = query_from_call_prompt(r["prompt"])
        if "\n" in r["cau"]:
            continue
        r["nguon"], r["nhom"] = "train_v61", nhom_cua_dap_an(doc_output(r["completion"]))
        rows.append(r)
    nhiem = set(tim_nhiem(rows, cam)["chi_so"])  # bỏ câu trùng / gần trùng mọi tập test + bộ ghi âm
    rows = [r for i, r in enumerate(rows) if i not in nhiem]
    random.Random(2031).shuffle(rows)
    theo_ham: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["nhom"] == "ro_rang":
            theo_ham[r["meta"]["fn"]].append(r)
    chon = []
    for ham in sorted(theo_ham):
        ds = theo_ham[ham]
        so = [r for r in ds if co_so(r["cau"])][:2]
        chon += so + [r for r in ds if r not in so][: 5 - len(so)]
    chon += [r for r in rows if r["nhom"] == "mo_ho"][:15]
    chon += [r for r in rows if r["nhom"] == "ngoai_pham_vi"][:15]
    return chon


def main() -> None:
    if RA_E.exists() or RA_FT.exists():
        raise SystemExit("bộ E / FT đã có — danh sách đã khoá, không bốc lại.")
    e = bo_e()
    ghi(RA_E, "e", e)
    cam = [r for p in TEST for r in doc(p)] + [r for p in DA_CO + [RA_E] for r in doc(p)]
    ft = bo_ft(cam)
    ghi(RA_FT, "f", ft)
    for ten, ds, ra in (("Bộ E", e, RA_E), ("Bộ FT", ft, RA_FT)):
        print(f"{ten}: {len(ds)} câu -> {ra}")
        for k in ("ro_rang", "mo_ho", "ngoai_pham_vi"):
            nhom = [r for r in ds if r["nhom"] == k]
            print(f"  {k}: {len(nhom)}  (có số: {sum(co_so(r['cau']) for r in nhom)})")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
