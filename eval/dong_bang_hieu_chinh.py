"""Đóng băng TẬP HIỆU CHỈNH (cách 2): kiểm file người dùng duyệt, bỏ dòng đánh `bo`, bỏ câu trùng / gần trùng với MỌI tập
test và train (v5, v6.1), ghi jsonl nhãn từ chối.

    python -m eval.dong_bang_hieu_chinh   # -> data/processed/final/hieu_chinh_v6.jsonl
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from eval.kiem_nhiem import tim_nhiem
from eval.kiem_test_v6 import doc_csv
from gen.records import IM_END
from schema.call_format import render_completion
from schema.loader import CATALOG

THU_MUC = Path("data/hieu_chinh_v6")
P = Path("data/processed")
TEST = [P / "final" / f"{t}.jsonl" for t in ("test_vi", "test_vi_sach_v2", "test_vi_bo_sung", "test_vi_mo_ho_3",
                                             "test_vi_mo_ho_4", "test_v6_mot_luot", "test_v6_hai_luot", "test_v6_nhieu")]
TRAIN = [P / "final_v5" / "train_vi.jsonl", P / "final_v6_1" / "train_vi.jsonl"]


def doc(p: Path) -> list[dict]:
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def main() -> None:
    goc = {r["id"]: r for r in doc_csv(THU_MUC / "1_viet_cau.csv")}
    da = doc_csv(THU_MUC / "2_da_viet_200.csv")
    van_de = [f"id {r['id']}: cột loai/yeu_cau bị đổi" for r in da
              if r["id"] in goc and (r.get("loai") != goc[r["id"]]["loai"] or r.get("yeu_cau") != goc[r["id"]]["yeu_cau"])]
    bo = [r for r in da if "bo" in (r.get("ghi_chu") or "").lower().split() or not r.get("cau")]
    giu = [r for r in da if r not in bo]
    trung = Counter(" ".join(r["cau"].lower().split()) for r in giu)
    rows = []
    da_thay: set[str] = set()
    for r in giu:
        k = " ".join(r["cau"].lower().split())
        if k in da_thay:
            continue
        da_thay.add(k)
        rows.append({"prompt": CATALOG.render_call_prompt(r["cau"]),
                     "completion": render_completion([("irrelevant_function", {})]) + IM_END,
                     "meta": {"lang": "vi", "type": "call", "fn": "irrelevant_function", "nhom_phu": f"hc_{r['loai']}",
                              "id_hc": int(r["id"]), "nguon": "chatgpt_nguoi_duyet"}})
    # bỏ câu trùng / gần trùng với test (không được dùng câu test) và train (model đã thấy -> tự tin giả)
    for ten, nguon in (("test", TEST), ("train", TRAIN)):
        nhiem = tim_nhiem(rows, [x for p in nguon for x in doc(p)])
        bi = set(nhiem["chi_so"])
        print(f"trùng / gần trùng {ten}: {len(bi)}", [rows[i]["prompt"].split("user\n")[1][:50] for i in sorted(bi)][:8])
        rows = [r for i, r in enumerate(rows) if i not in bi]
    ra = P / "final" / "hieu_chinh_v6.jsonl"
    with open(ra, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(da)} dòng; người dùng bỏ {len(bo)}; trùng trong file {sum(n - 1 for n in trung.values() if n > 1)}; "
          f"vấn đề cột: {len(van_de)} -> GIỮ {len(rows)} câu -> {ra}")
    print("theo loại:", dict(Counter(r["meta"]["nhom_phu"] for r in rows)))
    for v in van_de[:5]:
        print("  ", v)


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
