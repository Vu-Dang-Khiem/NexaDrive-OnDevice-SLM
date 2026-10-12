"""Tập đánh giá GRPO — ĐÓNG BĂNG (PLAN.md Phase 7 — GRPO): nhãn ChatGPT (`2_da_gan_nhan_1_200.csv`) + sửa sau duyệt
(`3_bang_duyet.csv`: cột SAI / sua_thanh) -> jsonl, kiểm nhiễm với dữ liệu train / val v6.1, mã băm.

    python -m eval.dong_bang_test_grpo

Ra trong data/processed/final/:
  test_grpo.jsonl              200 câu (mọi phần; câu hai lượt prompt định dạng v6 "Lượt trước: …"); meta.phan_de
                               R / M / N / H, meta.kieu_de, meta.nhiem (trùng / gần trùng câu train, val -> loại khỏi
                               số chính thức, như mọi tập trước)
  test_grpo_chu_thuong.jsonl   cùng câu, chữ đầu câu người dùng viết thường (đo độ bền hình thức — model nhạy chữ hoa)
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from eval.chuyen_test import TIEN_TO_HOI_LAI, doc_nhan, dong_thanh_row
from eval.kiem_nhiem import tim_nhiem
from eval.kiem_test_v6 import doc_csv
from eval.metrics import doc_output
from schema.call_format import render_call
from schema.loader import CATALOG
from train.grpo import ha_chu_dau

THU_MUC = Path("data/test_grpo")
RA = Path("data/processed/final")
NGUON_TRAIN = [Path("data/processed/final_v6_1/train_vi.jsonl"), Path("data/processed/final_v6_1/val_vi.jsonl")]


def bo_ask_chi_doc(nhan: str) -> str:
    """Quy ước đã chốt từ dữ liệu train (`eval/chuyen_test.dong_thanh_row`): hàm CHỈ ĐỌC (xăng, lốp, thời tiết…)
    thì LÀM LUÔN, không hỏi lại — xem thông tin không đụng gì tới xe. ChatGPT ghi `ask:` cho 5 câu bóng gió kiểu
    "chẳng biết còn được bao nhiêu xăng" -> bỏ tiền tố, giữ hàm + tham số (sửa TRƯỚC khi có số chấm nào)."""
    calls, hoi = doc_nhan(nhan)
    if hoi and all(CATALOG.by_name[t].chi_doc for t, _ in calls):
        return nhan.strip()[len(TIEN_TO_HOI_LAI):].strip()
    return nhan


def dung(nhan: list[dict[str, str]], duyet: list[dict[str, str]], khoa: dict[str, dict[str, str]]
         ) -> list[dict[str, Any]]:
    theo_id = {r["id"]: dict(r) for r in nhan}
    for d in duyet:  # người duyệt ghi nhãn đúng vào sua_thanh (để trống = giữ nhãn ChatGPT)
        if d.get("sua_thanh", "").strip():
            theo_id[d["id"]]["nhan"] = d["sua_thanh"].strip()
    for i, r in theo_id.items():
        if (moi := bo_ask_chi_doc(r["nhan"])) != r["nhan"]:
            print(f"  quy ước chỉ đọc (id {i}): {r['nhan']}  ->  {moi}")
            r["nhan"] = moi
    rows = []
    for i, r in sorted(theo_id.items(), key=lambda x: int(x[0])):
        k = khoa[i]
        row = dong_thanh_row(r["cau"], r["nhan"], nhom_phu=f"grpo_{k['phan']}")
        row["meta"].update(nguon="chatgpt_nguoi_duyet", id_grpo=int(i), phan_de=k["phan"], ham_de=k["ham_dich"],
                           kieu_de=k["kieu"], cau=r["cau"])
        if k["phan"] == "H":
            r1 = dong_thanh_row(r["luot_1"], r["nhan_luot_1"])
            ten1, args1 = r1["meta"]["fn"], dict(doc_output(r1["completion"]).calls[0][1])
            goi_1 = render_call(ten1, args1)
            row["prompt"] = CATALOG.render_call_prompt(r["cau"], goi_1)
            row["meta"].update(cau_1=r["luot_1"], fn_1=ten1, args_1=args1, goi_1=goi_1)
        rows.append(row)
    return rows


def danh_dau_nhiem(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Kiểm CÂU người dùng (lượt 2 với câu hai lượt) với mọi câu train / val v6.1: trùng y hệt hoặc gần trùng."""
    nguon = [json.loads(x) for p in NGUON_TRAIN for x in p.open(encoding="utf-8")]
    don = [{"prompt": CATALOG.render_call_prompt(r["meta"]["cau"]), "meta": r["meta"]} for r in rows]
    kq = tim_nhiem(don, nguon)
    for i in kq["chi_so"]:
        rows[i]["meta"]["nhiem"] = kq["kieu"][i]
    return kq


def ghi(path: Path, rows: list[dict[str, Any]]) -> str:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    nhan = doc_csv(THU_MUC / "2_da_gan_nhan_1_200.csv")
    with open(THU_MUC / "3_bang_duyet.csv", encoding="utf-8-sig", newline="") as f:
        duyet = list(csv.DictReader(f))
    khoa = {r["id"]: r for r in doc_csv(THU_MUC / "khoa_id.csv")}
    rows = dung(nhan, duyet, khoa)
    kq = danh_dau_nhiem(rows)
    thuong = []
    for r in rows:
        t = json.loads(json.dumps(r, ensure_ascii=False))
        t["prompt"] = ha_chu_dau(r["prompt"])
        t["meta"]["chu_thuong"] = True
        thuong.append(t)
    h1 = ghi(RA / "test_grpo.jsonl", rows)
    h2 = ghi(RA / "test_grpo_chu_thuong.jsonl", thuong)
    print(f"{len(rows)} câu; theo phần: {dict(Counter(r['meta']['phan_de'] for r in rows))}; "
          f"loại nhãn: {dict(Counter(r['meta']['type'] for r in rows))}")
    print(f"nhiễm (trùng / gần trùng train + val v6.1): {kq['so_bi_nhiem']} ({kq['ty_le_%']}%) "
          f"theo phần {dict(Counter(rows[i]['meta']['phan_de'] for i in kq['chi_so']))}")
    for v in kq["vi_du"]:
        print("   ", v)
    print(f"sha256 test_grpo.jsonl            {h1}\nsha256 test_grpo_chu_thuong.jsonl {h2}")


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
