"""Chấm baseline CHẠY BÙ (PLAN.md "CHẠY BÙ BASELINE CHO CÁC Ô —", khoá 2026-10-12): 72B + mô tả hàm (SIP) và 1,5B gốc + SIP
trên các tập chưa có số; so cặp McNemar với v6.1 (HF, không ngưỡng, từng hạt giống). Chỉ báo cáo.

    python -m eval.bao_cao.cham_bl_bu --tai    # tải dự đoán từ volume (kiểm kích thước, tải lại tới khi đủ) rồi chấm
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, ".")
from eval.metrics import DUNG, cham_mot_cau  # noqa: E402

TEST = Path("data/processed/final")
RUNS = Path("runs/v6")
VOL = "/data/vi/final_v6"
BL = {"sip_72b": ("test_v6_mot_luot", "test_v6_hai_luot", "test_v6_nhieu"),
      "sip_1_5b": ("test_vi_mo_ho_4", "test_v6_mot_luot", "test_v6_hai_luot", "test_v6_nhieu")}
NHOM = [  # (tên, tập, lọc theo nhom_phu)
    ("Mơ hồ dài (129)", "test_vi_mo_ho_4", None),
    ("Câu ngắn + cách gọi (120)", "test_v6_mot_luot", ("v6_A", "v6_B")),
    ("Tham số mới (30)", "test_v6_mot_luot", ("v6_C",)),
    ("Hai lượt (60)", "test_v6_hai_luot", None),
    ("Gõ sai (104)", "test_v6_nhieu", None),
]


def doc(p):
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    t = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (t - h), 100 * (t + h)


def tai():
    import modal

    vol = modal.Volume.from_name("nexa-drive")
    kt = {e.path.rsplit("/", 1)[-1]: e.size for e in vol.listdir(VOL)}
    for ten, taps in BL.items():
        for t in taps:
            f = f"du_doan_bl_{ten}_{t}.jsonl"
            for _ in range(6):  # đường tải Modal -> máy này hay cắt cụt âm thầm
                du = b"".join(vol.read_file(f"{VOL}/{f}"))
                if len(du) == kt[f]:
                    break
            else:
                raise RuntimeError(f"tải {f} không đủ")
            (RUNS / f).write_bytes(du)


def dung(ten_file, tap):
    dap = doc(TEST / f"{tap}.jsonl")
    du = {r["prompt"]: r["output"] for r in doc(RUNS / ten_file)}
    assert len(du) == len(dap), (ten_file, len(du), len(dap))
    return [cham_mot_cau(du[r["prompt"]], r["completion"])[0] == DUNG for r in dap], dap


def chon(ds, dap, loc):
    return [x for x, r in zip(ds, dap) if loc is None or r["meta"].get("nhom_phu") in loc]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tai", action="store_true")
    if ap.parse_args().tai:
        tai()
    for ten, taps in BL.items():
        print(f"\n===== {ten} (1 lần chạy, greedy) =====")
        for nhom, tap, loc in NHOM:
            if tap not in taps:
                continue
            b, dap = dung(f"du_doan_bl_{ten}_{tap}.jsonl", tap)
            b = chon(b, dap, loc)
            lo, hi = wilson(sum(b), len(b))
            dong = f"  {nhom:28} {100 * sum(b) / len(b):5.1f} [{lo:4.1f}–{hi:4.1f}]"
            for h in ("s42", "s7"):
                m, _ = dung(f"du_doan_v61{h}_{tap}.jsonl", tap)
                m = chon(m, dap, loc)
                x = sum(i and not j for i, j in zip(m, b))
                y = sum(j and not i for i, j in zip(m, b))
                dong += f" | v6.1 {h} {100 * sum(m) / len(m):5.1f} (hơn {x}, kém {y}, p {mcnemar(x, y):.3f})"
            print(dong)
        doc_khong = sum(1 for t in taps for r in doc(RUNS / f"du_doan_bl_{ten}_{t}.jsonl") if not r["output"])
        print(f"  JSON hỏng / không đọc được: {doc_khong}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
