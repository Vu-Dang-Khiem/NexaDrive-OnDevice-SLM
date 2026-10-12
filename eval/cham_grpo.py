"""Chấm GRPO vs v6.1 theo TIÊU CHÍ KHOÁ TRƯỚC (PLAN.md "GRPO — KẾ HOẠCH + TIÊU CHÍ KHOÁ TRƯỚC"). Model HF đã
gộp, KHÔNG ngưỡng, luật chấm C; dự đoán sinh trên Modal (`train/modal_train.py::sinh_du_doan`, greedy).

    python -m eval.cham_grpo --buoc val --hat s42 --tai    # ĐIỂM DỪNG 1: chỉ val mẫu, một hạt giống
    python -m eval.cham_grpo --buoc test --tai             # KẾT LUẬN: test_grpo (+ bản chữ thường), tb 2 hạt giống

Kết cục theo NGHĨA điểm thưởng (`train/grpo.loai_ket_cuc`): đúng · hỏi lại thừa · từ chối nhầm · đề xuất sai ·
làm sai lên xe · hỏng. Câu `meta.nhiem` (gần trùng train) bị loại khỏi số chính thức.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from train.grpo import loai_ket_cuc

TEST = Path("data/processed/final")
RUNS = Path("runs/grpo")
VOL_DIR = "/data/vi/final_v6_1"
HAT = ("s42", "s7")
BAN = {"v61": "v6.1", "grpo": "GRPO"}


def doc(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]


def tai(ten: list[str]) -> None:
    """Tải dự đoán từ volume, kiểm ĐỦ kích thước (đường tải Modal -> máy hay cắt cụt âm thầm — PLAN.md)."""
    import modal

    vol = modal.Volume.from_name("nexa-drive")
    RUNS.mkdir(parents=True, exist_ok=True)
    kt = {e.path.rsplit("/", 1)[-1]: e.size for e in vol.listdir(VOL_DIR)}
    for t in ten:
        if (RUNS / t).exists() and (RUNS / t).stat().st_size == kt[t]:
            continue
        for _ in range(6):
            du = b"".join(vol.read_file(f"{VOL_DIR}/{t}"))
            if len(du) == kt[t]:
                break
        else:
            raise RuntimeError(f"tải {t} không đủ sau 6 lần")
        (RUNS / t).write_bytes(du)


def ket_cuc(tap_file: Path, du_doan: Path) -> list[tuple[dict[str, Any], str]]:
    # ghép THEO THỨ TỰ (sinh_du_doan giữ thứ tự đầu vào) — val mẫu có prompt trùng nên không ghép theo prompt
    dap_an, du = doc(tap_file), doc(du_doan)
    assert len(du) == len(dap_an), f"{du_doan.name}: {len(du)} dự đoán / {len(dap_an)} câu"
    assert all(d["prompt"] == r["prompt"] for d, r in zip(du, dap_an, strict=True)), f"{du_doan.name}: lệch thứ tự"
    return [(r, loai_ket_cuc(d["output"], r["completion"])) for d, r in zip(du, dap_an, strict=True)]


def ty(kq: list[str], loai: str) -> float:
    return 100 * sum(k == loai for k in kq) / len(kq) if kq else float("nan")


def bang(ten: str, theo_ban: dict[str, list[tuple[dict[str, Any], str]]], nhom_cua) -> None:
    print(f"\n== {ten}")
    nhom = sorted({nhom_cua(r) for kq in theo_ban.values() for r, _ in kq})
    print(f"{'':10}" + "".join(f"{n:>14}" for n in nhom) + f"{'làm sai lên xe':>17}{'từ chối nhầm':>14}{'hỏi thừa':>10}")
    for b, kq in theo_ban.items():
        o = [f"{ty([k for r, k in kq if nhom_cua(r) == n], 'dung'):>13.1f}%" for n in nhom]
        tat = [k for _, k in kq]
        print(f"{b:10}" + "".join(o) + f"{ty(tat, 'lam_sai_len_xe'):>16.1f}%{ty(tat, 'tu_choi_nham'):>13.1f}%"
              f"{ty(tat, 'hoi_lai_thua'):>9.1f}%")
        print(f"{'':10}kết cục: {dict(Counter(tat))}")


def buoc_val(hat: str, co_tai: bool) -> None:
    ten = {f"v61{hat}": f"du_doan_v61{hat}ck4192_val_mau.jsonl", f"grpo{hat}": f"du_doan_grpo{hat}_val_mau.jsonl"}
    if co_tai:
        tai(list(ten.values()))
    from eval.metrics import doc_output, nhom_cua_dap_an

    tap = RUNS / "val_mau.jsonl"
    if not tap.exists():
        tai(["val_mau.jsonl"])
    theo_ban = {b: ket_cuc(tap, RUNS / f) for b, f in ten.items()}
    # bỏ mẫu lượt 2 (reply) — GRPO chỉ luyện lượt 1
    theo_ban = {b: [(r, k) for r, k in kq if r["completion"].lstrip().startswith("<nexa")] for b, kq in theo_ban.items()}
    bang(f"ĐIỂM DỪNG 1 — val mẫu (lượt 1), hạt giống {hat}", theo_ban,
         lambda r: nhom_cua_dap_an(doc_output(r["completion"])))


def buoc_test(co_tai: bool) -> None:
    tap = ("test_grpo", "test_grpo_chu_thuong")
    ten = [f"du_doan_{b}{h}_{t}.jsonl" for b in BAN for h in HAT for t in tap]
    if co_tai:
        tai(ten)
    for t in tap:
        theo_ban: dict[str, list[tuple[dict[str, Any], str]]] = {}
        for b in BAN:
            for h in HAT:
                kq = ket_cuc(TEST / f"{t}.jsonl", RUNS / f"du_doan_{b}{h}_{t}.jsonl")
                theo_ban[f"{b}{h}"] = [(r, k) for r, k in kq if not r["meta"].get("nhiem")]
        bang(f"{t} (bỏ câu nhiễm) — từng hạt giống", theo_ban, lambda r: r["meta"]["phan_de"])
    tieu_chi(tap)


def tieu_chi(tap: tuple[str, ...]) -> None:
    """5 tiêu chí khoá trước, trung bình 2 hạt giống, gộp bản gốc + bản chữ thường."""
    def gop(b: str) -> dict[str, float]:
        s: dict[str, list[float]] = {}
        for h in HAT:
            kq = [(r, k) for t in tap for r, k in ket_cuc(TEST / f"{t}.jsonl", RUNS / f"du_doan_{b}{h}_{t}.jsonl")
                  if not r["meta"].get("nhiem")]
            tat = [k for _, k in kq]
            m = [k for r, k in kq if r["meta"]["phan_de"] == "M"]
            rr = [k for r, k in kq if r["meta"]["phan_de"] == "R"]
            for ten, v in (("lam_sai", ty(tat, "lam_sai_len_xe")), ("M", ty(m, "dung")), ("R", ty(rr, "dung")),
                           ("tu_choi_nham", ty(tat, "tu_choi_nham")), ("hoi_thua_R", ty(rr, "hoi_lai_thua"))):
                s.setdefault(ten, []).append(v)
        return {k: sum(v) / len(v) for k, v in s.items()}

    a, g = gop("v61"), gop("grpo")
    ds = [
        ("(1) làm sai lên xe GIẢM", g["lam_sai"] < a["lam_sai"], "lam_sai"),
        ("(2) M không giảm quá 2 điểm", g["M"] >= a["M"] - 2, "M"),
        ("(3) R không giảm quá 2 điểm", g["R"] >= a["R"] - 2, "R"),
        ("(4) từ chối nhầm không tăng quá 1 điểm", g["tu_choi_nham"] <= a["tu_choi_nham"] + 1, "tu_choi_nham"),
        ("(5) hỏi thừa ở R không tăng quá 3 điểm", g["hoi_thua_R"] <= a["hoi_thua_R"] + 3, "hoi_thua_R"),
    ]
    print("\n== TIÊU CHÍ (tb 2 hạt giống, test_grpo + bản chữ thường, bỏ câu nhiễm)")
    for mo_ta, dat, k in ds:
        print(f"  {'ĐẠT ' if dat else 'TRƯỢT'} {mo_ta}: v6.1 {a[k]:.1f} -> GRPO {g[k]:.1f}")
    print("  => GRPO", "ĐẠT" if all(d for _, d, _ in ds) else "KHÔNG ĐẠT")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buoc", choices=["val", "test"], required=True)
    ap.add_argument("--hat", default="s42")
    ap.add_argument("--tai", action="store_true")
    a = ap.parse_args()
    if a.buoc == "val":
        buoc_val(a.hat, a.tai)
    else:
        buoc_test(a.tai)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
