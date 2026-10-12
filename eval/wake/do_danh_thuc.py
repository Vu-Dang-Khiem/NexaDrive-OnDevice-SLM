"""Wake-word — ĐO theo tiêu chí khoá (PLAN.md Phase 6b), dùng ngưỡng ĐÃ khoá trong
`models/wake/hey_nexa.joblib`. Chạy trong env `giong_noi`:

    python -m eval.wake.do_danh_thuc --phan test                                # (b) FLEURS · (b2) bộ C · (c)
    python -m eval.wake.do_danh_thuc --phan nguoi --bo bo_wake_2 --nguoi khiem  # (a) 40 lần "hey nexa"

Báo nhầm đo như lúc chạy thật: MỘT dòng âm thanh liên tục (các câu nối nhau, cách 0,3–1,0 s im), báo ->
nghỉ 2 s. Kết quả ghi `runs/wake/*_ban{bản model}.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from serve.danh_thuc import KHUNG, DacTrung, dem_bao, diem, nap_mo_hinh
from train.danh_thuc import BO_DAU, SR, doc_fleurs, doc_wav, sang_int16

RUNS = Path("runs/wake")


def diem_dong(dt: DacTrung, mo_hinh, doan: list[np.ndarray], seed: int) -> tuple[np.ndarray, float, float]:
    """Điểm từng khung trên dòng nối các đoạn; trả (điểm, số giờ, ms trung bình / khung)."""
    rng = np.random.default_rng(seed)
    dt.lam_moi()
    ra, n, t = [], 0, 0.0
    for a in doan:
        x = sang_int16(np.concatenate([a, np.zeros(int(rng.uniform(0.3, 1.0) * SR), np.float32)]))
        for i in range(len(x) // KHUNG):
            t0 = time.perf_counter()
            ra.append(diem(mo_hinh, dt.them(x[i * KHUNG : (i + 1) * KHUNG]))[0])
            t += time.perf_counter() - t0
            n += 1
    return np.array(ra[BO_DAU:]), n * KHUNG / SR / 3600, 1000 * t / max(n, 1)


def ghi(ten: str, kq: dict, ban: int) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{ten}_ban{ban}.json").write_text(json.dumps(kq, ensure_ascii=False, indent=1), encoding="utf-8")


def phan_test() -> None:
    goi = nap_mo_hinh()
    mo_hinh, nguong, lien = goi["mo_hinh"], goi["nguong"], goi.get("lien", 1)
    ban = goi["thong_tin"].get("ban", 1)
    dt = DacTrung()
    fl = doc_fleurs(Path("data/wake/fleurs/test.tar.gz"))
    d, gio, ms = diem_dong(dt, mo_hinh, [fl[k] for k in sorted(fl)], seed=1)
    bao = dem_bao(d, nguong, lien)
    man = [json.loads(x) for x in open("data/wake/tts/manifest.jsonl", encoding="utf-8")]
    tu = [doc_wav(Path("data/wake/tts") / m["tep"]) for m in man if m["loai"] == "tu_thuc"]
    d2, gio2, _ = diem_dong(dt, mo_hinh, tu, seed=2)
    bao2 = dem_bao(d2, nguong, lien)
    bo_c = [doc_wav(t) for t in sorted(Path("data/stt/bo_c/khiem").glob("*.wav"))]
    d3, gio3, _ = diem_dong(dt, mo_hinh, bo_c, seed=3)
    bao3 = dem_bao(d3, nguong, lien)
    kq = {
        "ban": ban,
        "nguong": nguong,
        "lien": lien,
        "b_fleurs_test": {
            "cau": len(fl),
            "gio": round(gio, 3),
            "bao_nham": len(bao),
            "bao_nham_gio": len(bao) / gio,
            "dat": len(bao) / gio <= 1.0,
        },
        "b2_bo_c_giong_nguoi_dung": {
            "cau": len(bo_c),
            "phut": round(gio3 * 60, 2),
            "bao": len(bao3),
            "dat": len(bao3) == 0,
        },
        "c_tu_thuc": {"cau": len(tu), "phut": round(gio2 * 60, 2), "bao": len(bao2), "dat": len(bao2) == 0},
        "ms_moi_khung_80ms": round(ms, 2),
        "phan_tram_1_nhan_cpu": round(100 * ms / 80, 1),
    }
    ghi("test", kq, ban)
    print(json.dumps(kq, ensure_ascii=False, indent=1))


def phan_nguoi(bo: str, nguoi: str) -> None:
    goi = nap_mo_hinh()
    mo_hinh, nguong, lien = goi["mo_hinh"], goi["nguong"], goi.get("lien", 1)
    ban = goi["thong_tin"].get("ban", 1)
    dt = DacTrung()
    thu_muc = Path("data/stt") / bo / nguoi
    rows = [json.loads(x) for x in open(Path("data/stt") / bo / "danh_sach.jsonl", encoding="utf-8")]
    ket = []
    for r in rows:
        tep = thu_muc / r["tep"]
        if not tep.exists():
            continue
        x = np.concatenate(
            [np.zeros(int(1.2 * SR), np.float32), doc_wav(tep), np.zeros(int(0.5 * SR), np.float32)]
        )
        d = diem(mo_hinh, dt.chuoi(sang_int16(x))[BO_DAU:])
        ket.append(
            {
                "tep": r["tep"],
                "cau": r["cau"],
                "diem_max": round(float(d.max()), 3),
                "bat": bool(dem_bao(d, nguong, lien)),
            }
        )
    bat = sum(k["bat"] for k in ket)
    kq = {
        "ban": ban,
        "bo": bo,
        "nguong": nguong,
        "lien": lien,
        "so_ban_ghi": len(ket),
        "bat": bat,
        "ty_le": bat / max(len(ket), 1),
        "dat": len(ket) == 40 and bat / 40 >= 0.9,
        "chi_tiet": ket,
    }
    ghi(f"nguoi_{bo}_{nguoi}", kq, ban)
    for k in ket:
        print(f"  {'✓' if k['bat'] else '✗'} {k['tep']} {k['diem_max']:.3f}  {k['cau']}")
    ket_luan = "ĐẠT" if kq["dat"] else "TRƯỢT"
    print(f"(a) {bo}: bắt được {bat}/{len(ket)} ({kq['ty_le']:.1%}) — {ket_luan} (cần ≥ 90% trên 40)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phan", choices=["test", "nguoi"], required=True)
    ap.add_argument("--bo", default="bo_wake_2", help="bộ ghi để đo (a): bo_wake (bản 1) / bo_wake_2 (bản 2)")
    ap.add_argument("--nguoi", default="khiem")
    a = ap.parse_args()
    phan_test() if a.phan == "test" else phan_nguoi(a.bo, a.nguoi)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
