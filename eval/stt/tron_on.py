"""Độ chịu ồn — bước 1 (PLAN.md "ĐỘ CHỊU ỒN — KẾ HOẠCH + LUẬT KHOÁ TRƯỚC"): trộn ồn vào 60 bản ghi giọng
thật bộ C + D ở SNR 20 / 10 / 5 / 0 dB, 3 loại ồn, rồi chạy STT (PhoWhisper-small G0, CÙNG cách gọi lúc đo)
cho mọi bộ. Env `giong_noi`:

    python -m eval.stt.tron_on        # -> runs/stt/on/<bộ>.jsonl (định dạng như eval/stt/chay_stt.py)
    conda run -n vdk1 python -m eval.stt.cham_chuoi runs/stt/on/sach.jsonl runs/stt/on/tcar_20db.jsonl ... \
        --gguf=models/gguf/lora-v61-s42-q4_k_m.gguf

Ồn: `tcar` = DEMAND TCAR (trong xe thật, Zenodo 1227121, CC BY 4.0, kênh 1) · `tram` = ồn 1/f² tổng hợp ·
`nguoi` = tiếng người nói nền (FLEURS vi dev). SNR tính trên phần CÓ TIẾNG NÓI của bản ghi; đoạn ồn chọn
ngẫu nhiên, seed cố định. Âm thanh trộn ghi ra `data/stt/on/<bộ>/` để nghe lại được.
"""

from __future__ import annotations

import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

SR = 16000
SEED = 2026_10_09
SNR = (20, 10, 5, 0)
TCAR = Path("data/stt/on_nen/demand/TCAR_16k.zip")
RA_AM = Path("data/stt/on")
RA = Path("runs/stt/on")


def doc_tcar() -> np.ndarray:
    import soundfile as sf

    with zipfile.ZipFile(TCAR) as z:
        ten = next(n for n in z.namelist() if n.lower().endswith("ch01.wav"))
        a, sr = sf.read(io.BytesIO(z.read(ten)), dtype="float32")
    assert sr == SR, f"TCAR {sr} Hz"
    return a


def tron(sach: np.ndarray, on: np.ndarray, snr: float, rng: np.random.Generator) -> np.ndarray:
    import librosa

    tieng, _ = librosa.effects.trim(sach, top_db=35)
    p_tieng = np.mean(tieng**2) + 1e-12
    if len(on) < len(sach):
        on = np.tile(on, len(sach) // len(on) + 1)
    s = rng.integers(0, len(on) - len(sach) + 1)
    doan = on[s : s + len(sach)]
    doan = doan * np.sqrt(p_tieng / (np.mean(doan**2) + 1e-12) / 10 ** (snr / 10))
    x = sach + doan
    return (x / max(1.0, float(np.abs(x).max()) / 0.99)).astype(np.float32)  # chỉ hạ khi sắp vỡ tiếng


def main() -> None:
    import soundfile as sf

    from eval.stt.chay_stt import nap
    from serve.so_chu import doi_so
    from train.danh_thuc import FLEURS_DEV, doc_fleurs, on

    rng = np.random.default_rng(SEED)
    rows = [
        json.loads(x)
        for b in ("bo_c", "bo_d")
        for x in open(f"data/stt/{b}/khiem/manifest.jsonl", encoding="utf-8")
    ]
    sach = {
        r["id"]: sf.read(
            f"data/stt/{r['id'][0] == 'c' and 'bo_c' or 'bo_d'}/khiem/{r['tep']}", dtype="float32"
        )[0]
        for r in rows
    }
    nguon_on = {
        "tcar": doc_tcar(),
        "tram": on("tram", 60 * SR, rng, []),
        "nguoi": np.concatenate([a for _, a in sorted(doc_fleurs(FLEURS_DEV).items())][:60]),
    }
    bo = {"sach": None} | {f"{loai}_{snr}db": (loai, snr) for loai in nguon_on for snr in SNR}
    nghe = nap("phowhisper-small")
    RA.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for ten_bo, cau_hinh in bo.items():
        thu_muc = RA_AM / ten_bo
        thu_muc.mkdir(parents=True, exist_ok=True)
        with open(RA / f"{ten_bo}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                a = (
                    sach[r["id"]]
                    if cau_hinh is None
                    else tron(sach[r["id"]], nguon_on[cau_hinh[0]], cau_hinh[1], rng)
                )
                tep = thu_muc / f"{r['id']}.wav"
                sf.write(str(tep), a, SR, subtype="PCM_16")
                t = time.perf_counter()
                chu = nghe(str(tep))
                ms = (time.perf_counter() - t) * 1000
                f.write(
                    json.dumps(
                        {
                            **{k: r[k] for k in ("id", "cau", "prompt", "completion")},
                            "stt": "phowhisper-small",
                            "chu_stt": chu,
                            "chu_doi_so": doi_so(chu),
                            "ms": round(ms, 1),
                            "giay_am_thanh": round(len(a) / SR, 2),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"  {ten_bo} xong ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
