"""FINE-TUNE PhoWhisper-small trên câu lệnh xe (PLAN.md "FINE-TUNE STT TRÊN CÂU LỆNH XE", 2026-10-12).

Dữ liệu trên volume `nexa-drive` (tải lên bằng `modal volume put`, xem PLAN.md):
  /data/stt_ft/tts/        manifest.jsonl + wav/   — ~5.000 câu lệnh, VieNeu 23 giọng, âm thanh SẠCH
  /data/stt_ft/that/       manifest.jsonl + *.wav  — bộ FT: 150 câu giọng thật người dùng (chỉ bản B)
  /data/stt_ft/fleurs/     manifest.jsonl + wav/   — FLEURS tiếng Việt (train), chống "quên" tiếng Việt thường
  /data/stt_ft/on/         *.wav                   — tiếng ồn xe DEMAND (TCAR) để làm bẩn
Bản A = TTS + FLEURS; bản B = A + giọng thật (lặp 4 lần cho đủ trọng số). Làm bẩn NGẪU NHIÊN mỗi lần đọc:
đổi tốc độ, âm lượng, trộn ồn xe, vang phòng, lọc kiểu mic laptop. KHÔNG làm sạch âm thanh (khử ồn đã
chứng minh hại).
Mỗi epoch lưu checkpoint + bản CTranslate2 int8 (giống bản đang chạy) để chọn trên bộ B ở máy.

    modal run train/modal_stt_ft.py::fine_tune --ban A --gioi-han-buoc 30     # chạy thử: đo giây / bước, VRAM
    modal run --detach train/modal_stt_ft.py::fine_tune --ban A
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import modal

VOL = "/vol"
GOC = "vinai/PhoWhisper-small"
app = modal.App("nexa-drive-stt-ft")
volume = modal.Volume.from_name("nexa-drive")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1")
    .pip_install(
        "torch==2.8.0",  # ≥ 2.6: PhoWhisper chỉ có pytorch_model.bin, transformers chặn torch.load bản cũ
        "transformers==4.57.3",
        "accelerate>=1.0",
        "soundfile",
        "soxr",
        "scipy",
        "numpy<2.3",
        "ctranslate2==4.6.0",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
SR = 16000


def _doc_manifest(thu_muc: str, lap: int = 1) -> list[tuple[str, str]]:
    d = Path(thu_muc)
    ds = [
        (str(d / r["tep"]), r["nhan"])
        for r in (json.loads(x) for x in (d / "manifest.jsonl").open(encoding="utf-8"))
    ]
    return ds * lap


class LamBan:
    """Làm bẩn âm thanh ngẫu nhiên (float32, 16 kHz)."""

    def __init__(self, thu_muc_on: str, seed: int) -> None:
        import soundfile as sf

        self.rng = random.Random(seed)
        self.on = [sf.read(str(p), dtype="float32")[0] for p in sorted(Path(thu_muc_on).glob("*.wav"))]

    def __call__(self, am):
        import numpy as np
        import soxr
        from scipy.signal import butter, fftconvolve, sosfilt

        r = self.rng
        if r.random() < 0.5:  # đổi tốc độ 0,9–1,1 (đổi cả cao độ — như cách "speed perturbation" thường dùng)
            k = r.choice([0.9, 0.95, 1.05, 1.1])
            am = soxr.resample(am, int(SR * k), SR).astype("float32")
        if r.random() < 0.3:  # vang phòng / khoang xe: đáp ứng xung tắt dần ngẫu nhiên
            dai = int(SR * r.uniform(0.08, 0.35))
            rir = np.random.default_rng(r.randrange(1 << 30)).standard_normal(dai).astype("float32")
            rir *= np.exp(-np.arange(dai) / (dai / r.uniform(3, 6))).astype("float32")
            rir[0] = 1.0
            am = fftconvolve(am, rir / np.abs(rir).sum() * 3)[: len(am)].astype("float32")  # FFT: nhanh
        if self.on and r.random() < 0.7:  # trộn ồn xe, SNR 0–20 dB
            on = self.on[r.randrange(len(self.on))]
            if len(on) > len(am):
                bd = r.randrange(len(on) - len(am))
                doan = on[bd : bd + len(am)]
                snr = r.uniform(0, 20)
                p_am, p_on = float((am**2).mean()) + 1e-9, float((doan**2).mean()) + 1e-9
                am = am + doan * np.sqrt(p_am / (p_on * 10 ** (snr / 10)))
        if r.random() < 0.4:  # mic laptop: cắt trầm < 150 Hz, cắt bổng > 5–7 kHz
            sos = butter(4, [150, r.uniform(5000, 7000)], btype="band", fs=SR, output="sos")
            am = sosfilt(sos, am).astype("float32")
        am = am * (10 ** (r.uniform(-12, 6) / 20))  # âm lượng
        dinh = float(np.abs(am).max()) + 1e-9
        return (am / dinh * 0.98 if dinh > 0.98 else am).astype("float32")


@app.function(image=image, gpu="A100-40GB", cpu=8, memory=32768, volumes={VOL: volume}, timeout=3 * 60 * 60)
def fine_tune(
    ban: str = "A",
    epochs: int = 3,
    lr: float = 1e-5,
    batch: int = 16,
    gioi_han_buoc: int = 0,
    so_fleurs: int = 1500,
    seed: int = 42,
) -> dict:
    import numpy as np
    import soundfile as sf
    import soxr
    import torch
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
        get_linear_schedule_with_warmup,
    )

    torch.manual_seed(seed)
    du = f"{VOL}/data/stt_ft"
    ds = _doc_manifest(f"{du}/tts")
    fl = _doc_manifest(f"{du}/fleurs")
    random.Random(seed).shuffle(fl)
    ds += fl[:so_fleurs]
    if ban == "B":
        ds += _doc_manifest(f"{du}/that", lap=4)
    print(f"bản {ban}: {len(ds)} mẫu", flush=True)

    proc = WhisperProcessor.from_pretrained(GOC, language="vietnamese", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(GOC).cuda()
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    lam_ban = LamBan(f"{du}/on", seed)

    def mau(duong: str, nhan: str):
        am, sr = sf.read(duong, dtype="float32")
        if am.ndim > 1:
            am = am.mean(1)
        if sr != SR:
            am = soxr.resample(am, sr, SR).astype("float32")
        am = lam_ban(am)[: SR * 30]
        dt = proc.feature_extractor(am, sampling_rate=SR, return_tensors="np").input_features[0]
        return dt, proc.tokenizer(nhan).input_ids

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    buoc_moi_epoch = len(ds) // batch
    tong = gioi_han_buoc or buoc_moi_epoch * epochs
    lich = get_linear_schedule_with_warmup(opt, min(100, tong // 10), tong)
    bat_dau = proc.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    model.train()
    buoc, t0, bao_cao = 0, time.time(), {"ban": ban, "so_mau": len(ds), "checkpoint": []}
    epoch_hien_tai = [0]

    # Đọc file + làm bẩn + trích đặc trưng chạy SONG SONG 8 tiến trình: chạy thử 2026-10-13 làm tuần tự mất
    # 5,84 s / bước (GPU ngồi chờ CPU) -> đắt gấp ~8 lần cần thiết.
    class Tap(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(ds)

        def __getitem__(self, i: int):
            return mau(*ds[i])

    def khoi_tao_tho(k: int) -> None:  # mỗi tiến trình, mỗi epoch một dãy ngẫu nhiên riêng
        lam_ban.rng = random.Random(seed * 100_003 + epoch_hien_tai[0] * 101 + k)

    def gop(lo):
        dt = torch.tensor(np.stack([a for a, _ in lo]), dtype=torch.float32)
        dai = max(len(b) for _, b in lo)
        nhan = torch.full((len(lo), dai), -100, dtype=torch.long)
        for i, (_, b) in enumerate(lo):
            nhan[i, : len(b)] = torch.tensor(b)
        # model tự thêm token bắt đầu khi dịch nhãn sang đầu vào decoder -> bỏ ở nhãn cho khỏi lặp
        if (nhan[:, 0] == bat_dau).all():
            nhan = nhan[:, 1:]
        return dt, nhan

    for ep in range(1, epochs + 1):
        epoch_hien_tai[0] = ep
        nap = torch.utils.data.DataLoader(
            Tap(), batch_size=batch, shuffle=True, drop_last=True, num_workers=8, collate_fn=gop,
            worker_init_fn=khoi_tao_tho, generator=torch.Generator().manual_seed(seed + ep),
        )
        for dt, nhan in nap:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_features=dt.cuda(), labels=nhan.cuda()).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            lich.step()
            opt.zero_grad()
            buoc += 1
            if buoc % 25 == 0:
                vram = torch.cuda.max_memory_allocated() / 2**30
                print(
                    f"  ep {ep} bước {buoc}/{tong} loss {loss.item():.3f} "
                    f"({(time.time() - t0) / buoc:.2f} s/bước, VRAM {vram:.1f} GB)",
                    flush=True,
                )
            if gioi_han_buoc and buoc >= gioi_han_buoc:
                bao_cao.update(
                    giay_moi_buoc=round((time.time() - t0) / buoc, 2),
                    vram_GB=round(torch.cuda.max_memory_allocated() / 2**30, 1),
                )
                print(json.dumps(bao_cao, ensure_ascii=False), flush=True)
                return bao_cao
        ra = Path(f"{VOL}/models/stt/phowhisper-small-ft{ban}/ep{ep}")
        model.save_pretrained(ra)
        proc.save_pretrained(ra)
        _chuyen_ct2(ra, Path(f"{ra}-ct2-int8"))
        volume.commit()
        bao_cao["checkpoint"].append(
            {"epoch": ep, "loss_cuoi": round(loss.item(), 4), "ct2": f"{ra}-ct2-int8"}
        )
        print(f"  lưu epoch {ep} -> {ra}-ct2-int8", flush=True)
    bao_cao["phut"] = round((time.time() - t0) / 60, 1)
    print(json.dumps(bao_cao, ensure_ascii=False, indent=2))
    return bao_cao


def _chuyen_ct2(vao: Path, ra: Path) -> None:
    """Giống train/modal_stt.py::chuyen_phowhisper (bản đang chạy): CT2 int8 + tokenizer + preprocessor."""
    import subprocess

    subprocess.run(
        [
            "ct2-transformers-converter",
            "--model",
            str(vao),
            "--output_dir",
            str(ra),
            "--quantization",
            "int8",
            "--force",
            "--copy_files",
            "tokenizer.json",
            "preprocessor_config.json",
        ],
        check=True,
    )
