"""Phase 6 trên Modal (CPU): chuyển STT `nvidia/parakeet-ctc-0.6b-Vietnamese` (.nemo) sang ONNX để chạy
bằng `onnx-asr` trên Windows CPU — NeMo chủ yếu chạy Linux, nên chuyển một lần ở đây (PLAN.md Phase 6).

Định dạng đầu ra theo `onnx-asr` (kiểm trong mã nguồn onnx_asr/models/nemo.py): `model.onnx` (+ dữ liệu
ngoài), `model.int8.onnx`, `vocab.txt` ("token id" mỗi dòng, thêm `<blk>` cuối), `config.json`
{"model_type": "nemo-conformer-ctc", "features_size": số dải mel, "subsampling_factor": 8}.

CÁCH CHẠY:
  modal run train/modal_stt.py::xuat_parakeet
  modal volume put nexa-drive runs/giong_noi/stt_thu /stt_thu        # vài file wav 16 kHz để so
  modal run train/modal_stt.py::chep_tham_chieu
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

VOL_PATH = "/vol"
MODEL_HF = "nvidia/parakeet-ctc-0.6b-Vietnamese"
RA = f"{VOL_PATH}/models/stt/parakeet-ctc-0.6b-vi-onnx"

app = modal.App("nexa-drive-stt")
volume = modal.Volume.from_name("nexa-drive", create_if_missing=True)
# Ghim phiên bản: lần dựng đầu (không ghim) pip dò ngược phiên bản rồi chọn `tokenizers` cũ phải biên dịch
# bằng Rust -> dựng ảnh hỏng. torch bản CPU cho nhẹ (chỉ chuyển đổi, không train).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "build-essential")
    .pip_install("torch==2.8.0", "torchaudio==2.8.0", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(
        "nemo_toolkit[asr]==3.0.0",
        "transformers==4.57.3",
        "lightning==2.4.0",
        "onnx",
        "onnxruntime",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


def _nap():
    """Repo HF chỉ chứa NGUYÊN một file `.nemo` -> `from_pretrained` (đòi thư mục đã giải nén có
    model_config.yaml) lỗi; tải đúng file `.nemo` rồi `restore_from`."""
    import nemo.collections.asr as nemo_asr
    from huggingface_hub import hf_hub_download, list_repo_files

    ten = next(f for f in list_repo_files(MODEL_HF) if f.endswith(".nemo"))
    m = nemo_asr.models.ASRModel.restore_from(hf_hub_download(MODEL_HF, ten), map_location="cpu")
    m.eval()
    return m


@app.function(image=image, volumes={VOL_PATH: volume}, cpu=8, memory=32768, timeout=60 * 60)
def xuat_parakeet() -> dict:
    import hashlib

    from omegaconf import OmegaConf
    from onnxruntime.quantization import QuantType, quantize_dynamic

    m = _nap()
    ra = Path(RA)
    ra.mkdir(parents=True, exist_ok=True)
    tien_xu_ly = OmegaConf.to_container(m.cfg.preprocessor, resolve=True)
    (ra / "nemo_preprocessor.json").write_text(json.dumps(tien_xu_ly, indent=2, ensure_ascii=False))
    print("loại model:", type(m).__name__, "· tiền xử lý:", tien_xu_ly)

    m.export(str(ra / "model.onnx"))
    with (ra / "vocab.txt").open("w", encoding="utf-8") as f:
        for i, tok in enumerate([*m.tokenizer.vocab, "<blk>"]):
            f.write(f"{tok} {i}\n")
    cfg = {"model_type": "nemo-conformer-ctc", "features_size": int(tien_xu_ly.get("features", 80)),
           "subsampling_factor": 8}
    (ra / "config.json").write_text(json.dumps(cfg))
    quantize_dynamic(str(ra / "model.onnx"), str(ra / "model.int8.onnx"), weight_type=QuantType.QInt8,
                     use_external_data_format=False)

    bam = {}
    for p in sorted(ra.iterdir()):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for k in iter(lambda: f.read(1 << 24), b""):
                h.update(k)
        bam[p.name] = {"sha256": h.hexdigest(), "byte": p.stat().st_size}
    (ra / "ma_bam.json").write_text(json.dumps(bam, indent=2))
    volume.commit()
    print(json.dumps({"config": cfg, "tep": bam}, indent=2))
    return bam


anh_ct2 = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("ctranslate2==4.6.0", "transformers==4.57.3", "huggingface_hub[hf_transfer]", "accelerate")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(image=anh_ct2, volumes={VOL_PATH: volume}, cpu=4, memory=16384, timeout=60 * 60)
def chuyen_phowhisper(kich_co: str = "small") -> dict:
    """PhoWhisper (vinai) -> CTranslate2 int8 cho faster-whisper trên CPU — CÙNG điều kiện nén với
    Parakeet int8 (PLAN.md Phase 6 (3)). Ra `models/stt/phowhisper-<kich_co>-ct2-int8/` + mã băm."""
    import hashlib
    import subprocess

    ra = Path(f"{VOL_PATH}/models/stt/phowhisper-{kich_co}-ct2-int8")
    ra.parent.mkdir(parents=True, exist_ok=True)
    lenh = ["ct2-transformers-converter", "--model", f"vinai/PhoWhisper-{kich_co}", "--output_dir", str(ra),
            "--quantization", "int8", "--force", "--copy_files", "tokenizer.json", "preprocessor_config.json"]
    subprocess.run(lenh, check=True)
    bam = {}
    for p in sorted(ra.iterdir()):
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        bam[p.name] = {"sha256": h, "byte": p.stat().st_size}
    (ra / "ma_bam.json").write_text(json.dumps(bam, indent=2))
    volume.commit()
    print(json.dumps(bam, indent=2))
    return bam


@app.function(image=image, volumes={VOL_PATH: volume}, cpu=8, memory=16384, timeout=30 * 60)
def chep_tham_chieu(thu_muc: str = f"{VOL_PATH}/stt_thu") -> dict:
    """Bản NeMo GỐC nghe các wav (16 kHz mono) -> chữ, làm mốc kiểm bản ONNX chạy trên máy ra cùng kết quả."""
    m = _nap()
    tep = sorted(str(p) for p in Path(thu_muc).glob("*.wav"))
    ra = m.transcribe(tep, batch_size=4)
    chu = [r.text if hasattr(r, "text") else str(r) for r in (ra[0] if isinstance(ra, tuple) else ra)]
    kq = {Path(p).name: c for p, c in zip(tep, chu, strict=True)}
    Path(RA, "tham_chieu_nemo.json").write_text(json.dumps(kq, indent=2, ensure_ascii=False))
    volume.commit()
    print(json.dumps(kq, indent=2, ensure_ascii=False))
    return kq
