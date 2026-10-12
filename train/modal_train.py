"""Phase 4 trên Modal: tải student → thêm token → train → sinh dự đoán.

Tách khỏi gen/modal_app.py (pipeline sinh data) vì khác hẳn về tài nguyên: sinh data cần
A100 80GB cho teacher 72B, còn train student 1.5B chỉ cần **A100 40GB** — tier rẻ hơn.

CÁCH CHẠY:

  # 1. Tải model gốc về Volume (máy CPU, rẻ)
  modal run train/modal_train.py::download_student

  # 2. Thêm 27 functional token (máy CPU, ~5 phút)
  modal run train/modal_train.py::them_token

  # 3. Chạy thử 30 bước để ĐO VRAM và tốc độ trước khi chạy thật (~$0.3)
  modal run train/modal_train.py::full_finetune --gioi-han-buoc 30

  # 4. Train thật (~20-30 phút, ~$1-2)
  modal run --detach train/modal_train.py::full_finetune

  # 5. Sinh dự đoán trên tập test rồi tải về chấm ở local
  modal run train/modal_train.py::sinh_du_doan
  modal volume get nexa-drive data/vi/final/du_doan_full_test_vi.jsonl ./runs/du_doan_full.jsonl

  # 6. (sau khi data đã chốt) nhánh LoRA để đối chiếu — Octopus-1 của paper
  modal run --detach train/modal_train.py::lora_finetune
  modal run train/modal_train.py::sinh_du_doan --nhanh lora \
      --model-dir /vol/models/student-lora-gop
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

STUDENT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TRAIN_GPU = "A100-40GB"  # student 1.5B: full fine-tune cần ~26GB -> 40GB là dư

VOL_NAME = "nexa-drive"
VOL_PATH = "/vol"
STUDENT_DIR = f"{VOL_PATH}/models/student-base"  # model gốc, chưa thêm token
STUDENT_TOKEN_DIR = f"{VOL_PATH}/models/student-token"  # đã thêm 27 token
OUT_FULL = f"{VOL_PATH}/models/student-full"  # sau full fine-tune (bản ĐEM DEPLOY)
OUT_LORA = f"{VOL_PATH}/models/student-lora"  # adapter LoRA (chỉ để đối chiếu)
OUT_LORA_GOP = f"{VOL_PATH}/models/student-lora-gop"  # đã gộp adapter, dùng để chấm
DATA_DIR = f"{VOL_PATH}/data"

TIMEOUT_TAI = 60 * 60
TIMEOUT_TRAIN = 60 * 60 * 4

app = modal.App("nexa-drive-train")
volume = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
REPO_ROOT = Path(__file__).resolve().parent.parent

_thu_vien = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.57.3",
        "accelerate>=1.0",
        "peft==0.17.1",  # chỉ dùng cho nhánh LoRA đối chiếu
        "huggingface_hub[hf_transfer]>=0.26",
        "pyyaml>=6.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)


def _kem_code(img: modal.Image) -> modal.Image:
    """Chép code dự án — PHẢI là bước cuối của image (Modal chặn bước dựng sau `add_local_*`)."""
    return (
        img.add_local_dir(REPO_ROOT / "schema", "/root/schema")
        .add_local_dir(REPO_ROOT / "gen", "/root/gen")
        .add_local_dir(REPO_ROOT / "eval", "/root/eval")
        .add_local_dir(REPO_ROOT / "train", "/root/train")
    )


image = _kem_code(_thu_vien)


def _stage(lang: str, ten: str, thu_muc: str = "final") -> str:
    """Dữ liệu train nằm ở `data/<lang>/final/`, TÁCH khỏi các file trung gian của
    pipeline sinh data cùng tên (ví dụ `data/vi/train_vi.jsonl` là pool thô 9.2 MiB
    chưa lọc, còn cần cho thí nghiệm quy mô dữ liệu — không được ghi đè).

    `thu_muc="final_v2"`: dữ liệu vòng 2 (`gen/dung_vong_2.py`), để riêng — vòng 1 phải còn
    nguyên để chấm lại / đối chiếu."""
    return f"{DATA_DIR}/{lang}/{thu_muc}/{ten}.jsonl"


# ── bước 1: tải model gốc (CPU) ────────────────────────────────────────────
@app.function(image=image, volumes={VOL_PATH: volume}, timeout=TIMEOUT_TAI)
def download_student(model: str = STUDENT_MODEL) -> None:
    from huggingface_hub import snapshot_download

    print(f"Tải {model} -> {STUDENT_DIR}")
    snapshot_download(model, local_dir=STUDENT_DIR)
    volume.commit()
    print("Xong")


# ── bước 2: thêm functional token (CPU) ────────────────────────────────────
@app.function(image=image, volumes={VOL_PATH: volume}, timeout=TIMEOUT_TAI)
def them_token() -> dict[str, Any]:
    from train.add_tokens import them_token as chay

    bao_cao = chay(STUDENT_DIR, STUDENT_TOKEN_DIR)
    volume.commit()
    print(json.dumps(bao_cao, ensure_ascii=False, indent=2))
    return bao_cao


# ── bước 3: full fine-tune (A100 40GB) ─────────────────────────────────────
@app.function(image=image, gpu=TRAIN_GPU, volumes={VOL_PATH: volume}, timeout=TIMEOUT_TRAIN)
def full_finetune(
    lang: str = "vi",
    epochs: float = 3.0,
    lr: float = 5e-5,
    batch_size: int = 8,
    gioi_han_buoc: int = 0,
    out_dir: str = OUT_FULL,
    thu_muc_data: str = "final",
    seed: int = 42,
) -> dict[str, Any]:
    """`gioi_han_buoc > 0`: chạy thử vài bước để đo VRAM/tốc độ trước khi chạy thật.

    Vòng 2: `--thu-muc-data final_v2 --out-dir /vol/models/student-full-v2` — ghi ra thư mục
    MỚI, không đè model vòng 1 (còn cần làm mốc so sánh).

    `seed`: đổi thứ tự xáo trộn dữ liệu -> đo nhiễu giữa các lần train (PLAN.md VÒNG 3 — M)."""
    import torch

    from train.full_finetune import CauHinhTrain, train

    cfg = CauHinhTrain(
        model_dir=STUDENT_TOKEN_DIR,
        train_file=_stage(lang, "train_vi", thu_muc_data),
        val_file=_stage(lang, "val_vi", thu_muc_data),
        out_dir=out_dir,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        gioi_han_buoc=gioi_han_buoc,
        luu_moi_epoch=not gioi_han_buoc,  # chạy thử thì khỏi lưu cho nhanh
        seed=seed,
    )
    bao_cao = train(cfg)
    bao_cao["vram_dinh_GB"] = round(torch.cuda.max_memory_allocated() / 1024**3, 1)
    volume.commit()
    print(json.dumps(bao_cao, ensure_ascii=False, indent=2))
    return bao_cao


# ── bước 3b: nhánh LoRA để đối chiếu (A100 40GB) ───────────────────────────
@app.function(image=image, gpu=TRAIN_GPU, volumes={VOL_PATH: volume}, timeout=TIMEOUT_TRAIN)
def lora_finetune(
    lang: str = "vi",
    epochs: float = 3.0,
    lr: float = 5e-5,
    batch_size: int = 8,
    r: int = 16,
    alpha: int = 32,
    gioi_han_buoc: int = 0,
    thu_muc_data: str = "final",
    out_dir: str = OUT_LORA,
    seed: int = 42,
) -> dict[str, Any]:
    """Octopus-1 của paper. Mọi siêu tham số giống hệt `full_finetune` — chỉ khác adapter.

    Chỉ chạy SAU khi dữ liệu đã chốt: so sánh full vs LoRA chỉ có nghĩa khi hai nhánh ăn
    đúng một bộ dữ liệu. Vòng 5 (PLAN.md mục V): `--thu-muc-data final_v5 --out-dir
    /vol/models/student-lora-v5-s42 --seed 42` (và s7) — CÙNG dữ liệu, hạt giống, lr, epoch với
    `student-v5-s42/s7`. Mỗi checkpoint epoch được gộp ra `<checkpoint>-gop` để chọn checkpoint.
    """
    import torch

    from train.full_finetune import CauHinhTrain
    from train.lora_finetune import CauHinhLora, train_lora

    cfg = CauHinhTrain(
        model_dir=STUDENT_TOKEN_DIR,
        train_file=_stage(lang, "train_vi", thu_muc_data),
        val_file=_stage(lang, "val_vi", thu_muc_data),
        out_dir=out_dir,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        gioi_han_buoc=gioi_han_buoc,
        luu_moi_epoch=not gioi_han_buoc,
        seed=seed,
    )
    bao_cao = train_lora(cfg, CauHinhLora(r=r, alpha=alpha), out_gop=f"{out_dir}-gop")
    bao_cao["vram_dinh_GB"] = round(torch.cuda.max_memory_allocated() / 1024**3, 1)
    volume.commit()
    print(json.dumps(bao_cao, ensure_ascii=False, indent=2))
    return bao_cao


@app.function(image=image, volumes={VOL_PATH: volume}, timeout=60 * 60, memory=16384)
def gop_mot(ckpt: str, ra: str) -> dict[str, Any]:
    """Gộp MỘT checkpoint adapter vào model gốc (CPU, không GPU). Dùng khi đổi tài khoản Modal (2026-10-06):
    chỉ chuyển adapter LoRA v5 (~1 GB) từ tài khoản cũ sang rồi gộp lại ở đây — kiểm bằng mã băm so với bản
    gộp cũ."""
    from train.lora_finetune import gop_checkpoint

    volume.reload()  # file vừa upload lại (bản trước bị `modal volume put` cắt cụt) -> đọc bản mới nhất
    bao_cao = gop_checkpoint(STUDENT_TOKEN_DIR, ckpt, ra)
    volume.commit()
    print(json.dumps(bao_cao, ensure_ascii=False, indent=2))
    return bao_cao


# ── GRPO nối sau SFT v6.1 (A100 40GB) — PLAN.md Phase 7 "GRPO — KẾ HOẠCH + TIÊU CHÍ KHOÁ TRƯỚC" ──────────
# image riêng (thêm TRL) — không đụng image SFT đang chạy ổn
# ghim datasets: bản mới nhất (5.x) kéo huggingface_hub 2.x -> transformers 4.57.3 không import được
# (chạy thử 2026-10-07 lỗi "Could not import module 'AutoModel'")
grpo_image = _kem_code(_thu_vien.pip_install("trl==0.24.0", "datasets==3.6.0", "huggingface_hub<1.0"))
GRPO_GOC = {  # bản v6.1 đã chọn (ck4192, cả hai hạt giống — luật chọn checkpoint mục v6.1)
    "s42": f"{VOL_PATH}/models/student-lora-v61-s42/checkpoint-4192-gop",
    "s7": f"{VOL_PATH}/models/student-lora-v61-s7/checkpoint-4192-gop",
}


@app.function(image=grpo_image, gpu=TRAIN_GPU, volumes={VOL_PATH: volume}, timeout=TIMEOUT_TRAIN)
def grpo(
    hat_giong: str = "s42", so_cau: int = 0, gioi_han_buoc: int = 0, out_dir: str = ""
) -> dict[str, Any]:
    """`gioi_han_buoc > 0`: CHẠY THỬ (không lưu) — đo giây / bước, VRAM, phân bố kết cục lúc train để ước
    tính lại chi phí trước khi chạy thật. Chạy thật:
    `modal run --detach train/modal_train.py::grpo --hat-giong s42`."""
    import torch

    from train.grpo import CauHinhGRPO, train_grpo

    cfg = CauHinhGRPO(
        model_dir=GRPO_GOC[hat_giong],
        tokenizer_dir=STUDENT_TOKEN_DIR,
        train_file=_stage("vi", "train_vi", "final_v6_1"),
        out_dir=out_dir or f"{VOL_PATH}/models/student-grpo-v61-{hat_giong}",
        seed=42 if hat_giong == "s42" else 7,
        so_cau=so_cau,
        gioi_han_buoc=gioi_han_buoc,
    )
    bao_cao = train_grpo(cfg)
    bao_cao["vram_dinh_GB"] = round(torch.cuda.max_memory_allocated() / 1024**3, 1)
    volume.commit()
    print(json.dumps(bao_cao, ensure_ascii=False, indent=2, default=str))
    return bao_cao


@app.function(image=grpo_image, volumes={VOL_PATH: volume}, timeout=60 * 60, memory=16384)
def gop_grpo(hat_giong: str = "s42") -> dict[str, Any]:
    """Gộp adapter GRPO (checkpoint CUỐI — chốt trước) vào v6.1 cùng hạt giống (CPU, không GPU)."""
    from train.grpo import gop_grpo as gop

    volume.reload()
    nguon = f"{VOL_PATH}/models/student-grpo-v61-{hat_giong}"
    bao_cao = gop(GRPO_GOC[hat_giong], nguon, f"{nguon}-gop", STUDENT_TOKEN_DIR)
    volume.commit()
    print(json.dumps(bao_cao, ensure_ascii=False, indent=2))
    return bao_cao


# ── bước 4: sinh dự đoán trên tập test (A100 40GB) ─────────────────────────
@app.function(image=image, gpu=TRAIN_GPU, volumes={VOL_PATH: volume}, timeout=TIMEOUT_TRAIN)
def sinh_du_doan(
    lang: str = "vi",
    model_dir: str = OUT_FULL,
    tap: str = "test_vi",
    batch_size: int = 16,
    nhanh: str = "full",
    thu_muc_data: str = "final",
) -> dict[str, Any]:
    """`nhanh` vào tên file đầu ra để dự đoán của full và LoRA không ghi đè lên nhau.
    `thu_muc_data`: nơi đọc tập đầu vào và ghi dự đoán (vd "final_v2" cho val vòng 2)."""
    from eval.sinh_du_doan import chay

    ra = _stage(lang, f"du_doan_{nhanh}_{tap}", thu_muc_data)
    # Checkpoint theo epoch không kèm tokenizer -> luôn lấy tokenizer từ model gốc đã
    # thêm token (27 token là như nhau ở mọi checkpoint nên hoàn toàn an toàn).
    vao = _stage(lang, tap, thu_muc_data)
    bao_cao = chay(model_dir, vao, ra, batch_size, tokenizer_dir=STUDENT_TOKEN_DIR)
    bao_cao["file_ra"] = ra
    volume.commit()
    print(json.dumps(bao_cao, ensure_ascii=False, indent=2))
    return bao_cao


# ── đo độ tự tin ở token quyết định (L4 — chỉ suy luận model 1.5B, không cần A100) ──
@app.function(image=image, gpu="L4", volumes={VOL_PATH: volume}, timeout=60 * 60)
def do_tu_tin(
    model_dir: str = f"{VOL_PATH}/models/student-v4dc-s42/checkpoint-2206",
    thu_muc_data: str = "final_v4_doi_chung",
    ten_ra: str = "tu_tin_v4dcs42ck2206",
    batch_size: int = 32,
    cac_tap: str = "val_vi,dev_vi_2",
    cho_phep_test: bool = False,
) -> None:
    """Chạy `eval/do_tu_tin.py` trên val + dev ngay trên Modal (model đã nằm sẵn trên Volume) —
    tránh tải 3 GB về máy qua mạng chập chờn. Chỉ val/dev: script tự chặn file có chữ "test".
    Tải kết quả: `modal volume get nexa-drive data/vi/<thu_muc_data>/<ten_ra>.jsonl`."""
    from eval.do_tu_tin import chay

    tap = [_stage("vi", t, thu_muc_data) for t in cac_tap.split(",")]
    ra = _stage("vi", ten_ra, thu_muc_data)
    chay(model_dir, tap, ra, batch_size, cho_phep_test)
    volume.commit()


# ── mã băm SHA256 từng file của một model trên Volume (CPU) ─────────────────
@app.function(image=image, volumes={VOL_PATH: volume}, timeout=30 * 60)
def ma_bam(model_dir: str) -> dict[str, str]:
    """Tải model về máy phải so mã băm với bản này — kích thước file KHÔNG chứng minh tải xong
    (Modal tải nhiều đoạn song song, file đủ kích thước ngay từ đầu; PLAN.md mục S)."""
    import hashlib

    kq = {}
    for p in sorted(Path(model_dir).iterdir()):
        if p.is_file():
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for khoi in iter(lambda: f.read(1 << 24), b""):
                    h.update(khoi)
            kq[p.name] = h.hexdigest()
    print(json.dumps(kq, indent=2))
    return kq


# ── tiện ích: xem Volume đang có gì ────────────────────────────────────────
@app.function(image=image, volumes={VOL_PATH: volume}, timeout=300)
def kiem_tra(lang: str = "vi") -> None:
    for ten, p in (
        ("model gốc", STUDENT_DIR),
        ("model đã thêm token", STUDENT_TOKEN_DIR),
        ("model sau full FT", OUT_FULL),
        ("model sau LoRA (gộp)", OUT_LORA_GOP),
    ):
        co = Path(p).exists()
        print(f"  {ten:22} {'CÓ' if co else 'chưa có'}  {p}")
    for ten in ("train_vi", "val_vi", "test_vi"):
        p = Path(_stage(lang, ten))
        n = sum(1 for _ in open(p, encoding="utf-8")) if p.exists() else 0
        print(f"  {ten:22} {n if n else 'chưa có'} dòng")
