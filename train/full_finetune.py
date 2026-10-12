"""Full fine-tune student — nhánh CHÍNH của Phase 4 (bản đem deploy).

Cùng một lần chạy này vừa là **SFT** (cách dạy: học từ cặp prompt → đáp án đúng) vừa là
**full fine-tune** (phạm vi sửa: toàn bộ 1,54 tỷ tham số). Nhánh LoRA dùng chung y hệt
file dữ liệu và hyperparam, chỉ khác đúng 1 biến — để so sánh công bằng như Table 1 của
Octopus v2 (Octopus-0 full vs Octopus-1 LoRA).

Hyperparam bám theo paper: AdamW, lr 5e-5, 10 bước warmup, lịch giảm tuyến tính, 3 epoch.

Dùng `Trainer` của transformers thay vì thư viện SFT có sẵn: ít phụ thuộc hơn, và quan
trọng là tự kiểm soát được phần mask loss (xem train/du_lieu.py) — thứ dễ sai nhất mà
lại không báo lỗi gì khi sai.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from train.du_lieu import GhepLo, doc_jsonl, tao_mau


@dataclass
class CauHinhTrain:
    model_dir: str  # model ĐÃ thêm token (train/add_tokens.py)
    train_file: str
    val_file: str
    out_dir: str

    epochs: float = 3.0
    lr: float = 5e-5
    warmup_steps: int = 10
    batch_size: int = 8  # mỗi bước GPU
    grad_accum: int = 2  # batch hiệu dụng = 8 × 2 = 16
    # Đo bằng tokenizer thật trên cả 3 tập: trung bình 48, p99 136, dài nhất 218 token.
    # Cắt ở 256 -> mất 0 mẫu. Để rộng hơn chỉ tổ phí, để hẹp hơn là cắt cụt câu trả lời.
    do_dai_toi_da: int = 256
    seed: int = 42
    # Tiết kiệm VRAM: tính lại activation khi lan ngược thay vì giữ hết trong bộ nhớ.
    # Chậm hơn ~20% nhưng giảm đáng kể VRAM — đổi lại chạy vừa A100 40GB thoải mái.
    gradient_checkpointing: bool = True
    gioi_han_buoc: int = 0  # >0: chạy thử vài bước để đo VRAM/tốc độ trước khi chạy thật
    luu_moi_epoch: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


def so_buoc_moi_epoch(so_mau: int, batch_size: int, grad_accum: int) -> int:
    """Số lần cập nhật trọng số trong 1 epoch — để ước lượng thời gian chạy."""
    so_lo = -(-so_mau // batch_size)  # làm tròn lên
    return max(1, -(-so_lo // grad_accum))


def nap_du_lieu(tokenizer: Any, path: str, do_dai_toi_da: int) -> list[dict[str, list[int]]]:
    return [tao_mau(tokenizer, r, do_dai_toi_da) for r in doc_jsonl(path)]


def train(
    cfg: CauHinhTrain,
    bo_model: Callable[[Any], Any] | None = None,
    sau_train: Callable[[Any, Any], None] | None = None,
) -> dict[str, Any]:
    """Chạy train, lưu checkpoint mỗi epoch, trả về báo cáo.

    `bo_model`: hàm bọc model TRƯỚC khi train (nhánh LoRA dùng để gắn adapter).
    `sau_train`: gọi với (model, tokenizer) sau khi đã lưu (nhánh LoRA dùng để gộp adapter).

    Hai móc này tồn tại để nhánh LoRA dùng CHUNG đúng vòng train và đúng bộ siêu tham số
    này — nếu copy thành file thứ hai thì chỉ cần lệch một dòng là bảng so sánh vô nghĩa.
    """
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_dir, dtype=torch.bfloat16)
    model.config.use_cache = False  # bắt buộc tắt khi dùng gradient checkpointing
    if bo_model is not None:
        model = bo_model(model)

    tap_train = nap_du_lieu(tokenizer, cfg.train_file, cfg.do_dai_toi_da)
    tap_val = nap_du_lieu(tokenizer, cfg.val_file, cfg.do_dai_toi_da)
    print(f"[train] {len(tap_train)} mẫu train · {len(tap_val)} mẫu val")
    print(f"[train] ~{so_buoc_moi_epoch(len(tap_train), cfg.batch_size, cfg.grad_accum)} bước/epoch")

    args = TrainingArguments(
        output_dir=cfg.out_dir,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.gioi_han_buoc or -1,
        learning_rate=cfg.lr,
        warmup_steps=cfg.warmup_steps,
        lr_scheduler_type="linear",
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        gradient_checkpointing=cfg.gradient_checkpointing,
        bf16=True,
        optim="adamw_torch",
        max_grad_norm=1.0,
        logging_steps=20,
        eval_strategy="epoch" if cfg.luu_moi_epoch else "no",
        save_strategy="epoch" if cfg.luu_moi_epoch else "no",
        save_total_limit=4,  # giữ đủ checkpoint của 3 epoch để còn chọn
        report_to=[],
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tap_train,
        eval_dataset=tap_val,
        data_collator=GhepLo(tokenizer.pad_token_id or tokenizer.eos_token_id),
        # Không có dòng này thì checkpoint theo epoch KHÔNG kèm tokenizer, và khâu chấm
        # điểm nạp checkpoint đó sẽ chết ngay ở `AutoTokenizer.from_pretrained`.
        processing_class=tokenizer,
    )
    ket_qua = trainer.train()

    trainer.save_model(cfg.out_dir)
    tokenizer.save_pretrained(cfg.out_dir)
    if sau_train is not None:
        sau_train(trainer.model, tokenizer)

    lich_su = [x for x in trainer.state.log_history if "eval_loss" in x or "loss" in x]
    bao_cao = {
        "so_mau_train": len(tap_train),
        "so_mau_val": len(tap_val),
        "so_buoc": ket_qua.global_step,
        "loss_cuoi": round(ket_qua.training_loss, 4),
        "lich_su": lich_su[-10:],
        "checkpoint": sorted(str(p) for p in Path(cfg.out_dir).glob("checkpoint-*")),
    }
    Path(cfg.out_dir, "bao_cao_train.json").write_text(
        json.dumps(bao_cao, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return bao_cao
