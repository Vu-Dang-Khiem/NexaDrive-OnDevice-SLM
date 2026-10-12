"""Nhánh LoRA — CHỈ để đối chiếu, không phải bản đem deploy.

Đây là thí nghiệm Octopus-0 (full) vs Octopus-1 (LoRA) ở Table 1 / §4.4 của paper. Paper
kết luận LoRA tụt nhẹ so với full, và mình cần con số đó cho bảng kết quả của luận văn —
nếu bảo "em làm theo Octopus v2" mà bỏ mất thí nghiệm này thì không có gì để trả lời khi
bị hỏi.

SO SÁNH CHỈ CÔNG BẰNG KHI MỌI THỨ KHÁC GIỐNG HỆT. Vì vậy file này KHÔNG chép lại vòng
train: nó gọi thẳng `train()` của train/full_finetune.py qua móc `bo_model`, nên dùng
chung y nguyên bộ dữ liệu, cách mask loss, optimizer, lr, warmup, scheduler, số epoch,
batch, seed. Khác biệt duy nhất giữa hai nhánh là dòng `bo_lora()` bên dưới.

CẤU HÌNH LORA BÁM ĐÚNG PAPER §4.2 (trích nguyên văn):
    "We specify the LoRA rank as 16 and apply LoRA to the following modules: q_proj,
     k_proj, v_proj, o_proj, up_proj, down_proj. The LoRA alpha parameter is set to 32."
Chú ý paper liệt kê ĐÚNG 6 module — KHÔNG có `gate_proj`, dù Qwen2.5 có lớp đó. Giữ đúng
6 module của paper để còn nói được "cấu hình y hệt paper".

    python -m train.lora_finetune --model models/student-token \
        --train data/processed/final/train_vi.jsonl \
        --val data/processed/final/val_vi.jsonl --out runs/student-lora
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from schema.loader import CATALOG
from train.full_finetune import CauHinhTrain, train

# Paper §4.2 — 6 module, không có gate_proj.
MODULE_LORA = ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj")

# BẮT BUỘC với dự án này: 27 functional token nằm trong 2 lớp này. LoRA mặc định đóng
# băng toàn bộ model gốc, nên nếu không khai ở đây thì embedding của `<nexa_11>` giữ
# nguyên giá trị khởi tạo (trung bình + nhiễu) suốt 3 epoch — model không bao giờ học
# được token đó nghĩa là gì, mà loss vẫn giảm nên nhìn log không thấy sai ở đâu.
LOP_PHAI_TRAIN = ("embed_tokens", "lm_head")


@dataclass
class CauHinhLora:
    r: int = 16  # paper
    alpha: int = 32  # paper
    dropout: float = 0.0
    target_modules: tuple[str, ...] = MODULE_LORA
    modules_to_save: tuple[str, ...] = LOP_PHAI_TRAIN
    extra: dict[str, Any] = field(default_factory=dict)


def dem_tham_so(model: Any) -> dict[str, Any]:
    """Đếm tham số được train / tổng — để ghi vào báo cáo và để nhìn là biết có hợp lý."""
    tong = sum(p.numel() for p in model.parameters())
    hoc = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"tong": tong, "duoc_train": hoc, "ty_le_phan_tram": round(hoc / tong * 100, 3)}


def thao_rang_buoc(model: Any) -> Any:
    """Tách lớp đầu ra khỏi lớp embedding nếu hai lớp DÙNG CHUNG trọng số.

    Qwen2.5-1.5B có `tie_word_embeddings: true` — một ma trận cho cả hai. `modules_to_save` của PEFT tạo
    bản sao RIÊNG cho từng lớp; khi gộp rồi lưu với cấu hình "dùng chung", transformers chỉ lưu một ma
    trận -> bản đã train của lớp đầu ra có thể mất, model không học được 27 token ở phía ĐẦU RA mà loss vẫn
    giảm đẹp (phát hiện khi đọc code trước lần chạy LoRA đầu tiên, 2026-10-04). Tách hẳn từ đầu: hai bản
    độc lập, đều được train, đều được lưu (`tie_word_embeddings=False`). Model gốc chưa train nên hai bản
    lúc đầu y hệt -> không đổi hành vi trước train.
    """
    import torch

    emb = model.get_input_embeddings().weight
    dau_ra = model.get_output_embeddings()
    if dau_ra is not None and dau_ra.weight.data_ptr() == emb.data_ptr():
        dau_ra.weight = torch.nn.Parameter(emb.detach().clone())
    model.config.tie_word_embeddings = False
    return model


def bo_lora(model: Any, cfg_lora: CauHinhLora) -> Any:
    """Gắn adapter LoRA vào model — ĐÂY là dòng duy nhất khác nhánh full fine-tune."""
    from peft import LoraConfig, get_peft_model

    model = thao_rang_buoc(model)
    peft_cfg = LoraConfig(
        r=cfg_lora.r,
        lora_alpha=cfg_lora.alpha,
        lora_dropout=cfg_lora.dropout,
        target_modules=list(cfg_lora.target_modules),
        modules_to_save=list(cfg_lora.modules_to_save),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    # Với gradient checkpointing, phần lớn model bị đóng băng nên không có gradient nào
    # chảy ngược tới đầu vào -> checkpoint không tái tạo được activation. Dòng này bật
    # requires_grad cho output của lớp embedding để mạch gradient liền trở lại.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model


def anh_chup_token_moi(model: Any, so_token_moi: int) -> Any:
    """Chụp lại vector của 27 token mới trước khi train, để sau đối chiếu."""
    return _lay_embedding(model)[-so_token_moi:].detach().clone().float()


def _lay_embedding(model: Any) -> Any:
    goc = getattr(model, "get_base_model", None)
    return (goc() if goc else model).get_input_embeddings().weight


def kiem_token_da_hoc(model: Any, truoc: Any, so_token_moi: int) -> dict[str, Any]:
    """Rủi ro số 1 của nhánh LoRA: adapter train ngon mà 27 token mới KHÔNG hề đổi.

    Khi đó loss vẫn giảm đẹp, log không báo gì, nhưng model không học nổi functional
    token — hỏng toàn bộ ý tưởng của Octopus v2. Nên kiểm thẳng: vector có đổi không.
    """
    sau = _lay_embedding(model)[-so_token_moi:].detach().float()
    lech = (sau - truoc.to(sau.device)).abs()
    doi = int((lech.max(dim=1).values > 1e-6).sum())
    return {
        "so_token_kiem": so_token_moi,
        "so_token_da_doi": doi,
        "lech_lon_nhat": round(float(lech.max()), 6),
    }


def gop_checkpoint(model_goc: str, ckpt: str, ra: str) -> dict[str, Any]:
    """Gộp adapter của MỘT checkpoint epoch vào model gốc -> thư mục model thường (chấm được như full).

    Checkpoint theo epoch của PEFT chỉ chứa adapter + 2 lớp `modules_to_save`, không phải model đầy đủ —
    phải gộp từng cái thì mới chọn checkpoint theo ĐÚNG luật đã dùng cho full fine-tune (val mẫu, rõ ràng
    cao nhất, từ chối nhầm ≤ 5%). Kèm kiểm: lớp đầu ra của 27 token mới phải KHÁC model gốc.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    goc = thao_rang_buoc(AutoModelForCausalLM.from_pretrained(model_goc, dtype=torch.bfloat16))
    dau_ra_truoc = goc.get_output_embeddings().weight[-len(CATALOG.special_tokens):].detach().float().clone()
    gop = PeftModel.from_pretrained(goc, ckpt).merge_and_unload()
    gop.config.tie_word_embeddings = False
    dau_ra_sau = gop.get_output_embeddings().weight[-len(CATALOG.special_tokens):].detach().float()
    lech = float((dau_ra_sau - dau_ra_truoc).abs().max())
    if lech < 1e-6:
        raise RuntimeError(f"{ckpt}: lớp đầu ra của token mới KHÔNG đổi sau gộp — model vô dụng")
    Path(ra).mkdir(parents=True, exist_ok=True)
    gop.save_pretrained(ra)
    AutoTokenizer.from_pretrained(model_goc).save_pretrained(ra)
    return {"checkpoint": ckpt, "gop": ra, "lech_dau_ra_token_moi": round(lech, 6)}


def train_lora(
    cfg: CauHinhTrain, cfg_lora: CauHinhLora | None = None, out_gop: str | None = None
) -> dict[str, Any]:
    """Train LoRA rồi GỘP adapter vào model gốc.

    Phải gộp vì khâu chấm điểm (`eval/sinh_du_doan.py`) và khâu quantize GGUF đều nạp
    model theo kiểu thông thường, không biết gì về adapter. Gộp xong thì hai nhánh ra
    hai thư mục model giống hệt nhau về hình thức -> chấm bằng đúng một lệnh.
    """
    cfg_lora = cfg_lora or CauHinhLora()
    out_gop = out_gop or f"{cfg.out_dir}-gop"
    so_token_moi = len(CATALOG.special_tokens)
    ghi_nhan: dict[str, Any] = {}

    def _bo(model: Any) -> Any:
        model = bo_lora(model, cfg_lora)
        ghi_nhan["tham_so"] = dem_tham_so(model)
        ghi_nhan["_truoc"] = anh_chup_token_moi(model, so_token_moi)
        print(f"[lora] train {ghi_nhan['tham_so']['duoc_train']:,} "
              f"/ {ghi_nhan['tham_so']['tong']:,} tham số "
              f"({ghi_nhan['tham_so']['ty_le_phan_tram']}%)")
        return model

    def _sau(model: Any, tokenizer: Any) -> None:
        ghi_nhan["token_moi"] = kiem_token_da_hoc(model, ghi_nhan.pop("_truoc"), so_token_moi)
        print(f"[lora] token mới đã đổi: {ghi_nhan['token_moi']}")
        gop = model.merge_and_unload()
        gop.config.tie_word_embeddings = False  # 2 lớp đã tách (thao_rang_buoc) -> lưu cả hai
        Path(out_gop).mkdir(parents=True, exist_ok=True)
        gop.save_pretrained(out_gop)
        tokenizer.save_pretrained(out_gop)
        print(f"[lora] đã gộp adapter -> {out_gop}")

    bao_cao = train(cfg, bo_model=_bo, sau_train=_sau)
    bao_cao.update(
        {
            "nhanh": "lora",
            "lora": {
                "r": cfg_lora.r,
                "alpha": cfg_lora.alpha,
                "target_modules": list(cfg_lora.target_modules),
                "modules_to_save": list(cfg_lora.modules_to_save),
            },
            "tham_so": ghi_nhan.get("tham_so"),
            "token_moi": ghi_nhan.get("token_moi"),
            "model_da_gop": out_gop,
        }
    )
    if bao_cao["token_moi"] and bao_cao["token_moi"]["so_token_da_doi"] == 0:
        raise RuntimeError(
            "27 token mới KHÔNG đổi sau khi train — modules_to_save không có tác dụng. "
            "Model này vô dụng, đừng đem đi chấm."
        )
    # gộp từng checkpoint epoch -> chọn checkpoint theo đúng luật của full fine-tune
    bao_cao["gop_checkpoint"] = [
        gop_checkpoint(cfg.model_dir, ck, f"{ck}-gop") for ck in bao_cao.get("checkpoint", [])
    ]
    Path(cfg.out_dir, "bao_cao_train.json").write_text(
        json.dumps(bao_cao, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return bao_cao


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Train nhánh LoRA để đối chiếu với full fine-tune")
    ap.add_argument("--model", required=True, help="model ĐÃ thêm token")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--gioi-han-buoc", type=int, default=0)
    args = ap.parse_args()

    cfg = CauHinhTrain(
        model_dir=args.model,
        train_file=args.train,
        val_file=args.val,
        out_dir=args.out,
        epochs=args.epochs,
        gioi_han_buoc=args.gioi_han_buoc,
        luu_moi_epoch=not args.gioi_han_buoc,
    )
    bao_cao = train_lora(cfg, CauHinhLora(r=args.r, alpha=args.alpha))
    print(json.dumps(bao_cao, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
