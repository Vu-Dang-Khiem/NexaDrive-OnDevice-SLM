"""GRPO nối sau SFT v6.1 (PLAN.md Phase 7 — "GRPO — KẾ HOẠCH + TIÊU CHÍ KHOÁ TRƯỚC").

Ba phần, hai phần đầu thuần Python — test ở máy không cần GPU / TRL:
  `diem_thuong`   điểm thưởng của MỘT output so với đáp án (dùng lại bộ chấm luật C
                  `eval/metrics.cham_mot_cau`)
  `chon_cau`      chọn câu luyện từ dữ liệu train v6.1 (KHÔNG test, KHÔNG câu lỗi demo), ưu tiên nhóm yếu
  `train_grpo`    TRL GRPOTrainer + LoRA mới trên model v6.1 đã gộp (chạy trên Modal,
                  `train/modal_train.py::grpo`)

Điểm thưởng (khoá trước khi chạy, người dùng chốt 2026-10-07): đúng +1 · hỏi lại thừa −0,2 · từ chối nhầm
−0,5 · đề xuất (hỏi lại) sai hàm / hướng −0,5 · output hỏng −1 · LÀM SAI LÊN XE −5 (gọi bừa, sai hàm / tham
số khi thực thi, làm luôn câu đáng hỏi lại). −5 neo theo ngưỡng runtime 0,80: làm ngang hỏi khi
p − (1−p)·W = −0,2 tại p = 0,8.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from eval.metrics import (
    DUNG,
    HOI_LAI_THUA,
    KHONG_PARSE,
    TU_CHOI_NHAM,
    cham_mot_cau,
    doc_output,
    nhom_cua_dap_an,
)
from eval.sinh_du_doan import cat_output

THUONG = {"dung": 1.0, "hoi_lai_thua": -0.2, "tu_choi_nham": -0.5, "de_xuat_sai": -0.5, "khong_parse": -1.0,
          "lam_sai_len_xe": -5.0}


def loai_ket_cuc(output: str, dap_an: str) -> str:
    """Kết cục theo NGHĨA điểm thưởng (gộp các kết cục luật C theo hậu quả lên xe)."""
    out = cat_output(output)
    ket, _ = cham_mot_cau(out, dap_an)
    if ket == DUNG:
        return "dung"
    if ket == KHONG_PARSE:
        return "khong_parse"
    if ket == HOI_LAI_THUA:
        return "hoi_lai_thua"
    if ket == TU_CHOI_NHAM:
        return "tu_choi_nham"
    # còn lại: gọi bừa / sai hàm / sai tham số / thiếu hỏi lại. Model chỉ ĐỀ XUẤT (hỏi lại) thì chưa đụng xe.
    return "de_xuat_sai" if doc_output(out).hoi_lai else "lam_sai_len_xe"


def diem_thuong(output: str, dap_an: str) -> float:
    return THUONG[loai_ket_cuc(output, dap_an)]


# ── chọn câu luyện ─────────────────────────────────────────────────────────────────────────────────
@dataclass
class CauHinhChon:
    """Tỉ lệ nhóm (khoá cùng kế hoạch): nặng về hỏi lại / ngoài phạm vi / hai lượt — nơi model phân vân; một
    phần câu rõ ràng để không quên. `ty_le_chu_thuong`: đổi chữ đầu câu thành thường (đa dạng hình thức, từ
    chính câu train — model nhạy chữ hoa chữ đầu, PLAN.md Phase 7)."""

    so_cau: int = 3200
    mo_ho: float = 0.35
    ngoai_pham_vi: float = 0.30
    ro_rang: float = 0.35
    ty_le_hai_luot_toi_thieu: float = 0.25
    ty_le_chu_thuong: float = 0.25
    seed: int = 2026_10_08


_MO_USER = "<|im_start|>user\n"


def ha_chu_dau(prompt: str) -> str:
    """Đổi chữ cái đầu CÂU NGƯỜI DÙNG trong prompt ChatML thành thường. Prompt hai lượt: câu nằm sau dòng
    "Lượt trước: …"."""
    i = prompt.index(_MO_USER) + len(_MO_USER)
    if prompt.startswith("Lượt trước:", i):
        i = prompt.index("\n", i) + 1
    return prompt[:i] + prompt[i:i + 1].lower() + prompt[i + 1:]


def nhom(row: dict[str, Any]) -> str:
    c = row["completion"]
    if not c.lstrip().startswith("<nexa"):
        return "reply"
    return nhom_cua_dap_an(doc_output(c))


def chon_cau(rows: list[dict[str, Any]], cfg: CauHinhChon | None = None) -> list[dict[str, Any]]:
    """-> [{"prompt", "dap_an", "nhom", "hai_luot", "chu_thuong"}] (bỏ mẫu lượt 2 "reply" — GRPO chỉ luyện
    quyết định lượt 1)."""
    cfg = cfg or CauHinhChon()
    rng = random.Random(cfg.seed)
    theo_nhom: dict[str, list[dict[str, Any]]] = {"mo_ho": [], "ngoai_pham_vi": [], "ro_rang": []}
    for r in rows:
        n = nhom(r)
        if n in theo_nhom:
            theo_nhom[n].append(r)
    chon: list[dict[str, Any]] = []
    for n, ty_le in (("mo_ho", cfg.mo_ho), ("ngoai_pham_vi", cfg.ngoai_pham_vi), ("ro_rang", cfg.ro_rang)):
        can = round(cfg.so_cau * ty_le)
        ds = theo_nhom[n][:]
        rng.shuffle(ds)
        hai = [r for r in ds if "Lượt trước:" in r["prompt"]]
        mot = [r for r in ds if "Lượt trước:" not in r["prompt"]]
        so_hai = min(len(hai), max(round(can * cfg.ty_le_hai_luot_toi_thieu), 0))
        lay = hai[:so_hai] + mot[: can - so_hai]
        for r in lay:
            chu_thuong = rng.random() < cfg.ty_le_chu_thuong
            chon.append({
                "prompt": ha_chu_dau(r["prompt"]) if chu_thuong else r["prompt"],
                "dap_an": r["completion"],
                "nhom": n,
                "hai_luot": "Lượt trước:" in r["prompt"],
                "chu_thuong": chu_thuong,
            })
    rng.shuffle(chon)
    return chon


# ── chọn câu KHÓ (người dùng duyệt 2026-10-07 sau chạy thử) ─────────────────────────────────────────
# Chạy thử: chọn ngẫu nhiên từ câu train -> v6.1 đúng 97%, 99% nhóm 8 đáp án y hệt -> GRPO không có tín hiệu.
# GRPO chỉ học ở câu model LÚC ĐÚNG LÚC SAI -> chấm xác suất token quyết định (token đầu) của chính model trên
# MỌI câu train lượt 1 + bản chữ đầu viết thường (dạng model đang yếu), giữ câu PHÂN VÂN (p_top1 < ngưỡng) +
# một ít câu chắc chắn để không quên. Vẫn chỉ dữ liệu train; không đổi điểm thưởng / tiêu chí.
def ung_vien(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mọi câu train lượt 1 (bỏ mẫu lượt 2 "reply") ở dạng gốc + dạng chữ đầu viết thường."""
    ra: list[dict[str, Any]] = []
    for r in rows:
        n = nhom(r)
        if n == "reply":
            continue
        hai = "Lượt trước:" in r["prompt"]
        ra.append({"prompt": r["prompt"], "dap_an": r["completion"], "nhom": n, "hai_luot": hai,
                   "chu_thuong": False})
        thuong = ha_chu_dau(r["prompt"])
        if thuong != r["prompt"]:
            ra.append({"prompt": thuong, "dap_an": r["completion"], "nhom": n, "hai_luot": hai,
                       "chu_thuong": True})
    return ra


def chon_kho(ds: list[dict[str, Any]], p_top1: list[float], so_kho: int, so_de: int, nguong: float,
             seed: int) -> list[dict[str, Any]]:
    """Giữ tối đa `so_kho` câu có p_top1 < `nguong` (phân vân) + `so_de` câu chắc chắn (chống quên)."""
    rng = random.Random(seed)
    kho = [d for d, p in zip(ds, p_top1, strict=True) if p < nguong]
    de = [d for d, p in zip(ds, p_top1, strict=True) if p >= nguong]
    rng.shuffle(kho)
    rng.shuffle(de)
    chon = kho[:so_kho] + de[:so_de]
    rng.shuffle(chon)
    return chon


# ── train (Modal, A100) ────────────────────────────────────────────────────────────────────────────
@dataclass
class CauHinhGRPO:
    model_dir: str  # model v6.1 đã gộp (checkpoint-4192-gop)
    train_file: str  # final_v6_1/train_vi.jsonl
    out_dir: str
    tokenizer_dir: str = ""  # rỗng = model_dir; trên Modal lấy student-token (27 token như nhau ở mọi bản)
    seed: int = 42
    so_cau: int = 0  # 0 = HẾT câu phân vân + `so_de` câu chắc (chốt 2026-10-07); > 0 = giới hạn tổng số câu
    so_epoch: int = 2
    num_generations: int = 8
    batch_completion: int = 64  # mỗi bước = 64 output = 8 câu × 8 đáp án
    grad_accum: int = 2
    lr: float = 1e-5
    beta: float = 0.04  # phạt KL giữ gần v6.1
    temperature: float = 1.0
    max_completion: int = 48
    lora_r: int = 16
    lora_alpha: int = 32
    gioi_han_buoc: int = 0  # > 0: chạy thử, đo tốc độ
    chon_kho: bool = True  # False = chọn ngẫu nhiên theo nhóm (`chon_cau`, cách của lần chạy thử đầu)
    nguong_kho: float = 0.95  # p_top1 < 0,95 -> 8 lần lấy mẫu ở nhiệt 1,0 có ~34% khả năng khác nhau trở lên
    so_de: int = 400


def gop_grpo(model_goc: str, adapter: str, ra: str, tokenizer_dir: str) -> dict[str, Any]:
    """Gộp adapter GRPO vào model v6.1 -> model thường (chấm / GGUF như mọi bản). Khác
    `lora_finetune.gop_checkpoint`: GRPO KHÔNG train lớp embedding / đầu ra (token mới giữ nguyên từ SFT) ->
    kiểm thay vào đó là trọng số attention đã ĐỔI sau gộp (adapter rỗng / không học gì thì dừng)."""
    from pathlib import Path

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    goc = AutoModelForCausalLM.from_pretrained(model_goc, dtype=torch.bfloat16)
    truoc = goc.model.layers[0].self_attn.q_proj.weight.detach().float().clone()
    gop = PeftModel.from_pretrained(goc, adapter).merge_and_unload()
    lech = float((gop.model.layers[0].self_attn.q_proj.weight.detach().float() - truoc).abs().max())
    if lech < 1e-7:
        raise RuntimeError(f"{adapter}: trọng số không đổi sau gộp — adapter GRPO không học gì")
    gop.config.tie_word_embeddings = False
    Path(ra).mkdir(parents=True, exist_ok=True)
    gop.save_pretrained(ra)
    AutoTokenizer.from_pretrained(tokenizer_dir).save_pretrained(ra)
    return {"adapter": adapter, "gop": ra, "lech_q_proj_lop_0": lech}


def _p_top1(model: Any, tok: Any, prompts: list[str], lo: int = 128) -> list[float]:
    """Xác suất lớn nhất ở token ĐẦU câu trả lời (token quyết định: hàm / hỏi lại / từ chối), nhiệt 1,0."""
    import torch

    model.eval()
    tok.padding_side = "left"
    ra: list[float] = []
    with torch.no_grad():
        for i in range(0, len(prompts), lo):
            b = tok(prompts[i:i + lo], return_tensors="pt", padding=True, add_special_tokens=False)
            b = b.to(model.device)
            logit = model(**b, logits_to_keep=1).logits[:, -1, :].float()  # chỉ vị trí cuối: đỡ ~4 GB / lô
            ra.extend(torch.softmax(logit, dim=-1).max(dim=-1).values.tolist())
    model.train()
    return ra


def train_grpo(cfg: CauHinhGRPO) -> dict[str, Any]:
    import json
    import time

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from train.lora_finetune import MODULE_LORA

    rows = [json.loads(x) for x in open(cfg.train_file, encoding="utf-8")]
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_dir or cfg.model_dir)
    id_end = tok.convert_tokens_to_ids("<nexa_end>")
    id_im_end = tok.convert_tokens_to_ids("<|im_end|>")
    tok.pad_token = tok.pad_token or "<|endoftext|>"
    model = AutoModelForCausalLM.from_pretrained(cfg.model_dir, torch_dtype=torch.bfloat16)
    if torch.cuda.is_available():
        model = model.to("cuda")  # chấm độ chắc chắn ~60 nghìn câu trên GPU
    bao_chon: dict[str, Any] = {}
    if cfg.chon_kho:
        uv = ung_vien(rows)
        t0 = time.time()
        p = _p_top1(model, tok, [u["prompt"] for u in uv])
        kho = [u for u, q in zip(uv, p, strict=True) if q < cfg.nguong_kho]
        so_kho = cfg.so_cau - cfg.so_de if cfg.so_cau else len(kho)
        cau = chon_kho(uv, p, so_kho, cfg.so_de, cfg.nguong_kho, cfg.seed)
        bao_chon = {
            "ung_vien": len(uv), "phan_van": len(kho), "giay_cham": round(time.time() - t0),
            "phan_van_theo_nhom": {n: sum(u["nhom"] == n for u in kho)
                                   for n in ("mo_ho", "ngoai_pham_vi", "ro_rang")},
            "phan_van_chu_thuong": sum(u["chu_thuong"] for u in kho),
            "phan_van_hai_luot": sum(u["hai_luot"] for u in kho),
        }
        print(json.dumps(bao_chon, ensure_ascii=False))
    else:
        cau = chon_cau(rows, CauHinhChon(so_cau=cfg.so_cau or 3200, seed=cfg.seed))

    ds = Dataset.from_list([{"prompt": c["prompt"], "dap_an": c["dap_an"], "nhom": c["nhom"]} for c in cau])
    dem: dict[str, int] = {}

    def thuong(completions: list[str], dap_an: list[str], completion_ids: list[list[int]] | None = None,
               **_: Any) -> list[float]:
        # functional token là token ĐẶC BIỆT -> chữ TRL giải mã sẵn bị bỏ; giải mã lại từ id, giữ token đặc
        # biệt (chạy thử: bản TRL không đưa completion_ids thì gần như mọi output "khong_parse" -> lộ ngay)
        ra = []
        for k, (txt, d) in enumerate(zip(completions, dap_an, strict=True)):
            ids = completion_ids[k] if completion_ids is not None else None
            out = tok.decode(ids, skip_special_tokens=False) if ids is not None else txt
            loai = loai_ket_cuc(out, d)
            dem[loai] = dem.get(loai, 0) + 1
            ra.append(THUONG[loai])
        return ra

    args = GRPOConfig(
        output_dir=cfg.out_dir,
        seed=cfg.seed,
        learning_rate=cfg.lr,
        beta=cfg.beta,
        num_generations=cfg.num_generations,
        per_device_train_batch_size=cfg.batch_completion,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.so_epoch,
        max_steps=cfg.gioi_han_buoc if cfg.gioi_han_buoc else -1,
        max_completion_length=cfg.max_completion,
        temperature=cfg.temperature,
        bf16=True,
        logging_steps=5,
        save_strategy="no" if cfg.gioi_han_buoc else "steps",
        save_steps=50,
        report_to=[],
        generation_kwargs={"eos_token_id": [id_end, id_im_end]},  # dừng ngay ở <nexa_end>
    )
    lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.0,
                      target_modules=list(MODULE_LORA), bias="none", task_type="CAUSAL_LM")
    trainer = GRPOTrainer(model=model, reward_funcs=thuong, args=args, train_dataset=ds,
                          processing_class=tok, peft_config=lora)
    t0 = time.time()
    trainer.train()
    giay = time.time() - t0
    if not cfg.gioi_han_buoc:
        trainer.save_model(cfg.out_dir)
    tong = sum(dem.values()) or 1
    return {
        "chon_cau": bao_chon,
        "so_cau": len(cau), "buoc": trainer.state.global_step, "giay": round(giay),
        "giay_moi_buoc": round(giay / max(trainer.state.global_step, 1), 2),
        "ket_cuc_luc_train_%": {k: round(100 * v / tong, 1) for k, v in sorted(dem.items())},
        "log_cuoi": trainer.state.log_history[-3:],
    }
