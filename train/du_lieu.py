"""Biến JSONL thành tensor để train — chỗ quyết định MASK LOSS.

Nguyên tắc quan trọng nhất ở đây: **loss chỉ tính trên phần model phải SINH RA**.

    <|im_start|>user\\nMở cốp xe<|im_end|>\\n<|im_start|>assistant\\n   <- prompt, KHÔNG tính
    <nexa_11>()<nexa_end><|im_end|>                                    <- completion, TÍNH

Nếu tính loss cả trên prompt, model sẽ dành một phần năng lực để học cách "sinh ra câu
người dùng nói" — việc hoàn toàn vô dụng, và làm loãng tín hiệu học phần thật sự cần.

Cách mask: token nào không tính loss thì đặt nhãn = -100 (quy ước của PyTorch:
CrossEntropyLoss bỏ qua mọi vị trí có nhãn -100).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BO_QUA = -100  # nhãn đặc biệt: PyTorch bỏ qua vị trí này khi tính loss


def doc_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tao_mau(tokenizer: Any, row: dict[str, Any], do_dai_toi_da: int = 512) -> dict[str, list[int]]:
    """1 dòng JSONL -> {input_ids, labels, attention_mask}.

    Cắt bớt nếu quá dài (data của dự án rất ngắn, ~100-150 token, nên hiếm khi chạm trần).
    """
    id_prompt = tokenizer.encode(row["prompt"], add_special_tokens=False)
    id_completion = tokenizer.encode(row["completion"], add_special_tokens=False)

    input_ids = (id_prompt + id_completion)[:do_dai_toi_da]
    # nhãn: bỏ qua phần prompt, chỉ học phần completion
    labels = ([BO_QUA] * len(id_prompt) + id_completion)[:do_dai_toi_da]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


def dem_token_hoc(mau: dict[str, list[int]]) -> int:
    """Số token thực sự được tính loss — dùng để kiểm mask có đúng không."""
    return sum(1 for x in mau["labels"] if x != BO_QUA)


class GhepLo:
    """Gom nhiều mẫu thành 1 lô, đệm cho bằng độ dài mẫu dài nhất trong lô.

    Đệm theo LÔ chứ không đệm tất cả về 512: data dài ngắn rất khác nhau, đệm cố định
    làm GPU tính trên hàng đống ô trống vô ích.
    """

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, mau_list: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        dai_nhat = max(len(m["input_ids"]) for m in mau_list)
        lo: dict[str, list[list[int]]] = {"input_ids": [], "labels": [], "attention_mask": []}
        for m in mau_list:
            thieu = dai_nhat - len(m["input_ids"])
            lo["input_ids"].append(m["input_ids"] + [self.pad_token_id] * thieu)
            # phần đệm KHÔNG được tính loss, và cũng không được nhìn thấy
            lo["labels"].append(m["labels"] + [BO_QUA] * thieu)
            lo["attention_mask"].append(m["attention_mask"] + [0] * thieu)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in lo.items()}


def thong_ke_do_dai(tokenizer: Any, rows: list[dict[str, Any]], mau_kiem: int = 500) -> dict[str, Any]:
    """Đo độ dài thật của data — để chọn `do_dai_toi_da` cho hợp lý, đỡ phí VRAM."""
    do_dai = []
    hoc = []
    for r in rows[:mau_kiem]:
        m = tao_mau(tokenizer, r, do_dai_toi_da=10_000)
        do_dai.append(len(m["input_ids"]))
        hoc.append(dem_token_hoc(m))
    do_dai.sort()
    return {
        "so_mau_kiem": len(do_dai),
        "trung_binh": round(sum(do_dai) / len(do_dai), 1),
        "p95": do_dai[int(len(do_dai) * 0.95)],
        "dai_nhat": do_dai[-1],
        "token_tinh_loss_trung_binh": round(sum(hoc) / len(hoc), 1),
    }
