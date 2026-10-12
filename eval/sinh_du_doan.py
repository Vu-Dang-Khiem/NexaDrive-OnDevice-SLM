"""Cho model chạy qua tập test → file dự đoán để `eval/run_eval.py` chấm.

Tách khỏi phần chấm điểm: bước này cần GPU và model, bước chấm thì không. Nhờ vậy chấm
lại được bao nhiêu lần tuỳ thích mà không phải bật máy, và so nhiều model chỉ bằng cách
đổi file dự đoán.

Lưu ý khi sinh:
  • **Đệm bên TRÁI** khi chạy theo lô. Model sinh tiếp từ token cuối cùng, nếu đệm bên
    phải thì token cuối là ô trống → model sinh tiếp từ khoảng trắng, ra kết quả rác.
  • **Greedy** (không lấy mẫu ngẫu nhiên): lúc chạy thật cũng phải như vậy — cùng một câu
    lệnh phải luôn ra cùng một lời gọi hàm, không được hôm nay mở cốp mai đóng cốp.
  • Dừng sớm ở `<nexa_end>`: đỡ tốn thời gian sinh thừa.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from schema.loader import CATALOG


def cat_output(text: str) -> str:
    """Bỏ phần thừa sau khi model đã sinh xong lời gọi.

    Model hay sinh tiếp `<|im_end|>` rồi lảm nhảm thêm; chỉ giữ tới hết lời gọi.
    """
    t = text.strip()
    for dau in ("<|im_end|>", "<|endoftext|>"):
        if dau in t:
            t = t.split(dau, 1)[0]
    ket = CATALOG.nexa_end
    if ket in t:
        t = t.split(ket, 1)[0] + ket
    return t.strip()


def sinh_hang_loat(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int = 64,
    batch_size: int = 16,
) -> tuple[list[str], list[float]]:
    """Trả về (danh sách output đã cắt, thời gian sinh từng câu tính bằng giây)."""
    import torch

    tokenizer.padding_side = "left"  # xem docstring đầu file
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dừng NGAY ở <nexa_end> (như paper và như runtime sẽ làm), không chờ tới <|im_end|>.
    # Cần cho model học "mô tả hàm" sau <nexa_end>: không dừng thì nó sinh thêm cả đoạn mô tả
    # -> độ trễ đo sai.
    # Model cũ không đổi output: phần sau <nexa_end> vốn đã bị `cat_output` cắt bỏ.
    dung_o = [tokenizer.convert_tokens_to_ids(CATALOG.nexa_end), tokenizer.eos_token_id]

    ket_qua: list[str] = []
    do_tre: list[float] = []
    model.eval()
    for i in range(0, len(prompts), batch_size):
        lo = prompts[i : i + batch_size]
        vao = tokenizer(lo, return_tensors="pt", padding=True, add_special_tokens=False)
        vao = {k: v.to(model.device) for k, v in vao.items()}

        t0 = time.perf_counter()
        with torch.no_grad():
            ra = model.generate(
                **vao,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy — cùng câu lệnh phải ra cùng kết quả
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=dung_o,
            )
        giay = (time.perf_counter() - t0) / len(lo)

        dai_vao = vao["input_ids"].shape[1]
        for dong in ra:
            sinh = tokenizer.decode(dong[dai_vao:], skip_special_tokens=False)
            ket_qua.append(cat_output(sinh))
            do_tre.append(giay)
        print(f"  đã sinh {len(ket_qua)}/{len(prompts)}")
    return ket_qua, do_tre


def thong_ke_do_tre(do_tre: list[float]) -> dict[str, float]:
    if not do_tre:
        return {}
    s = sorted(do_tre)
    return {
        "p50_ms": round(s[len(s) // 2] * 1000, 1),
        "p95_ms": round(s[int(len(s) * 0.95)] * 1000, 1),
        "trung_binh_ms": round(sum(s) / len(s) * 1000, 1),
    }


def chay(
    model_dir: str,
    test_file: str,
    out_file: str,
    batch_size: int = 16,
    tokenizer_dir: str | None = None,
) -> dict[str, Any]:
    """`tokenizer_dir`: checkpoint theo epoch do `Trainer` lưu KHÔNG kèm tokenizer, nên
    khi chấm các bản đó phải trỏ sang thư mục model đã thêm token."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with open(test_file, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir or model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map="auto"
    )

    outs, do_tre = sinh_hang_loat(model, tokenizer, [r["prompt"] for r in rows], batch_size=batch_size)

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for r, o in zip(rows, outs, strict=True):
            f.write(json.dumps({"prompt": r["prompt"], "output": o}, ensure_ascii=False) + "\n")

    bao_cao = {"so_cau": len(rows), "do_tre": thong_ke_do_tre(do_tre), "ghi_vao": out_file}
    print(json.dumps(bao_cao, ensure_ascii=False, indent=2))
    return bao_cao


def main() -> None:
    ap = argparse.ArgumentParser(description="Cho model sinh dự đoán trên tập test")
    ap.add_argument("--model", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--tokenizer", default=None, help="dùng khi model_dir là checkpoint epoch")
    args = ap.parse_args()
    chay(args.model, args.test, args.output, args.batch_size, args.tokenizer)


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
