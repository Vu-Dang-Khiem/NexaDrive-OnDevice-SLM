"""Đo ĐỘ TỰ TIN của model ở token quyết định — chuẩn bị cho ngưỡng từ chối (2026-10-02).

Với functional token, token đầu tiên model sinh ra CHÍNH LÀ quyết định: `<nexa_i>` (gọi hàm i,
kể cả hàm từ chối `irrelevant_function`) hoặc `<nexa_ask>` (hỏi lại). Xác suất của token đó là
độ tự tin của model. Script này ghi lại, cho từng câu:
  • output greedy + kết quả chấm (đúng / gọi bừa / ...) như `eval/run_eval.py`,
  • top-5 token và xác suất ở bước 1 (và bước 2 nếu bước 1 là `<nexa_ask>`).

Mục đích: so phân bố xác suất của câu ĐÚNG với câu GỌI BỪA. Tách nhau -> ngưỡng có ích;
chồng nhau -> ngưỡng không cứu được (model sai mà vẫn tự tin).

CHỈ chạy trên val / dev. Không chạy trên test khi còn đang chọn ngưỡng (chọn trên test là tự ra
đề tự chấm). Test chấm MỘT lần, sau khi ngưỡng đã khoá.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.metrics import cham_mot_cau, doc_output, nhom_cua_dap_an
from eval.sinh_du_doan import cat_output
from gen.records import query_from_call_prompt
from schema.loader import CATALOG

SO_TOP = 5


def _top(xac_suat: Any, tokenizer: Any) -> list[tuple[str, float]]:
    import torch

    p, idx = torch.topk(xac_suat, SO_TOP)
    return [
        (tokenizer.convert_ids_to_tokens(int(i)), round(float(v), 5)) for v, i in zip(p, idx, strict=True)
    ]


def do(model: Any, tokenizer: Any, rows: list[dict[str, Any]], batch_size: int = 16) -> list[dict[str, Any]]:
    import torch

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dung_o = [tokenizer.convert_tokens_to_ids(CATALOG.nexa_end), tokenizer.eos_token_id]
    id_ask = tokenizer.convert_tokens_to_ids(CATALOG.nexa_ask)

    ket_qua: list[dict[str, Any]] = []
    model.eval()
    for i in range(0, len(rows), batch_size):
        lo = rows[i : i + batch_size]
        vao = tokenizer(
            [r["prompt"] for r in lo], return_tensors="pt", padding=True, add_special_tokens=False
        )
        vao = {k: v.to(model.device) for k, v in vao.items()}
        with torch.no_grad():
            ra = model.generate(
                **vao,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=dung_o,
                output_scores=True,
                return_dict_in_generate=True,
            )
        dai_vao = vao["input_ids"].shape[1]
        # scores[k]: logits ở bước sinh thứ k, cho cả lô. Greedy không có bộ xử lý logits nào
        # -> softmax ra đúng xác suất của model.
        b0 = torch.softmax(ra.scores[0].float(), dim=-1)
        b1 = torch.softmax(ra.scores[1].float(), dim=-1) if len(ra.scores) > 1 else None
        for j, r in enumerate(lo):
            out = cat_output(tokenizer.decode(ra.sequences[j][dai_vao:], skip_special_tokens=False))
            tok0 = int(ra.sequences[j][dai_vao])
            dong = {
                "cau": query_from_call_prompt(r["prompt"]),
                "nhom": nhom_cua_dap_an(doc_output(r["completion"])),
                "nhom_phu": r.get("meta", {}).get("nhom_phu"),
                "dap_an": r["completion"],
                "output": out,
                "ket_qua": cham_mot_cau(out, r["completion"])[0],
                "p_buoc_1": round(float(b0[j, tok0]), 5),
                "top_buoc_1": _top(b0[j], tokenizer),
            }
            if tok0 == id_ask and b1 is not None:
                tok1 = int(ra.sequences[j][dai_vao + 1])
                dong["p_buoc_2"] = round(float(b1[j, tok1]), 5)
                dong["top_buoc_2"] = _top(b1[j], tokenizer)
            ket_qua.append(dong)
        print(f"  đã đo {len(ket_qua)}/{len(rows)}")
    return ket_qua


def chay(
    model_dir: str, tap: list[str], out_file: str, batch_size: int = 16, cho_phep_test: bool = False
) -> None:
    """`cho_phep_test=True` CHỈ dùng sau khi ngưỡng đã khoá trong PLAN.md (chấm test một lần)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    for p in tap:
        if "test" in Path(p).name and not cho_phep_test:
            raise SystemExit(f"Không đo trên tập test khi đang chọn ngưỡng: {p}")

    rows: list[dict[str, Any]] = []
    for p in tap:
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                # chỉ mẫu lượt 1 (gọi hàm / hỏi lại); mẫu `reply` là lượt 2, không có token quyết định
                if r.get("meta", {}).get("type") in ("call", "ask"):
                    r.setdefault("meta", {})["tap"] = Path(p).stem
                    rows.append(r)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # không dùng device_map (cần `accelerate`, máy local không cài) — tự đưa lên GPU nếu có
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    kq = do(model, tokenizer, rows, batch_size)

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for r, d in zip(rows, kq, strict=True):
            d["tap"] = r["meta"]["tap"]
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"ghi vào {out_file}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Đo xác suất token quyết định trên val/dev")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tap", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--cho-phep-test", action="store_true", help="chỉ sau khi ngưỡng đã khoá")
    a = ap.parse_args()
    chay(a.model, a.tap, a.output, a.batch_size, a.cho_phep_test)


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
