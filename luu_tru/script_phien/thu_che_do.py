import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")
M = "models/student-v4dc-s42-ck2206"
rows = [json.loads(x) for x in open("data/processed/final_v4_doi_chung/val_vi.jsonl", encoding="utf-8")]
cau = [r for r in rows if r["meta"]["type"] == "call" and ("trip_km" in r["completion"] or "level=" in r["completion"])][:4]
tok = AutoTokenizer.from_pretrained(M)
end = [tok.convert_tokens_to_ids("<nexa_end>"), tok.eos_token_id]


def chay(model, ten):
    print("==", ten)
    for r in cau:
        vao = tok(r["prompt"], return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            ra = model.generate(**vao, max_new_tokens=40, do_sample=False, eos_token_id=end, pad_token_id=tok.pad_token_id)
        print("  ", tok.decode(ra[0][vao["input_ids"].shape[1]:]), "| cần", r["completion"][:50])


for dtype, dev, attn in [(torch.bfloat16, "cuda", "sdpa"), (torch.float16, "cuda", "sdpa"), (torch.bfloat16, "cuda", "eager"), (torch.float32, "cpu", "sdpa")]:
    m = AutoModelForCausalLM.from_pretrained(M, dtype=dtype, attn_implementation=attn).to(dev)
    chay(m, f"{dtype} {dev} {attn}")
    del m
    torch.cuda.empty_cache()
