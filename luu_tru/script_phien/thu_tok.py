import json
import sys

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")
print("transformers", transformers.__version__)
M = sys.argv[1]
rows = [json.loads(x) for x in open("data/processed/final_v4_doi_chung/val_vi.jsonl", encoding="utf-8")]
cau = [r for r in rows if r["meta"]["type"] == "call" and ("trip_km" in r["completion"] or "level=" in r["completion"])][:4]
tok = AutoTokenizer.from_pretrained(M)
print("tách 'trip_km=290':", tok.tokenize('(trip_km=290.0)'))
print("số token prompt 1:", len(tok(cau[0]["prompt"], add_special_tokens=False)["input_ids"]))
end = [tok.convert_tokens_to_ids("<nexa_end>"), tok.eos_token_id]
m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32)
for r in cau:
    vao = tok(r["prompt"], return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        ra = m.generate(**vao, max_new_tokens=40, do_sample=False, eos_token_id=end, pad_token_id=tok.pad_token_id)
    print("  ", tok.decode(ra[0][vao["input_ids"].shape[1]:]), "| cần", r["completion"][:50])
