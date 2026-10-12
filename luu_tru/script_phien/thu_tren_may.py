"""Chạy model tải về trên MÁY (CPU) với 10 câu dev, so output + xác suất token đầu với bản đo trên Modal."""
import json
import sys
import time

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.do_tu_tin import do

MD = "models/lora-v5-s7-ck2416"
modal = {d["cau"]: d for d in map(json.loads, open("runs/tu_tin/test_lorav5_s7.jsonl", encoding="utf-8"))
         if d["tap"] == "dev_vi_2"}
rows = [json.loads(x) for x in open("data/dev_tay_2/dev_vi_2.jsonl", encoding="utf-8")][:10]
tok = AutoTokenizer.from_pretrained(MD)
t0 = time.time()
m = AutoModelForCausalLM.from_pretrained(MD, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda")
print(f"nạp model {time.time() - t0:.0f}s; tie_word_embeddings={m.config.tie_word_embeddings}")
t0 = time.time()
kq = do(m, tok, rows, batch_size=5)
print(f"10 câu trên CPU: {time.time() - t0:.0f}s")
khop = 0
for r in kq:
    mo = modal[r["cau"]]
    ok = r["output"] == mo["output"]
    khop += ok
    print(f"{'KHỚP' if ok else 'LỆCH'} p={r['p_buoc_1']:.3f} (Modal {mo['p_buoc_1']:.3f}) | {r['cau'][:45]} -> {r['output'][:60]}")
print(f"==> {khop}/10 output trùng Modal")
