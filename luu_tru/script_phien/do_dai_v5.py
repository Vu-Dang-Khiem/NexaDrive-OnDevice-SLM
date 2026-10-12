import json
import sys

from transformers import AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")
tok = AutoTokenizer.from_pretrained("models/student-v4dc-s42-ck2206")
R = [json.loads(x) for x in open("data/processed/final_v5/train_vi.jsonl", encoding="utf-8")]
L = sorted(len(tok(r["prompt"] + r["completion"], add_special_tokens=False)["input_ids"]) for r in R)
print("số mẫu", len(L), "| tối đa", L[-1], "| p99", L[int(len(L) * 0.99)], "| >256:", sum(x > 256 for x in L))
