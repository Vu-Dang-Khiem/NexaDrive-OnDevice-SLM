import json

from transformers import AutoTokenizer

t = AutoTokenizer.from_pretrained("models/student-full-v3")
for tap in ("train_vi", "val_vi"):
    r = [json.loads(x) for x in open(f"data/processed/final_cuoi/{tap}.jsonl", encoding="utf-8")]
    d = sorted(len(t.encode(x["prompt"] + x["completion"])) for x in r)
    print(tap, "token max", d[-1], "p99", d[int(len(d) * 0.99)], ">256", sum(x > 256 for x in d))
