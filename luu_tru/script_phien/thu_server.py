import json
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("models/lora-v5-s7-ck2416")
row = json.loads(open("data/dev_tay_2/dev_vi_2.jsonl", encoding="utf-8").readline())
ids = tok(row["prompt"], add_special_tokens=False)["input_ids"]
body = {"prompt": ids, "n_predict": 64, "temperature": 0, "n_probs": 5, "return_tokens": True, "cache_prompt": False}
req = urllib.request.Request("http://127.0.0.1:8088/completion", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
r = json.loads(urllib.request.urlopen(req).read())
print("keys:", list(r.keys()))
print("content:", repr(r.get("content")))
print("tokens:", r.get("tokens"), "->", repr(tok.decode(r.get("tokens") or [], skip_special_tokens=False)))
cp = r.get("completion_probabilities") or []
print("so buoc probs:", len(cp))
print("buoc 1:", json.dumps(cp[0], ensure_ascii=False)[:600] if cp else None)
print("timings:", r.get("timings"))
print("stop_type:", r.get("stop_type"), r.get("stopping_word"))
