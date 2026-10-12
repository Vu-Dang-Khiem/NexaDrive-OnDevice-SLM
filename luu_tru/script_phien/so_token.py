import json
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
from transformers import AutoTokenizer

GG = "models/gguf/lora-v5-s7-f16.gguf"
tok = AutoTokenizer.from_pretrained("models/lora-v5-s7-ck2416")
rows = [json.loads(x) for x in open("data/dev_tay_2/dev_vi_2.jsonl", encoding="utf-8")]
print("repr prompt 0:", repr(rows[0]["prompt"]))
for extra in ([], ["--parse-special"]):
    trung = 0
    for i, row in enumerate(rows):
        prompt = row["prompt"]
        ids_hf = tok(prompt, add_special_tokens=False)["input_ids"]
        p = Path("_prompt_tmp.txt")
        p.write_bytes(prompt.encode("utf-8"))
        ra = subprocess.run([str(Path.home() / "llama-bin" / "llama-tokenize.exe"), "-m", GG, "-f", str(p), "--ids",
                             "--log-disable", "--no-bos", *extra], capture_output=True, text=True, encoding="utf-8")
        p.unlink()
        ids_gg = json.loads(ra.stdout.strip().splitlines()[-1])
        trung += ids_hf == ids_gg
        if i == 0 and ids_hf != ids_gg:
            for j, (a, b) in enumerate(zip(ids_hf, ids_gg)):
                if a != b:
                    print(f"  {extra} lệch ở vị trí {j}: HF {a} {tok.convert_ids_to_tokens(a)!r} · GGUF {b} {tok.convert_ids_to_tokens(b)!r}")
            print(f"  độ dài HF {len(ids_hf)} · GGUF {len(ids_gg)}")
    print(f"{extra or 'mặc định'}: {trung}/{len(rows)} prompt trùng HF")
