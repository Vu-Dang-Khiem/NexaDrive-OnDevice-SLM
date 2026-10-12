"""Kiểm 27 functional token trong GGUF: cùng id với tokenizer HF, loại token; và cách cắt token prompt thật trùng HF."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from gguf import GGUFReader
from transformers import AutoTokenizer

from schema.loader import CATALOG

GG = "models/gguf/lora-v61-s42-q4_k_m.gguf"  # lúc đầu kiểm trên v5 (đã xoá); bộ từ vựng mọi bản nén giống nhau
tok = AutoTokenizer.from_pretrained("models/gguf/tokenizer")  # trùng tokenizer models/lora-v5-s7-ck2416 (đã xoá)
r = GGUFReader(GG)
f_tok = r.fields["tokenizer.ggml.tokens"]
f_typ = r.fields["tokenizer.ggml.token_type"]
toks = [bytes(f_tok.parts[i]).decode("utf-8") for i in f_tok.data]
typs = [int(f_typ.parts[i][0]) for i in f_typ.data]
TEN_LOAI = {1: "NORMAL", 2: "UNKNOWN", 3: "CONTROL", 4: "USER_DEFINED", 5: "UNUSED", 6: "BYTE"}
dac_biet = [h.nexa_token for h in CATALOG.functions] + [CATALOG.nexa_end, CATALOG.nexa_ask]
dac_biet = list(dict.fromkeys(dac_biet))
print(f"số token GGUF {len(toks)} · HF {len(tok)} · token chức năng cần kiểm {len(dac_biet)}")
loi = 0
for t in dac_biet:
    id_hf = tok.convert_tokens_to_ids(t)
    id_gg = toks.index(t) if t in toks else None
    ok = id_hf == id_gg
    loi += not ok
    print(f"{'OK ' if ok else 'LỖI'} {t:14} HF {id_hf} · GGUF {id_gg} · loại {TEN_LOAI.get(typs[id_gg]) if id_gg is not None else '-'}")

row = json.loads(open("data/dev_tay_2/dev_vi_2.jsonl", encoding="utf-8").readline())
prompt = row["prompt"]
ids_hf = tok(prompt, add_special_tokens=False)["input_ids"]
p = Path("_prompt_tmp.txt")
p.write_bytes(prompt.encode("utf-8"))  # KHÔNG write_text: Windows đổi \n -> \r\n làm lệch token (PLAN.md mục W)
ra = subprocess.run([str(Path.home() / "llama-bin" / "llama-tokenize.exe"), "-m", GG, "-f", str(p), "--ids",
                     "--log-disable", "--no-bos"], capture_output=True, text=True, encoding="utf-8")
p.unlink()
ids_gg = json.loads(ra.stdout.strip().splitlines()[-1])
print(f"prompt dev: HF {len(ids_hf)} token · GGUF {len(ids_gg)} token · TRÙNG = {ids_hf == ids_gg}")
mau = "<nexa_6>(state=\"on\")<nexa_end>"
print("mẫu đáp án HF:", tok(mau, add_special_tokens=False)["input_ids"][:3], "...")
print("==> ĐẠT" if loi == 0 and ids_hf == ids_gg else "==> CÓ LỖI")
