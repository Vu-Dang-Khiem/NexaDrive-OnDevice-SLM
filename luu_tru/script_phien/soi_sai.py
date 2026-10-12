import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
R = [json.loads(x) for x in open("runs/tu_tin/v4dc_s42_ck2206.jsonl", encoding="utf-8")]
k = 0
for d in R:
    if d["tap"] == "val_vi" and d["ket_qua"] == "sai_tham_so" and k < 12:
        k += 1
        print(d["cau"][:60], "\n   đáp án:", d["dap_an"][:110], "\n   output:", d["output"][:110])
