import collections
import json
import sys

sys.path.insert(0, ".")
from eval.metrics import doc_output, nhom_cua_dap_an

for p in [
    "data/processed/final_v4_doi_chung/val_mau.jsonl",
    "data/processed/final_v4_doi_chung/val_vi.jsonl",
    "data/dev_tay_2/dev_vi_2.jsonl",
]:
    rows = [json.loads(x) for x in open(p, encoding="utf-8")]
    print(p, len(rows), collections.Counter(r.get("meta", {}).get("type") for r in rows))
    print("   keys", list(rows[0].keys()), rows[0].get("meta"))
    goi = [r for r in rows if "<nexa_" in r["completion"]]
    print("   nhom", collections.Counter(nhom_cua_dap_an(doc_output(r["completion"])) for r in goi))
