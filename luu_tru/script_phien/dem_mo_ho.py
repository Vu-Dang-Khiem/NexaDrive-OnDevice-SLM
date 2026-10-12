import collections
import json
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from eval.metrics import doc_output, nhom_cua_dap_an

for p in ("data/processed/final/test_vi_sach_v2.jsonl", "data/processed/final/test_vi_bo_sung.jsonl",
          "data/processed/final_v4_doi_chung/train_vi.jsonl"):
    rows = [json.loads(x) for x in open(p, encoding="utf-8")]
    c = collections.Counter()
    for r in rows:
        if "<nexa_" not in r["completion"]:
            continue
        d = doc_output(r["completion"])
        if nhom_cua_dap_an(d) == "mo_ho":
            for h in set(d.ten_ham):
                c[h] += 1
    print(p, sum(1 for r in rows if "<nexa_ask>" in r["completion"]), "câu mơ hồ")
    print("  ", dict(c.most_common()))
