import collections
import json
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from eval.metrics import doc_output, nhom_cua_dap_an
from gen.records import query_from_call_prompt as q

R = [json.loads(x) for x in open("data/processed/final_v4_doi_chung/train_vi.jsonl", encoding="utf-8")]
c = collections.Counter()
dai = collections.defaultdict(list)
for r in R:
    t = r["meta"]["type"]
    if t == "reply":
        c["reply (lượt 2: trả lời)"] += 1
        continue
    nhom = nhom_cua_dap_an(doc_output(r["completion"]))
    ten = {"ro_rang": "gọi hàm", "mo_ho": "hỏi lại", "ngoai_pham_vi": "từ chối"}[nhom]
    c[ten] += 1
    n = len(q(r["prompt"]).split())
    dai[ten].append(n)
for k, v in c.most_common():
    print(f"{k:25} {v:6d}  ({100*v/len(R):4.1f}%)")
print("tổng", len(R))
print("\nđộ dài câu (chữ): ≤10 / >10 (kể hoàn cảnh)")
for k, xs in dai.items():
    print(f"  {k:10} ngắn {sum(x <= 10 for x in xs):5d} | dài {sum(x > 10 for x in xs):5d} | trung vị {sorted(xs)[len(xs)//2]}")
