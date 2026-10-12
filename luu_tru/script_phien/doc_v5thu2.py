import collections
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
R = [json.loads(x) for x in open("data/processed/v5/v5thu2_hoi_lai_hc_chi_tiet.jsonl", encoding="utf-8")]
print("theo hàm đích -> lý do:")
c = collections.defaultdict(collections.Counter)
for r in R:
    c[r["fn_dich"]][r["loai_vi"] or "GIU"] += 1
for h, v in sorted(c.items()):
    print(f"  {h:18} {dict(v)}")
print("\n=== operate_sunroof (mọi câu) ===")
for r in R:
    if r["fn_dich"] == "operate_sunroof":
        print(f"  ({r['loai_vi'] or 'GIU'}) {r.get('calls', '')} {r['query']}")
print("\n=== lệch hàm đích ===")
for r in R:
    if r["loai_vi"] == "lech_ham_dich":
        print(f"  [{r['fn_dich']} -> {','.join(r['calls'])}] {r['query']}")
print("\n=== tiếng Anh bị bắt ===")
for r in R:
    if r["loai_vi"] == "tieng_anh":
        print("  ", r["query"])
print("\n=== TOÀN BỘ câu GIỮ ===")
for r in sorted((r for r in R if not r["loai_vi"]), key=lambda r: r["fn_dich"]):
    print(f"  [{','.join(r['calls'])}] {r['query']}")
