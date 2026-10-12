import json
import random
import sys

sys.stdout.reconfigure(encoding="utf-8")
R = [json.loads(x) for x in open("data/processed/v5/v5thu_hoi_lai_hc_chi_tiet.jsonl", encoding="utf-8")]
giu = [r for r in R if not r["loai_vi"]]
lech = [r for r in giu if r["fn_dich"] not in r["calls"]]
print(f"giữ {len(giu)} | lệch hàm đích {len(lech)}\n")
print("=== LỆCH (đích -> teacher gán) ===")
for r in lech:
    print(f"  [{r['fn_dich']} -> {','.join(r['calls'])}] {r['query']}")
print("\n=== bị loại ở code / teacher ===")
for r in R:
    if r["loai_vi"] and r["loai_vi"] != "cung_khuon":
        print(f"  ({r['loai_vi']}) [{r['fn_dich']}] {r['query']}")
print("\n=== 30 câu GIỮ & khớp đích, ngẫu nhiên ===")
khop = [r for r in giu if r["fn_dich"] in r["calls"]]
random.Random(1).shuffle(khop)
for r in khop[:30]:
    print(f"  [{','.join(r['calls'])}] ({len(r['query'].split())} chữ) {r['query']}")
