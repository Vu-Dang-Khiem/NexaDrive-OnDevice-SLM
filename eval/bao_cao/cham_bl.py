import json
import sys

sys.stdout = open(sys.argv[1], "w", encoding="utf-8")
sys.path.insert(0, ".")
from eval.metrics import cham_mot_cau, cham_toan_bo, doc_output, nhom_cua_dap_an
from gen.records import query_from_call_prompt as q


def doc(p):
    return [json.loads(x) for x in open(p, encoding="utf-8")]


def du(p):
    return {r["prompt"]: r["output"] for r in doc(p)}


sach = doc("data/processed/final/test_vi_sach_v2.jsonl")
bs = doc("data/processed/final/test_vi_bo_sung.jsonl")
dev = doc("data/dev_tay_2/dev_vi_2.jsonl")


def cham_dev(D):
    tc = ld = 0
    for r in dev:
        out = D[r["prompt"]]
        if r["meta"]["nhom_phu"].startswith("dev_lam_duoc"):
            p, d = doc_output(out), doc_output(r["completion"])
            ld += p.doc_duoc and p.ten_ham[:1] == d.ten_ham[:1]
        else:
            tc += cham_mot_cau(out, r["completion"])[0] == "dung"
    return tc, ld


M = {
    "sip_72b": {t: f"runs/baseline/sip_72b_{t}.jsonl" for t in ("test_vi_sach_v2", "test_vi_bo_sung", "dev_vi_2")},
    "sip_1_5b": {t: f"runs/baseline/sip_1_5b_{t}.jsonl" for t in ("test_vi_sach_v2", "test_vi_bo_sung", "dev_vi_2")},
    "student_vllm": {t: f"runs/baseline/student_{t}.jsonl" for t in ("test_vi_sach_v2", "test_vi_bo_sung", "dev_vi_2")},
    "student_hf_s42": {t: f"runs/student-v4dc-s42/du_doan_{t}.jsonl" for t in ("test_vi_sach_v2", "test_vi_bo_sung", "dev_vi_2")},
}
for ten, f in M.items():
    a = cham_toan_bo(sach, du(f["test_vi_sach_v2"]))
    b = cham_toan_bo(bs, du(f["test_vi_bo_sung"]))
    tc, ld = cham_dev(du(f["dev_vi_2"]))
    n, e = a["theo_nhom"], a["ty_le_loi_%"]
    print(f"{ten:15} rõ ràng {n['ro_rang']['dung_%']:5.1f} | mơ hồ {n['mo_ho']['dung_%']:5.1f} | ngoài pv {n['ngoai_pham_vi']['dung_%']:5.1f}"
          f" | gọi bừa {e['goi_bua']:4.1f} | từ chối nhầm {e['tu_choi_nham']:4.1f} | hỏi thừa {e['hoi_lai_thua']:4.1f}"
          f" | 28 câu {b['theo_nhom']['mo_ho']['dung_%']:5.1f} | dev từ chối {tc}/20 làm được {ld}/10")
    print("      kiểu sai:", a["loi_theo_kieu"])
    print("      nhóm phụ ngoài pv:", {k: v["dung_%"] for k, v in a["theo_nhom_phu"].items() if k.startswith("oos")})

# đồng nhất student vLLM vs HF
A, B = du(M["student_vllm"]["test_vi_sach_v2"]), du(M["student_hf_s42"]["test_vi_sach_v2"])
khac = [r for r in sach if cham_mot_cau(A[r["prompt"]], r["completion"])[0] != cham_mot_cau(B[r["prompt"]], r["completion"])[0]]
print("\nstudent vLLM vs HF khác kết quả:", len(khac))

# mẫu câu 72B sai
print("\n=== 72B sai (rõ ràng + mơ hồ), 20 câu đầu ===")
D = du(M["sip_72b"]["test_vi_sach_v2"])
R = {r["prompt"]: r["raw"] for r in doc(M["sip_72b"]["test_vi_sach_v2"])}
k = 0
for r in sach:
    kq = cham_mot_cau(D[r["prompt"]], r["completion"])[0]
    if kq != "dung" and nhom_cua_dap_an(doc_output(r["completion"])) != "ngoai_pham_vi" and k < 20:
        k += 1
        print(f"[{kq}] {q(r['prompt'])[:55]} | cần {r['completion'][:40]} | ra {R[r['prompt']][:90]}")
print("\n=== 72B gọi bừa, 12 câu ===")
k = 0
for r in sach:
    if cham_mot_cau(D[r["prompt"]], r["completion"])[0] == "goi_bua" and k < 12:
        k += 1
        print(f"{q(r['prompt'])[:60]} | ra {R[r['prompt']][:90]}")
