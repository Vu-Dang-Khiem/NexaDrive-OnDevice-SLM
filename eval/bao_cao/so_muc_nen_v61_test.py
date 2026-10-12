"""(3) + (4) Rà soát 2026-10-12: v6.1 hạt giống 42 ở BA mức nén GGUF (f16 = chưa nén, q8_0, q4_K_M) trên đúng 701 câu
của 6 tập test đóng băng, chạy LẦN LƯỢT cùng máy, cùng điều kiện (CPU 8 luồng, greedy, 1 khe llama-server) bằng
`eval/do_tu_tin_gguf.py` -> runs/gguf/do_tre/test_v61s42_{f16,q8_0,q4_k_m}.jsonl. Chỉ báo cáo — bản triển khai vẫn là
q4 + 0,80 (chọn theo luật trên val); không chọn / đổi gì theo số test này.
"""
import math
import sys
from statistics import median

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from eval.metrics import DUNG
from eval.so_muc_nen import doc, kq_nguong

SAI_LEN_XE = {"goi_bua", "sai_ham", "sai_tham_so", "thieu_hoi_lai"}
MUC = [("f16", "f16"), ("q8_0", "q8_0"), ("q4_K_M", "q4_k_m")]
R = {ten: doc(f"runs/gguf/do_tre/test_v61s42_{f}.jsonl") for ten, f in MUC}


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def nhom(rows, tap, loc=None):
    return [d for d in rows if d["tap"] == tap and (loc is None or loc(d))]


DONG = [
    ("Rõ ràng (74)", "test_vi_sach_v2", lambda d: d["nhom"] == "ro_rang"),
    ("Mơ hồ (39)", "test_vi_sach_v2", lambda d: d["nhom"] == "mo_ho"),
    ("Ngoài phạm vi (117)", "test_vi_sach_v2", lambda d: d["nhom"] == "ngoai_pham_vi"),
    ("Bổ sung (28)", "test_vi_bo_sung", None),
    ("Mơ hồ dài (129)", "test_vi_mo_ho_4", None),
    ("Câu ngắn + cách gọi (120)", "test_v6_mot_luot", lambda d: d["nhom_phu"] in ("v6_A", "v6_B")),
    ("Tham số mới (30)", "test_v6_mot_luot", lambda d: d["nhom_phu"] == "v6_C"),
    ("Hai lượt (60)", "test_v6_hai_luot", None),
    ("Gõ sai (104)", "test_v6_nhieu", None),
]

for nguong, ten_ng in ((0.0, "KHÔNG ngưỡng"), (0.80, "ngưỡng 0,80")):
    print(f"\n=== {ten_ng} — % đúng (701 câu)")
    print(f"{'':28}" + "".join(f"{t:>9}" for t, _ in MUC))
    for ten, tap, loc in DONG:
        print(f"{ten:28}" + "".join(
            f"{100 * sum(kq_nguong(d, nguong) == DUNG for d in nhom(R[t], tap, loc)) / len(nhom(R[t], tap, loc)):9.1f}"
            for t, _ in MUC))
    sai = []
    for t, _ in MUC:
        k = [kq_nguong(d, nguong) for d in nhom(R[t], "test_vi_sach_v2")]
        sai.append(sum(x in SAI_LEN_XE for x in k))
    print(f"{'Sai lên xe (test sạch)':28}" + "".join(f"{s:9d}" for s in sai))

print("\n=== McNemar so với f16 (chưa nén), không ngưỡng — b: bản nén đúng / f16 sai, c: ngược lại")
for t in ("q8_0", "q4_K_M"):
    for ten, tap, loc in DONG:
        g = nhom(R["f16"], tap, loc)
        q = nhom(R[t], tap, loc)
        assert [d["cau"] for d in g] == [d["cau"] for d in q], (t, ten)
        b = sum(y["ket_qua"] == DUNG and x["ket_qua"] != DUNG for x, y in zip(g, q))
        c = sum(x["ket_qua"] == DUNG and y["ket_qua"] != DUNG for x, y in zip(g, q))
        print(f"  {t:7} {ten:28} hơn {b:2d} / kém {c:2d}, p = {mcnemar(b, c):.3f}")
    giong = sum(x["output"] == y["output"] for x, y in zip(R["f16"], R[t]))
    print(f"  {t:7} output giống f16: {giong}/{len(R[t])}")

print("\n=== Độ trễ lượt 1 (đọc prompt + sinh), CPU 8 luồng, cùng phiên, 701 câu")
for t, _ in MUC:
    ms = sorted(d["ms_doc_prompt"] + d["ms_sinh"] for d in R[t])
    print(f"  {t:7} p50 {median(ms):5.0f} ms · p95 {ms[int(0.95 * len(ms)) - 1]:5.0f} ms · lớn nhất {ms[-1]:5.0f} ms")
