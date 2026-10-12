"""v6.1 với BA hạt giống (42, 7, 2026) — HF không ngưỡng, 6 tập test đóng băng. Chỉ để ước lượng độ dao động giữa các
lần train (PLAN.md "HẠT GIỐNG THỨ BA CHO v6.1", khoá trước khi train 2026-10-12); không chọn / đổi gì theo số này.
Đọc dự đoán đã lưu ở runs/v6/du_doan_v61<hạt>_<tập>.jsonl.
"""
import json
import sys
from collections import Counter
from statistics import mean, stdev

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from eval.metrics import DUNG, cham_mot_cau, doc_output, nhom_cua_dap_an

SAI_LEN_XE = {"goi_bua", "sai_ham", "sai_tham_so", "thieu_hoi_lai"}
HAT = ("s42", "s7", "s2026")
PHEP = [
    ("test_vi_sach_v2", "ro_rang", "Rõ ràng (74)"),
    ("test_vi_sach_v2", "mo_ho", "Mơ hồ (39)"),
    ("test_vi_sach_v2", "ngoai_pham_vi", "Ngoài phạm vi (117)"),
    ("test_vi_bo_sung", None, "Bổ sung (28)"),
    ("test_vi_mo_ho_4", None, "Mơ hồ dài (129)"),
    ("test_v6_mot_luot", ("v6_A", "v6_B"), "Câu ngắn + cách gọi (120)"),
    ("test_v6_mot_luot", ("v6_C",), "Tham số mới (30)"),
    ("test_v6_hai_luot", None, "Hai lượt (60)"),
    ("test_v6_nhieu", None, "Gõ sai (104)"),
]


def tai_s2026():
    """Tải dự đoán test của s2026 từ volume (kiểm kích thước — đường tải hay cắt cụt âm thầm)."""
    from pathlib import Path

    import modal

    vol = modal.Volume.from_name("nexa-drive")
    kt = {e.path.rsplit("/", 1)[-1]: e.size for e in vol.listdir("/data/vi/final_v6")}
    for tap in dict.fromkeys(t for t, _, _ in PHEP):
        f = f"du_doan_v61s2026_{tap}.jsonl"
        if Path("runs/v6", f).exists() and Path("runs/v6", f).stat().st_size == kt[f]:
            continue
        for _ in range(6):
            du = b"".join(vol.read_file(f"/data/vi/final_v6/{f}"))
            if len(du) == kt[f]:
                break
        else:
            raise RuntimeError(f"tải {f} không đủ")
        Path("runs/v6", f).write_bytes(du)


if "--tai" in sys.argv:
    tai_s2026()


def doc(p):
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def ket_cuc(hat, tap):
    du = {r["prompt"]: r["output"] for r in doc(f"runs/v6/du_doan_v61{hat}_{tap}.jsonl")}
    rows = doc(f"data/processed/final/{tap}.jsonl")
    assert len(du) == len(rows), (hat, tap, len(du), len(rows))
    kq = [cham_mot_cau(du[r["prompt"]], r["completion"])[0] for r in rows]
    nhom = [r["meta"].get("nhom_phu") if tap == "test_v6_mot_luot" else nhom_cua_dap_an(doc_output(r["completion"]))
            for r in rows]
    return kq, nhom


print(f"{'v6.1, HF không ngưỡng':28}" + "".join(f"{h:>8}" for h in HAT) + f"{'tb':>8}{'đlc':>7}{'tb s42+s7':>11}")
for tap, loc, ten in PHEP:
    tl = []
    for h in HAT:
        kq, nh = ket_cuc(h, tap)
        chon = [k for k, g in zip(kq, nh) if loc is None or g == loc or (isinstance(loc, tuple) and g in loc)]
        tl.append(100 * sum(k == DUNG for k in chon) / len(chon))
    print(f"{ten:28}" + "".join(f"{x:8.1f}" for x in tl) + f"{mean(tl):8.1f}{stdev(tl):7.1f}{mean(tl[:2]):11.1f}")

sai = []
for h in HAT:
    dem = Counter(ket_cuc(h, "test_vi_sach_v2")[0])
    sai.append(sum(dem[k] for k in SAI_LEN_XE))
    print(f"  {h:6} sai lên xe {sai[-1]:3d}  ({', '.join(f'{k} {dem[k]}' for k in sorted(SAI_LEN_XE))})")
print(f"  tb {mean(sai):.1f} ± {stdev(sai):.1f}")
