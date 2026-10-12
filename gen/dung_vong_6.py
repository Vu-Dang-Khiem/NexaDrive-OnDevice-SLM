"""Vòng 6 — phần chạy Ở MÁY, miễn phí (PLAN.md Phase 7 "KẾ HOẠCH v6").

    python -m gen.dung_vong_6 --luot-1      # chọn câu lượt 1 cho K4 -> data/processed/v6/v6_luot_1.jsonl
    (upload Modal)
    python -m gen.dung_vong_6 --test-nhieu  # bộ test E (bỏ dấu / lỗi gõ) từ test_vi_sach_v2 — sinh TRƯỚC
    train
    python -m gen.dung_vong_6 --dung        # dựng data/processed/final_v6 từ train v5 + K1–K4 (đã tải về) +
    K5

Mọi phép biến đổi bằng code ở đây có seed cố định (tái lập được).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from gen.records import IM_END, query_from_call_prompt
from gen.v6 import HAM_LUOT_1, ap_quy_uoc, ap_quy_uoc_quat, them_nhieu
from schema.call_format import parse_calls, render_call, render_completion
from schema.loader import CATALOG

P = Path("data/processed")
SEED = 2026_10_06
NEN = P / "final_v5"
V6 = P / "v6"


def doc(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def ghi(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _goi(row: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [(c.name, dict(c.args)) for c in parse_calls(row["completion"].replace(IM_END, ""))]


# ── lượt 1 cho K4 ────────────────────────────────────────────────────────────
def chon_luot_1(train: list[dict[str, Any]], so: int, seed: int = SEED) -> list[dict[str, Any]]:
    """Câu gọi hàm (một lời gọi, ≤ 14 chữ) của các hàm điều khiển, chia đều theo hàm."""
    rng = random.Random(seed)
    theo_ham: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in train:
        if r["meta"]["type"] != "call" or r["meta"]["fn"] not in HAM_LUOT_1:
            continue
        goi = _goi(r)
        q = query_from_call_prompt(r["prompt"])
        if len(goi) == 1 and len(q.split()) <= 14:
            theo_ham[goi[0][0]].append({"query": q, "fn": goi[0][0], "args": goi[0][1]})
    moi_ham = max(1, so // len(theo_ham))
    ra = [x for ds in theo_ham.values() for x in rng.sample(ds, min(moi_ham, len(ds)))]
    rng.shuffle(ra)
    return ra[:so]


# ── bộ test E (nhiễu) ────────────────────────────────────────────────────────
def tao_test_nhieu(test: list[dict[str, Any]], seed: int = SEED) -> list[dict[str, Any]]:
    """Câu rõ ràng + mơ hồ của tập test đóng băng -> một bản bỏ dấu / lỗi gõ (nhãn giữ nguyên). Câu có tham số
    chữ (tên nơi, tên bài) bỏ qua: bỏ dấu câu thì nhãn có dấu sẽ không còn khớp chữ người gõ."""
    rng = random.Random(seed)
    ra = []
    for r in test:
        nhom = r["meta"].get("nhom_phu", "")
        if nhom not in ("ro_rang", "mo_ho") or any(isinstance(v, str) and k in ("destination", "query")
                                                 for _, a in _goi(r) for k, v in a.items()):
            continue
        q = query_from_call_prompt(r["prompt"])
        moi, kieu = them_nhieu(q, rng)
        if moi == q:
            continue
        ra.append({**r, "prompt": CATALOG.render_call_prompt(moi),
                   "meta": {**r["meta"], "nhieu": kieu, "cau_goc": q, "nguon_test": "test_vi_sach_v2"}})
    return ra


# ── sửa nhãn cũ theo schema v6 ───────────────────────────────────────────────
_LANH = re.compile(r"\b(lạnh|rét|cóng|buốt|giá rét)\b")
_NHAC_DH = re.compile(r"điều h(?:oà|òa|oa)|máy lạnh|làm lạnh|gió lạnh từ|cửa gió|\bac\b")
_SUOI_DH = re.compile(r"\bsưởi\b")
_SUOI_KHAC = re.compile(r"ghế|vô lăng|gương|kính|tay lái|chân")
_TUONG_DOI = re.compile(r"\b(tăng|giảm|hạ|nâng)\b[^\d]*\bnhiệt\b")


def sua_nhan_v6(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Mẫu cũ -> (mẫu theo schema v6 | None nếu loại, lý do; "" = giữ nguyên). Chỉ mẫu gọi hàm / hỏi lại."""
    if row["meta"]["type"] not in ("call", "ask"):
        return row, ""
    q = query_from_call_prompt(row["prompt"]).lower()
    goi = _goi(row)
    hoi = row["meta"]["type"] == "ask"
    moi = list(goi)
    ly_do = ""
    # (1) than lạnh không nhắc điều hoà -> sưởi (quy ước PLAN.md mục U) ; "bật sưởi" (không phải sưởi ghế...)
    # -> sưởi
    if any(n == "set_ac" and a.get("state") != "off" for n, a in goi) and (
        (_LANH.search(q) and not _NHAC_DH.search(q)) or (_SUOI_DH.search(q) and not _SUOI_KHAC.search(q))
    ):
        moi = [(n, {**a, "mode": "heat"}) if n == "set_ac" and a.get("state") != "off" else (n, a)
               for n, a in moi]
        ly_do = "them_mode_heat"
    # (2) "bật sưởi" mà nhãn từ chối (không phải sưởi ghế / vô lăng / kính...) -> mâu thuẫn với v6 -> loại
    if goi and goi[0][0] == "irrelevant_function" and _SUOI_DH.search(q) and not _SUOI_KHAC.search(q):
        return None, "tu_choi_suoi_nay_lam_duoc"
    # (3) tăng / giảm nhiệt độ KHÔNG số mà nhãn nhiệt độ cụ thể -> delta ±2
    if not re.search(r"\d", q) and _TUONG_DOI.search(q):
        for i, (n, a) in enumerate(moi):
            if n == "set_ac" and "temperature" in a:
                d = -2 if re.search(r"\b(giảm|hạ)\b", q) else 2
                moi[i] = (n, {k: v for k, v in a.items() if k != "temperature"} | {"delta": d})
                ly_do = "nhiet_tuong_doi_thanh_delta"
    # (4) hai lời gọi song song cho hai kính sau / trước cùng hướng -> window="rear" / "front"
    for cap, gop in ((("rear_left", "rear_right"), "rear"), (("front_left", "front_right"), "front")):
        kinh = [(i, a) for i, (n, a) in enumerate(moi) if n == "control_window" and a.get("window") in cap]
        if len(kinh) == 2 and {a["window"] for _, a in kinh} == set(cap):
            a0, a1 = (dict(a) for _, a in kinh)
            a0.pop("window"), a1.pop("window")
            if a0 == a1:
                i0 = kinh[0][0]
                moi = [(n, {**a0, "window": gop}) if i == i0 else (n, a) for i, (n, a) in enumerate(moi)
                       if i != kinh[1][0]]
                ly_do = f"gop_kinh_{gop}"
    # (5) quạt gió tăng / giảm TƯƠNG ĐỐI -> delta (người dùng chốt 2026-10-06); chỉ câu RA LỆNH — câu than
    # phiền (hỏi lại)
    # giữ đề xuất mức tuyệt đối như cũ
    if not hoi:
        for i, (n, a) in enumerate(moi):
            if n == "set_fan_speed" and (b := ap_quy_uoc_quat(q, a)) != a:
                moi[i] = (n, b)
                ly_do = ly_do or "quat_tuong_doi_thanh_delta"
    if not ly_do:
        return row, ""
    return {**row, "completion": render_completion(moi, ask=hoi) + IM_END,
            "meta": {**row["meta"], "sua_v6": ly_do}}, ly_do


# ── ghép lượt trước KHÔNG liên quan (dạy model bỏ qua ngữ cảnh thừa) ─────────
_CAP = {"open_trunk": "close_trunk", "close_trunk": "open_trunk", "lock_doors": "unlock_doors",
        "unlock_doors": "lock_doors"}


def gan_luot_truoc_thua(rows: list[dict[str, Any]], kho_luot_1: list[dict[str, Any]], so: int,
                        seed: int = SEED) -> list[dict[str, Any]]:
    """Bản sao của `so` mẫu cũ (gọi hàm / hỏi lại / từ chối, câu > 3 chữ) + một lời gọi lượt trước của
    hàm KHÁC."""
    rng = random.Random(seed)
    ung_vien = [r for r in rows if r["meta"]["type"] in ("call", "ask")
                and len(query_from_call_prompt(r["prompt"]).split()) > 3]
    ra = []
    for r in rng.sample(ung_vien, min(so, len(ung_vien))):
        fn = r["meta"]["fn"]
        khac = [x for x in kho_luot_1 if x["fn"] != fn and _CAP.get(x["fn"]) != fn]
        t = rng.choice(khac)
        q = query_from_call_prompt(r["prompt"])
        ra.append({**r, "prompt": CATALOG.render_call_prompt(q, render_call(t["fn"], t["args"])),
                   "meta": {**r["meta"], "v6": "luot_truoc_thua", "fn_1": t["fn"]}})
    return ra


# ── K5: nhiễu bằng code ──────────────────────────────────────────────────────
def them_mau_nhieu(rows: list[dict[str, Any]], ti_le: float, seed: int = SEED) -> list[dict[str, Any]]:
    """Bản sao bỏ dấu / lỗi gõ của ~`ti_le` mẫu gọi hàm / hỏi lại / từ chối (KHÔNG có tham số chữ tự do)."""
    rng = random.Random(seed)
    ra = []
    for r in rows:
        if r["meta"]["type"] not in ("call", "ask") or rng.random() >= ti_le:
            continue
        if any(isinstance(v, str) and k in ("destination", "query")
               for _, a in _goi(r) for k, v in a.items()):
            continue
        # mẫu hai lượt: nhiễu CHỈ vào câu người nói, giữ nguyên dòng "Lượt trước: <lời gọi>" (lần dựng đầu áp
        # cả
        # dòng đó -> "Luot truoc: ..." nằm trong câu — lỗi của tôi, đã sửa)
        truoc, q = tach_luot_truoc(query_from_call_prompt(r["prompt"]))
        moi, kieu = them_nhieu(q, rng)
        if moi != q:
            ra.append({**r, "prompt": CATALOG.render_call_prompt(moi, truoc),
                       "meta": {**r["meta"], "v6": f"nhieu_{kieu}"}})
    return ra


def tach_luot_truoc(noi_dung: str) -> tuple[str | None, str]:
    """'Lượt trước: <nexa_11>()\\nđóng đi' -> ('<nexa_11>()', 'đóng đi'); không có lượt trước ->
    (None, câu)."""
    if noi_dung.startswith("Lượt trước: ") and "\n" in noi_dung:
        dau, cau = noi_dung.split("\n", 1)
        return dau[len("Lượt trước: "):], cau
    return None, noi_dung


# ── K1–K4 (tải về từ Modal) -> mẫu train ─────────────────────────────────────
def mau_tu_khoi(ds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ra = []
    for m in ds:
        if not m.get("giu"):
            continue
        # quạt -> quạt kiểu đổi chiều ("giảm về lúc nãy"): không biết lượt trước đổi bao nhiêu -> bỏ
        k4_quat = m["khoi"] == "K4" and m.get("fn_1") == "set_fan_speed"
        if m["fn"] == "set_fan_speed" and k4_quat and m.get("kieu") == "doi_chieu":
            continue
        # quy ước chốt / sửa SAU khi khối đã sinh (quạt `delta`; "một chút" không phải số; âm lượng ±3)
        # -> áp lại ở đây (lặp lại luật đã áp trên Modal không đổi gì)
        args = ap_quy_uoc(m["fn"], m["query"], m.get("args", {}))
        if args is None:
            continue
        truoc = m.get("goi_1") if m["khoi"] == "K4" else None
        if m["khoi"] == "K4" and m.get("fn_1") == "set_fan_speed":  # lời gọi lượt trước cũng theo quy ước mới
            truoc = render_call("set_fan_speed", ap_quy_uoc_quat(m["cau_1"], m["args_1"]))
        ra.append({
            "prompt": CATALOG.render_call_prompt(m["query"], truoc),
            "completion": render_completion([(m["fn"], args)]) + IM_END,
            "meta": {"lang": "vi", "type": "call", "fn": m["fn"], "v6": m["khoi"],
                     **({"kieu": m["kieu"], "fn_1": m["fn_1"]} if m["khoi"] == "K4" else {})},
        })
    return ra


# ── dựng train v6 ────────────────────────────────────────────────────────────
TEST_DONG_BANG = [
    P / "final" / f"{t}.jsonl"
    for t in ("test_vi", "test_vi_sach_v2", "test_vi_bo_sung", "test_vi_mo_ho_3", "test_vi_mo_ho_4",
              "test_v6_mot_luot", "test_v6_hai_luot", "test_v6_nhieu")
]


def _cac_cau(row: dict[str, Any]) -> list[str]:
    """Mọi câu NGƯỜI NÓI trong prompt: câu hiện tại + câu lượt 1 (tập test hai lượt giữ ở meta)."""
    noi_dung = query_from_call_prompt(row["prompt"])
    cau = [noi_dung.split("\n")[-1]]
    if row["meta"].get("cau_1"):
        cau.append(row["meta"]["cau_1"])
    return cau


def bo_nhiem(train: list[dict[str, Any]], tests: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Bỏ mẫu train có câu trùng / gần trùng bất kỳ câu nào của các tập test đóng băng."""
    from eval.kiem_nhiem import tim_nhiem

    def xem(cau: str) -> dict[str, Any]:
        return {"prompt": CATALOG.render_call_prompt(cau), "meta": {"fn": ""}}

    test_xem = [xem(c) for r in tests for c in _cac_cau(r)]
    train_xem, chu = [], []
    for i, r in enumerate(train):
        for c in _cac_cau(r):
            train_xem.append(xem(c))
            chu.append(i)
    nhiem = tim_nhiem(train_xem, test_xem)
    bo = {chu[j] for j in nhiem["chi_so"]}
    return [r for i, r in enumerate(train) if i not in bo], len(bo)


def kho_tu_choi_cu(da_co: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """v6.1 — câu TỪ CHỐI có sẵn của các vòng sau (`final_cuoi`, `final_v4_them`; đã qua lọc + giám khảo ở
    vòng của chúng) chưa có trong train. Trả về (từ chối khó — lệnh cho tính năng xe không có, từ chối khác).
    Bỏ "bật sưởi" (nay làm được)."""
    kho: dict[str, dict[str, Any]] = {}
    for ten in ("final_cuoi", "final_v4_them"):
        for r in doc(P / ten / "train_vi.jsonl"):
            if r["meta"]["fn"] == "irrelevant_function" and r["meta"]["type"] == "call" \
                    and r["prompt"] not in da_co and r["prompt"] not in kho and sua_nhan_v6(r)[0] is not None:
                kho[r["prompt"]] = {**r, "meta": {**r["meta"], "v6": "tu_choi_v61"}}
    kho_kho = [r for r in kho.values() if r["meta"].get("nhom_phu") == "tu_choi_kho"]
    return kho_kho, [r for r in kho.values() if r["meta"].get("nhom_phu") != "tu_choi_kho"]


def dung_v6(so_thua: int = 2000, ti_le_nhieu: float = 0.15, so_tu_choi_them: int = 0,
            so_tu_choi_co_luot_truoc: int = 0, ra_ten: str = "final_v6") -> dict[str, Any]:
    """`so_tu_choi_them` > 0 = v6.1: thêm câu từ chối có sẵn (hết nhóm khó, phần còn lại bốc ngẫu nhiên) để
    đưa tỉ lệ
    từ chối về như v5 (v6 33,3% vs v5 40,5% -> v6 gọi bừa nhiều, trượt tiêu chí (4));
    `so_tu_choi_co_luot_truoc` câu trong
    số đó gắn thêm một lượt trước ngẫu nhiên."""
    dem: Counter[str] = Counter()
    nen: list[dict[str, Any]] = []
    for r in doc(NEN / "train_vi.jsonl"):
        r2, ly_do = sua_nhan_v6(r)
        if ly_do:
            dem[f"v5:{ly_do}"] += 1
        if r2 is not None:
            nen.append(r2)
    moi: list[dict[str, Any]] = []
    for khoi in ("k1_da_dang", "k2_moi", "k3_doi_xung", "k4_hai_luot"):
        ds = mau_tu_khoi(doc(V6 / f"v6_{khoi}.jsonl"))
        dem[f"moi:{khoi}"] = len(ds)
        moi += ds
    # bỏ mẫu mới trùng chữ y hệt mẫu đã có (cùng prompt)
    da_co = {r["prompt"] for r in nen}
    giu_moi = []
    for r in moi:
        if r["prompt"] not in da_co:
            da_co.add(r["prompt"])
            giu_moi.append(r)
    dem["moi_trung_bo"] = len(moi) - len(giu_moi)
    moi = giu_moi
    kho_1 = [{**x, "args": ap_quy_uoc(x["fn"], x["query"], x["args"]) or x["args"]}
             for x in doc(V6 / "v6_luot_1.jsonl")]
    thua = gan_luot_truoc_thua(nen, kho_1, so_thua)
    nhieu = them_mau_nhieu(nen + moi, ti_le_nhieu)
    dem.update(luot_truoc_thua=len(thua), nhieu=len(nhieu))
    tat_ca = nen + moi + thua + nhieu
    if so_tu_choi_them:  # v6.1: phần v6 giữ NGUYÊN (cùng seed), chỉ CỘNG THÊM — seed riêng cho phần thêm
        kho_kho, kho_khac = kho_tu_choi_cu(da_co)
        them = kho_kho + random.Random(SEED + 61).sample(kho_khac, so_tu_choi_them - len(kho_kho))
        co_truoc = gan_luot_truoc_thua(them, kho_1, so_tu_choi_co_luot_truoc, seed=SEED + 62)
        them_nhieu_ = them_mau_nhieu(them, ti_le_nhieu, seed=SEED + 63)
        dem.update(tu_choi_them=len(them), tu_choi_kho=len(kho_kho), tu_choi_co_luot_truoc=len(co_truoc),
                   tu_choi_nhieu=len(them_nhieu_))
        tat_ca += them + co_truoc + them_nhieu_
    tests = [r for p in TEST_DONG_BANG for r in doc(p)]
    tat_ca, so_bo = bo_nhiem(tat_ca, tests)
    dem["bo_nhiem_test"] = so_bo
    random.Random(SEED).shuffle(tat_ca)
    ra = P / ra_ten
    ghi(ra / "train_vi.jsonl", tat_ca)
    for tap in ("val_vi", "val_mau"):  # val giữ Y HỆT v5 -> chọn checkpoint theo luật cũ, so được với v5
        ghi(ra / f"{tap}.jsonl", doc(NEN / f"{tap}.jsonl"))
    bao_cao = {
        "train": len(tat_ca),
        "theo_loai": dict(Counter(r["meta"]["type"] for r in tat_ca)),
        "theo_nguon_v6": dict(Counter(r["meta"].get("v6", "v5") for r in tat_ca)),
        "co_luot_truoc": sum("Lượt trước:" in r["prompt"] for r in tat_ca),
        "dem": dict(dem),
    }
    (ra / "bao_cao_dung.json").write_text(json.dumps(bao_cao, ensure_ascii=False, indent=2), encoding="utf-8")
    return bao_cao


def main() -> None:
    ap = argparse.ArgumentParser(description="Vòng 6 — phần chạy ở máy")
    ap.add_argument("--luot-1", action="store_true")
    ap.add_argument("--so-luot-1", type=int, default=750)
    ap.add_argument("--test-nhieu", action="store_true")
    ap.add_argument("--dung", action="store_true")
    ap.add_argument("--dung-61", action="store_true", help="v6.1: v6 + ~3.300 câu từ chối có sẵn (PLAN.md)")
    a = ap.parse_args()
    if a.dung:
        print(json.dumps(dung_v6(), ensure_ascii=False, indent=2))
    if a.dung_61:
        print(json.dumps(dung_v6(so_tu_choi_them=3300, so_tu_choi_co_luot_truoc=1000, ra_ten="final_v6_1"),
                         ensure_ascii=False, indent=2))
    if a.luot_1:
        ra = chon_luot_1(doc(NEN / "train_vi.jsonl"), a.so_luot_1)
        ghi(V6 / "v6_luot_1.jsonl", ra)
        print(len(ra), dict(Counter(x["fn"] for x in ra)))
    if a.test_nhieu:
        ra = tao_test_nhieu(doc(P / "final" / "test_vi_sach_v2.jsonl"))
        ghi(P / "final" / "test_v6_nhieu.jsonl", ra)
        print(len(ra), dict(Counter(r["meta"]["nhieu"] for r in ra)),
              dict(Counter(r["meta"]["nhom_phu"] for r in ra)))


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
