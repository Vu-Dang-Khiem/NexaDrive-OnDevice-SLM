"""Chấm mẫu HỎI LẠI (`<nexa_ask>`) theo quy trình mượn từ các paper — xem PLAN.md mục E0b.

VÌ SAO KHÔNG DÙNG CÁCH CŨ. Giám khảo cũ được xem đề xuất rồi hỏi "có hợp lý không?" —
giữ 517/518 mẫu vòng 2, kể cả "Chỗ đỗ không an toàn" -> bật đèn khẩn cấp. Sửa prompt hai
lần thì lỗi chỉ đổi chỗ (sửa được câu ra lệnh thì hỏng câu sai hàm và ngược lại). Nên đổi
KIỂU câu hỏi, không vặn câu chữ nữa:

    [CODE]  câu có động từ ra lệnh?        có -> LOẠI      (APIGen, ToolACE: code trước LLM)
    [CODE]  đề xuất vi phạm luật dữ liệu?  có -> LOẠI      (ToolACE: lớp luật)
    [Qwen]  tự chọn ≤2 hàm, KHÔNG xem nhãn ┐               (Zheng 2023: tự giải trước)
    [Llama] tự chọn ≤2 hàm, KHÔNG xem nhãn ┘               (Verga 2024: giám khảo khác họ)
    [CODE]  hàm đề xuất chính có trong lựa chọn của CẢ HAI?  có -> GIỮ
    Độ tin của cả quy trình: Cohen's kappa trên bộ hiệu chỉnh có đáp án   (Thakur 2024)

File này chỉ chứa phần CODE (không gọi model). Phần gọi model nằm ở
`gen/modal_app.py::chon_de_xuat`, ghi lựa chọn của từng giám khảo ra file; bước quyết định
cuối cùng chạy ở máy, miễn phí:

    python -m gen.cham_ask --ask data/processed/ask/ask_train_vi_sach.jsonl \\
        data/processed/v2/v2_ask_train_vi.jsonl \\
        --chon data/processed/v2/chon_qwen.jsonl data/processed/v2/chon_llama.jsonl \\
        --out-dir data/processed/v2/ask_cham_moi
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from gen.ra_soat_tu_vung import DONG_TU_LENH, tach_tu
from gen.records import query_from_call_prompt
from schema.call_format import CallParseError, parse_calls
from schema.loader import CATALOG

# Hàm agent KHÔNG được tự gợi ý từ một câu than phiền (luật an toàn đã chốt).
HAM_NHAY_CAM = frozenset({"unlock_doors", "open_trunk", "open_charge_port", "open_fuel_flap"})

# Động từ ra lệnh — rộng hơn bộ của ra_soat_tu_vung vì ở đây mục đích khác: chặn câu ra lệnh
# lọt vào nhãn hỏi lại. Thêm đúng những động từ đã lọt thật trong bộ hiệu chỉnh ("Chỉ cho tôi
# trạm đổ xăng...") và các động từ gọi chức năng tra cứu / dẫn đường / phát nhạc.
DONG_TU_RA_LENH = DONG_TU_LENH | {"chỉ", "tìm", "dẫn", "phát", "tra", "kiểm", "đưa", "gọi"}

# Từ mở đầu lịch sự đứng TRƯỚC động từ: "Làm ơn bật...", "Bạn hãy mở...", "Giúp tôi tìm..."
_TU_DEM_DAU = {"làm", "ơn", "hãy", "bạn", "em", "anh", "chị", "giúp", "hộ", "giùm", "tôi",
               "mình", "xin", "vui", "lòng", "ê", "này", "ơi", "nhờ", "cho"}

SO_LUA_CHON_TOI_DA = 2


# ── lớp 1: code ─────────────────────────────────────────────────────────────
def co_ra_lenh(cau: str) -> bool:
    """True nếu câu là YÊU CẦU THẲNG: động từ ra lệnh đứng đầu, sau khi bỏ từ đệm lịch sự.

    Chỉ nhìn đầu câu — "Nóng quá, bật điều hoà đi" cũng là ra lệnh nhưng hiếm trong dữ liệu
    hỏi lại; bắt thêm kiểu đó dễ bắt oan câu than phiền có chữ "bật" ở giữa.
    """
    tu = tach_tu(cau)
    i = 0
    while i < len(tu) and tu[i] in _TU_DEM_DAU:
        i += 1
    if i >= len(tu) or tu[i] not in DONG_TU_RA_LENH:
        return False
    # "chỉ" hay là phó từ ("Chỉ cần ngồi trong xe là mệt", "Chỉ thấy mờ mờ") — đo thật trên
    # dữ liệu vòng 1: 20 câu than phiền bị bắt oan. Chỉ tính là động từ khi đi với "cho"/"đường".
    if tu[i] == "chỉ":
        return i + 1 < len(tu) and tu[i + 1] in {"cho", "đường"}
    return True


# Tiểu từ sai bảo "đi" ở CUỐI vế câu: "Quạt gió mạnh hơn tí đi", "Nhiên, mở quạt lên đi, ...".
# Đo trên 400 câu nhãn người: 4 câu ra lệnh lọt qua vì động từ không đứng đầu câu.
# "đi" còn là động từ di chuyển, nên loại trừ:
#   - đứng trước dấu "?"      : "Đường này toàn cao tốc, làm sao đi?"
#   - sau từ hỏi / trợ động từ: "không biết đi đâu", "phải đi", "muốn đi"
#   - sau đại từ              : "Thôi mình đi" (rủ nhau đi, không phải sai bảo xe)
_DI_CUOI_VE = re.compile(r"(\S+)\s+đi\s*(?=[,.!…]|$)")
_TRUOC_DI_KHONG_PHAI_SAI_BAO = {
    "sao", "đâu", "nào", "phải", "muốn", "sẽ", "đang", "chưa", "không", "được", "cần",
    "mình", "tôi", "ta", "tớ", "em", "anh", "chị", "chúng",
    # "đi" là một phần của DANH TỪ: "không biết đường đi", "lối đi", "sau chuyến đi".
    # Bắt oan thật khi kiểm trên phần kho ngoài 400 câu nhãn người (5 câu đèn pha "đường đi").
    "đường", "lối", "chuyến", "lượt", "cuộc",
}


def co_tieu_tu_sai_bao(cau: str) -> bool:
    for m in _DI_CUOI_VE.finditer(cau.lower()):
        truoc = re.sub(r"[^\wÀ-ỹ]", "", m.group(1))
        if truoc and truoc not in _TRUOC_DI_KHONG_PHAI_SAI_BAO:
            return True
    return False


# Câu nói về đèn / sự cố của XE KHÁC. Teacher được bảo viết câu cho đèn khẩn cấp CỦA MÌNH lại
# viết "Phía trước có xe gặp sự cố, đèn cảnh báo nháy liên tục" — và cả hai giám khảo cũng chọn
# bật đèn khẩn cấp (cùng mù). Trên 400 câu nhãn người: 14 mẫu kiểu này lọt lưới.
_XE_KHAC = re.compile(
    r"xe (phía|đằng) (trước|sau)|xe khác|có xe (bị|gặp)|xe (bị|gặp) (hỏng|sự cố|tai nạn|nạn)"
)
# Nhưng xe MÌNH gặp sự cố mà nhắc tới xe sau thì vẫn là tình huống bật đèn khẩn cấp thật:
# "Xe mình chết máy, mấy xe đằng sau đang lao tới".
_XE_MINH = re.compile(
    r"xe (mình|tôi|của mình|của tôi)|chết máy|tắt máy|nổ lốp|xe tự nhiên|mình phải|tôi phải"
)
# Chỉ nhắc tới xe khác thì CHƯA đủ — phải có dấu hiệu đèn/sự cố CỦA xe đó. Bắt oan thật khi kiểm
# trên phần kho ngoài 400 câu: "Nhìn mờ mờ, không biết có xe phía trước không" (đang nói tầm nhìn
# của mình -> bật đèn là hợp lý); "Đèn xe phía trước yếu quá" (tiếng Việt còn nghĩa là đèn trước
# của CHÍNH xe mình).
_DAU_HIEU_XE_KHAC = re.compile(
    r"chớp|nháy|chói|đèn cảnh báo|đèn hậu|đèn khẩn cấp|sự cố|tai nạn|hỏng|phanh gấp|dừng đột ngột"
)
HAM_DEN = frozenset({"set_hazard_lights", "set_headlights"})


def noi_ve_xe_khac(cau: str) -> bool:
    t = cau.lower()
    return bool(_XE_KHAC.search(t)) and bool(_DAU_HIEU_XE_KHAC.search(t)) and not _XE_MINH.search(t)


def doc_de_xuat(row: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    text = row["completion"].replace("<|im_end|>", "")
    return [(c.name, dict(c.args)) for c in parse_calls(text)]


def kiem_luat(cau: str, de_xuat: list[tuple[str, dict[str, Any]]]) -> tuple[bool, str]:
    """Lớp luật (ToolACE). Trả về (qua, lý do nếu loại)."""
    if co_ra_lenh(cau) or co_tieu_tu_sai_bao(cau):
        return False, "câu ra lệnh thẳng — phải làm luôn, không hỏi lại"
    if not de_xuat:
        return False, "không có đề xuất"
    if de_xuat[0][0] in HAM_DEN and noi_ve_xe_khac(cau):
        return False, "câu nói về đèn/sự cố của xe khác — không điều khiển được"
    if len(de_xuat) > SO_LUA_CHON_TOI_DA:
        return False, f"đề xuất quá {SO_LUA_CHON_TOI_DA} hàm"
    ten = [n for n, _ in de_xuat]
    nhay = [n for n in ten if n in HAM_NHAY_CAM]
    if nhay:
        return False, f"đề xuất hàm nhạy cảm: {nhay}"
    if all(CATALOG.by_name[n].chi_doc for n in ten):
        return False, "chỉ đề xuất hàm chỉ đọc — hàm chỉ đọc làm luôn, không hỏi lại"
    return True, ""


# ── lớp 2: quyết định từ lựa chọn của giám khảo ─────────────────────────────
def doc_lua_chon(text: str) -> list[str]:
    """Đọc JSON {"ham": [...]} giám khảo trả về; bỏ tên hàm không tồn tại, giữ ≤2."""
    import re

    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"không thấy JSON: {text[:80]!r}")
    data = json.loads(m.group(0))
    ds = data.get("ham", [])
    if isinstance(ds, str):
        ds = [ds]
    hop_le = [str(x).strip() for x in ds if str(x).strip() in CATALOG.by_name]
    return hop_le[:SO_LUA_CHON_TOI_DA]


def quyet_dinh(ham_chinh: str, cac_lua_chon: list[list[str]]) -> bool:
    """Giữ khi hàm đề xuất CHÍNH nằm trong lựa chọn của MỌI giám khảo (Verga: bồi thẩm đoàn).

    Chỉ đòi hàm chính (hàm đầu tiên) chứ không đòi cả hai hàm của nhãn: nhãn "điều hoà ;
    hạ kính" mà giám khảo chọn "điều hoà" là đã đồng ý về hướng xử lý.
    """
    return bool(cac_lua_chon) and all(ham_chinh in ds for ds in cac_lua_chon)


# ── lớp 3: đo độ tin của cả quy trình ───────────────────────────────────────
def cohen_kappa(dap_an: list[bool], du_doan: list[bool]) -> float:
    """Kappa nhị phân. Khác % đồng ý ở chỗ trừ đi phần 'đồng ý do may' (Thakur 2024):
    quy trình gật đầu với MỌI mẫu vẫn có % đồng ý cao nếu đa số mẫu đúng, nhưng kappa = 0."""
    n = len(dap_an)
    if n == 0 or n != len(du_doan):
        raise ValueError("hai danh sách phải cùng độ dài và khác rỗng")
    p_o = sum(a == b for a, b in zip(dap_an, du_doan, strict=True)) / n
    pa, pb = sum(dap_an) / n, sum(du_doan) / n
    p_e = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if p_e == 1 else (p_o - p_e) / (1 - p_e)


def cham_mot(
    cau: str, de_xuat: list[tuple[str, dict[str, Any]]], lua_chon: list[list[str]]
) -> tuple[bool, str]:
    """Toàn bộ quy trình cho một mẫu. Trả về (giữ, lý do nếu loại)."""
    ok, ly_do = kiem_luat(cau, de_xuat)
    if not ok:
        return False, ly_do
    chinh = de_xuat[0][0]
    if quyet_dinh(chinh, lua_chon):
        return True, ""
    return False, f"giám khảo không chọn {chinh}: " + " | ".join(",".join(x) or "-" for x in lua_chon)


def hieu_chinh(
    bo: list[tuple[str, list[tuple[str, dict[str, Any]]], bool]],
    lua_chon_theo_cau: list[dict[str, list[str]]],
) -> dict[str, Any]:
    """Chấm bộ hiệu chỉnh có đáp án bằng ĐÚNG quy trình thật, báo cả hai chiều + kappa."""
    dap_an, du_doan, chi_tiet = [], [], []
    for cau, de_xuat, dung in bo:
        giu, ly_do = cham_mot(cau, de_xuat, [lc.get(cau, []) for lc in lua_chon_theo_cau])
        dap_an.append(dung)
        du_doan.append(giu)
        chi_tiet.append(
            {"cau": cau, "de_xuat": [n for n, _ in de_xuat], "dung": dung, "giu": giu, "ly_do": ly_do}
        )
    giu_dung = sum(1 for a, b in zip(dap_an, du_doan, strict=True) if a and b)
    loai_dung = sum(1 for a, b in zip(dap_an, du_doan, strict=True) if not a and not b)
    return {
        "giu_dung": f"{giu_dung}/{sum(dap_an)}",
        "loai_dung": f"{loai_dung}/{len(dap_an) - sum(dap_an)}",
        "kappa": round(cohen_kappa(dap_an, du_doan), 3),
        "chi_tiet": chi_tiet,
    }


def _doc(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def main() -> None:
    from gen.teacher_prompts import BO_HIEU_CHINH_HOI_LAI

    ap = argparse.ArgumentParser(description="Giữ/loại mẫu hỏi lại theo lựa chọn của giám khảo")
    ap.add_argument("--ask", required=True, nargs="+", type=Path)
    ap.add_argument("--chon", required=True, nargs="+", type=Path, help="file {cau, ham} của từng giám khảo")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--nguong-kappa", type=float, default=0.6)
    args = ap.parse_args()

    lua_chon = [{r["cau"]: r["ham"] for r in _doc(p)} for p in args.chon]

    hc = hieu_chinh(BO_HIEU_CHINH_HOI_LAI, lua_chon)
    print("=== HIỆU CHỈNH (bộ câu có đáp án, chạy bằng ĐÚNG quy trình thật) ===")
    for x in hc["chi_tiet"]:
        dau = "✓" if x["dung"] == x["giu"] else "✗"
        nhan = "giữ" if x["dung"] else "loại"
        print(f"  {dau} [{nhan}] {x['cau'][:36]:38} {'+'.join(x['de_xuat']):20} {x['ly_do'][:48]}")
    print(f"giữ đúng {hc['giu_dung']} | loại đúng {hc['loai_dung']} | Cohen's kappa = {hc['kappa']}")
    if hc["kappa"] < args.nguong_kappa:
        raise SystemExit(
            f"kappa {hc['kappa']} < {args.nguong_kappa} — quy trình chưa đáng tin, KHÔNG ghi kết quả"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for p in args.ask:
        rows = _doc(p)
        giu, loai = [], []
        for r in rows:
            cau = query_from_call_prompt(r["prompt"])
            try:
                de_xuat = doc_de_xuat(r)
            except CallParseError:
                loai.append({**r, "ly_do_loai": "không đọc được nhãn"})
                continue
            ok, ly_do = cham_mot(cau, de_xuat, [lc.get(cau, []) for lc in lua_chon])
            (giu if ok else loai).append(r if ok else {**r, "ly_do_loai": ly_do})
        ten = p.stem
        for duoi, ds in (("giu", giu), ("loai", loai)):
            with open(args.out_dir / f"{ten}_{duoi}.jsonl", "w", encoding="utf-8") as f:
                for r in ds:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        dem: dict[str, int] = {}
        for r in loai:
            k = r["ly_do_loai"].split(":")[0]
            dem[k] = dem.get(k, 0) + 1
        print(f"\n{ten}: giữ {len(giu)}/{len(rows)}  | lý do loại: {dem}")


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
