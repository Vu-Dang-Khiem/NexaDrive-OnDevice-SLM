"""Chấm điểm output của model — logic thuần, không gọi model, test được ở local.

Nguyên tắc đã chốt (xem PLAN.md Phase 3):

1. **Không bao giờ báo cáo một con số gộp.** Tách theo nhóm câu (rõ ràng / mơ hồ /
   ngoài phạm vi), vì con số gộp che mất model kém: model từ chối tất cả vẫn được
   130/296 ≈ 44% trên tập test hiện tại, nghe như "có làm được gì đó".

2. **Ghi rõ SAI KIỂU GÌ, không chỉ đúng/sai.** Hai câu cùng sai nhưng "gọi bừa hàm
   cho câu ngoài phạm vi" và "từ chối nhầm câu hợp lệ" cần cách chữa ngược nhau.

3. **Tách nhóm phụ của câu từ chối.** Từ chối câu tán gẫu là dễ; từ chối yêu cầu ĐÚNG
   chủ đề xe mà xe không hỗ trợ ("bật sưởi ghế") mới khó — đây là điểm yếu đã biết của
   functional token (Jafarzadeh Asl et al. 2026 đo được over-trigger ~17%).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from gen.records import query_from_call_prompt, result_from_reply_prompt
from schema.call_format import CallParseError, parse_calls
from schema.loader import CATALOG

NEGATIVE_FN = "irrelevant_function"

# Kết cục của một câu — đặt tên theo CÁCH SAI để biết đường sửa data
DUNG = "dung"
SAI_HAM = "sai_ham"  # chọn nhầm hàm khác
SAI_THAM_SO = "sai_tham_so"  # đúng hàm, sai/thiếu/thừa tham số
TU_CHOI_NHAM = "tu_choi_nham"  # câu hợp lệ mà lại gọi irrelevant_function
GOI_BUA = "goi_bua"  # câu ngoài phạm vi mà vẫn gọi hàm thật (over-trigger)
HOI_LAI_THUA = "hoi_lai_thua"  # câu rõ ràng mà lại <nexa_ask>
THIEU_HOI_LAI = "thieu_hoi_lai"  # câu mơ hồ mà làm luôn, không hỏi
KHONG_PARSE = "khong_parse"  # output không đọc được thành lời gọi

# Kết cục riêng của LƯỢT 2 (mẫu `reply`) — model đọc kết quả hàm rồi viết câu trả lời.
# Lượt này không có "đáp án đúng" duy nhất (nhiều cách diễn đạt đều đúng), nên không
# so chuỗi mà kiểm hai điều kiểm được bằng CODE, không cần nhờ model chấm.
BIA_SO = "bia_so"  # trong câu trả lời có số KHÔNG truy được về kết quả hàm
TRA_LOI_RONG = "tra_loi_rong"  # không trả lời, hoặc lại sinh ra lời gọi hàm


@dataclass(frozen=True)
class LoiGoi:
    """Một dự đoán đã chuẩn hoá: danh sách (tên hàm, tham số) + có phải đề xuất không."""

    calls: tuple[tuple[str, dict[str, Any]], ...]
    hoi_lai: bool
    doc_duoc: bool = True

    @property
    def ten_ham(self) -> tuple[str, ...]:
        return tuple(n for n, _ in self.calls)


def doc_output(text: str) -> LoiGoi:
    """Đọc output thô của model. Không đọc được -> `doc_duoc=False` (tính là sai)."""
    try:
        calls = parse_calls(text)
    except CallParseError:
        return LoiGoi(calls=(), hoi_lai=False, doc_duoc=False)
    return LoiGoi(
        calls=tuple((c.name, dict(c.args)) for c in calls),
        hoi_lai=bool(calls and calls[0].suggested),
    )


def nhom_cua_dap_an(dap_an: LoiGoi) -> str:
    if any(n == NEGATIVE_FN for n in dap_an.ten_ham):
        return "ngoai_pham_vi"
    return "mo_ho" if dap_an.hoi_lai else "ro_rang"


def _so_sanh_tham_so(pred: dict[str, Any], gold: dict[str, Any]) -> tuple[int, int, int]:
    """Trả về (số cặp khớp, số cặp model sinh ra, số cặp đáp án có) để tính P/R/F1."""
    khop = sum(1 for k, v in gold.items() if k in pred and pred[k] == v)
    return khop, len(pred), len(gold)


# ── LUẬT C (chốt 2026-09-27, xem PLAN.md) ───────────────────────────────────
# Luật cũ so KHÍT từng tham số -> chấm sai hai lời gọi làm xe chạy Y HỆT nhau:
#   "Cho xem số km xe đã đi": nhãn get_odometer(), model get_odometer(unit="km") -> SAI
# trong khi `unit` mặc định đã là "km". Tập test còn tự mâu thuẫn (hai câu cùng có chữ "km",
# một nhãn ghi unit, một không). Luật C:
#   (1) mọi nhóm: bỏ tham số có giá trị BẰNG mặc định khai trong schema ở CẢ nhãn lẫn output;
#   (2) nhóm MƠ HỒ (hỏi lại): chỉ chấm ĐÚNG HÀM + ĐÚNG HƯỚNG (bật/tắt, mở/đóng, tăng/giảm).
#       "Quạt kêu to quá" -> giảm về mức 2 hay 3 không có đáp án nào đúng hơn; nhãn liệt kê
#       nhiều hàm nghĩa là nhiều LỰA CHỌN chấp nhận được, model chọn một là đủ.
# Luật C KHÔNG cứu khác biệt về HÀNH VI: "tăng âm lượng xíu" delta=2 vs delta=1 ở nhóm rõ ràng
# vẫn tính sai. Luật cũ giữ lại (luat="cu") để báo cáo cả hai con số.
LUAT_MAC_DINH = "C"


def bo_mac_dinh(ten: str, args: dict[str, Any]) -> dict[str, Any]:
    """Bỏ tham số có giá trị bằng mặc định khai trong schema — hai lời gọi này làm xe chạy y hệt."""
    spec = CATALOG.by_name.get(ten)
    if spec is None:
        return dict(args)
    mac_dinh = {p.name: p.default for p in spec.params if p.default is not None}
    return {k: v for k, v in args.items() if not (k in mac_dinh and mac_dinh[k] == v)}


def huong(ten: str, args: dict[str, Any]) -> tuple[Any, ...]:
    """Rút gọn một lời gọi về HƯỚNG hành động; None = đáp án không chỉ định (chấp nhận mọi hướng)."""
    if ten in ("set_ac", "set_defrost", "set_hazard_lights"):
        return (args.get("state"),)
    if ten == "control_window":
        return (args.get("position"),)
    if ten == "operate_sunroof":
        a = args.get("action")
        return ({"close": "dong", "open": "mo", "tilt": "mo"}.get(a) if a else None,)
    if ten == "set_fan_speed":
        # v6 có `delta` (người dùng chốt 2026-10-06, TRƯỚC khi chấm v6): giảm ~ hướng "thấp", tăng ~ hướng
        # "cao" —
        # "gió mạnh quá" đề xuất level=2 hay delta=-1 đều là giảm gió
        if args.get("delta"):
            return ("thap" if args["delta"] < 0 else "cao",)
        lv = args.get("level")
        return (None if lv is None else "tat" if lv == 0 else "thap" if lv <= 3 else "cao",)
    if ten == "set_headlights":
        m = args.get("mode")
        # tách riêng pha xa: gợi ý pha xa khi trời tối là lỗi an toàn (loá xe ngược chiều)
        return (None if m is None else m if m in ("off", "high_beam") else "bat",)
    if ten == "set_volume":
        d = args.get("delta")
        return (None if not d else "tang" if d > 0 else "giam",)
    return ()


def _cung_huong(pred: tuple[Any, ...], gold: tuple[Any, ...]) -> bool:
    return all(g is None or p == g for p, g in zip(pred, gold, strict=False))


def cham_mot_cau(
    output: str, dap_an_completion: str, luat: str = LUAT_MAC_DINH
) -> tuple[str, tuple[int, int, int]]:
    """Chấm 1 câu -> (kết cục, số liệu tham số để cộng dồn F1). `luat`: "C" (mặc định) | "cu"."""
    pred = doc_output(output)
    gold = doc_output(dap_an_completion)
    if luat == "C" and pred.doc_duoc:
        pred = LoiGoi(tuple((n, bo_mac_dinh(n, a)) for n, a in pred.calls), pred.hoi_lai)
        gold = LoiGoi(tuple((n, bo_mac_dinh(n, a)) for n, a in gold.calls), gold.hoi_lai)

    if not pred.doc_duoc:
        return KHONG_PARSE, (0, 0, sum(len(a) for _, a in gold.calls))

    gold_la_negative = any(n == NEGATIVE_FN for n in gold.ten_ham)
    pred_la_negative = any(n == NEGATIVE_FN for n in pred.ten_ham)

    if gold_la_negative:
        return (DUNG if pred_la_negative else GOI_BUA), (0, 0, 0)
    if pred_la_negative:
        return TU_CHOI_NHAM, (0, 0, sum(len(a) for _, a in gold.calls))

    # số liệu tham số tính trên các hàm khớp tên, theo thứ tự
    khop = sinh_ra = can_co = 0
    for (ten_p, args_p), (ten_g, args_g) in zip(pred.calls, gold.calls, strict=False):
        if ten_p == ten_g:
            k, s, c = _so_sanh_tham_so(args_p, args_g)
            khop, sinh_ra, can_co = khop + k, sinh_ra + s, can_co + c
        else:
            sinh_ra += len(args_p)
            can_co += len(args_g)
    so_lieu = (khop, sinh_ra, can_co)

    if pred.hoi_lai != gold.hoi_lai:
        return (HOI_LAI_THUA if pred.hoi_lai else THIEU_HOI_LAI), so_lieu
    if luat == "C" and gold.hoi_lai:
        # nhãn nhiều hàm = nhiều lựa chọn chấp nhận được; model chọn MỘT (hàm đầu) là đủ
        chon = pred.calls[0]
        cung_ten = [c for c in gold.calls if c[0] == chon[0]]
        if not cung_ten:
            return SAI_HAM, so_lieu
        if not _cung_huong(huong(*chon), huong(*cung_ten[0])):
            return SAI_THAM_SO, so_lieu
        return DUNG, so_lieu
    if pred.ten_ham != gold.ten_ham:
        return SAI_HAM, so_lieu
    if [a for _, a in pred.calls] != [a for _, a in gold.calls]:
        return SAI_THAM_SO, so_lieu
    return DUNG, so_lieu


# ── kiểm câu trả lời tự nhiên (lượt 2) không bịa số ─────────────────────────
_SO = re.compile(r"-?\d[\d.,]*")
SAI_LECH_CHO_PHEP = 1.0  # cho phép làm tròn: 75.8 phút -> "khoảng 75 phút"


def _doc_so(chuoi: str) -> list[float]:
    """Đọc mọi số trong chuỗi, hiểu cả cách viết của người Việt.

    "42.750 km" là 42750 (dấu chấm ngăn nghìn), còn "75.8" là 75,8 — phân biệt bằng
    độ dài nhóm chữ số sau dấu: đúng 3 chữ số thì coi là ngăn nghìn.
    """
    ket_qua: list[float] = []
    for raw in _SO.findall(chuoi):
        s = raw.rstrip(".,")
        # bỏ dấu ngăn nghìn: 42.750 / 40,181 -> 42750 / 40181
        s = re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", s)
        s = s.replace(",", ".")
        try:
            ket_qua.append(float(s))
        except ValueError:
            continue
    return ket_qua


def khong_bia_so(reply: str, result: dict[str, Any], query: str = "") -> bool:
    """True nếu mọi số trong câu trả lời đều truy được về kết quả hàm (hoặc về câu hỏi).

    Đây là chốt chặn quan trọng nhất của lượt 2 — model bịa số trên xe rất nguy hiểm
    ("xăng còn 60%" trong khi thực tế 12%) — và kiểm được bằng CODE nên không dính rủi ro
    thiên vị khi nhờ model cùng họ Qwen chấm.

    Nhưng phải chấp nhận 4 kiểu diễn đạt hợp lệ, nếu không sẽ bắt nhầm hàng loạt câu trả
    lời tốt (đo thật: 83/2122 mẫu bị cờ oan):
      • số người dùng nói trong câu hỏi  ("đi 450 km có đủ xăng không")
      • làm tròn                          (eta 75.8 -> "khoảng 75 phút")
      • dấu ngăn nghìn kiểu Việt          (42750 -> "42.750 km")
      • bỏ dấu âm, diễn đạt bằng chữ      (margin -218.6 -> "còn thiếu 218.6 km")
    """
    nguon = _doc_so(str(result)) + _doc_so(query)
    nguon += [abs(x) for x in nguon]
    return all(
        any(abs(s - g) <= SAI_LECH_CHO_PHEP for g in nguon) for s in _doc_so(reply)
    )


def la_mau_tra_loi(row: dict[str, Any]) -> bool:
    return row.get("meta", {}).get("type") == "reply"


def cham_cau_tra_loi(output: str, prompt: str) -> str:
    """Chấm LƯỢT 2: model đã có kết quả hàm, giờ viết câu trả lời cho người dùng.

    Không so chuỗi với đáp án vì "Đèn pha đã tắt." và "Mình đã tắt đèn pha rồi nhé."
    đều đúng — so chuỗi sẽ đánh trượt oan gần hết. Chỉ kiểm 2 điều thật sự quan trọng
    và kiểm được bằng code:
      • có trả lời bằng câu chữ không (không được im lặng, không được gọi hàm tiếp);
      • mọi con số trong câu có truy được về kết quả hàm không — đây mới là thứ nguy
        hiểm trên xe ("xăng còn 60%" trong khi thực tế 12%).
    """
    t = output.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    if not t or parse_calls_duoc(t):
        return TRA_LOI_RONG
    try:
        ket_qua = result_from_reply_prompt(prompt)
    except (ValueError, KeyError):
        ket_qua = {}
    cau_hoi = ""
    try:
        cau_hoi = query_from_call_prompt(prompt)
    except ValueError:
        pass
    return DUNG if khong_bia_so(t, ket_qua, cau_hoi) else BIA_SO


def parse_calls_duoc(text: str) -> bool:
    """True nếu chuỗi đọc được thành lời gọi hàm — dùng để bắt lượt 2 gọi hàm tiếp."""
    try:
        return bool(parse_calls(text))
    except CallParseError:
        return False


# ── cộng dồn toàn bộ tập test ───────────────────────────────────────────────
@dataclass
class KetQuaNhom:
    so_cau: int = 0
    dung: int = 0
    ket_cuc: Counter = field(default_factory=Counter)

    @property
    def do_chinh_xac(self) -> float:
        return round(self.dung / self.so_cau * 100, 1) if self.so_cau else 0.0


def cham_toan_bo(
    dap_an: list[dict[str, Any]], du_doan: dict[str, str], luat: str = LUAT_MAC_DINH
) -> dict[str, Any]:
    """`dap_an` = các dòng test; `du_doan` = {prompt: output thô của model}."""
    theo_nhom: dict[str, KetQuaNhom] = defaultdict(KetQuaNhom)
    theo_nhom_phu: dict[str, KetQuaNhom] = defaultdict(KetQuaNhom)
    theo_ham: dict[str, KetQuaNhom] = defaultdict(KetQuaNhom)
    nham_lan: Counter = Counter()
    ket_cuc_chung: Counter = Counter()
    khop = sinh_ra = can_co = 0
    thieu_du_doan = 0

    for r in dap_an:
        prompt = r["prompt"]
        if prompt not in du_doan:
            thieu_du_doan += 1
            continue
        if la_mau_tra_loi(r):
            # Lượt 2 chấm bằng luật riêng và đứng thành NHÓM RIÊNG — gộp vào "rõ ràng"
            # sẽ làm hỏng cả hai con số: lượt 1 đo chọn đúng hàm, lượt 2 đo viết đúng câu.
            kc, nhom = cham_cau_tra_loi(du_doan[prompt], prompt), "tra_loi"
        else:
            kc, (k, s, c) = cham_mot_cau(du_doan[prompt], r["completion"], luat)
            khop, sinh_ra, can_co = khop + k, sinh_ra + s, can_co + c
            nhom = nhom_cua_dap_an(doc_output(r["completion"]))
        nhom_phu = r["meta"].get("nhom_phu", "khac")
        for bang, khoa in ((theo_nhom, nhom), (theo_nhom_phu, nhom_phu), (theo_ham, r["meta"]["fn"])):
            bang[khoa].so_cau += 1
            bang[khoa].dung += int(kc == DUNG)
            bang[khoa].ket_cuc[kc] += 1
        ket_cuc_chung[kc] += 1

        if kc == SAI_HAM:
            pred = doc_output(du_doan[prompt])
            nham_lan[(r["meta"]["fn"], pred.ten_ham[0] if pred.ten_ham else "?")] += 1

    tong = sum(ket_cuc_chung.values())
    p = khop / sinh_ra if sinh_ra else 0.0
    r_ = khop / can_co if can_co else 0.0
    return {
        "so_cau_da_cham": tong,
        "thieu_du_doan": thieu_du_doan,
        "theo_nhom": {
            k: {"so_cau": v.so_cau, "dung_%": v.do_chinh_xac} for k, v in sorted(theo_nhom.items())
        },
        "theo_nhom_phu": {
            k: {"so_cau": v.so_cau, "dung_%": v.do_chinh_xac} for k, v in sorted(theo_nhom_phu.items())
        },
        "theo_ham": {k: {"so_cau": v.so_cau, "dung_%": v.do_chinh_xac} for k, v in sorted(theo_ham.items())},
        "loi_theo_kieu": dict(ket_cuc_chung.most_common()),
        "ty_le_loi_%": {
            "tu_choi_nham": round(ket_cuc_chung[TU_CHOI_NHAM] / tong * 100, 1) if tong else 0.0,
            "goi_bua": round(ket_cuc_chung[GOI_BUA] / tong * 100, 1) if tong else 0.0,
            "hoi_lai_thua": round(ket_cuc_chung[HOI_LAI_THUA] / tong * 100, 1) if tong else 0.0,
            "thieu_hoi_lai": round(ket_cuc_chung[THIEU_HOI_LAI] / tong * 100, 1) if tong else 0.0,
            "khong_parse": round(ket_cuc_chung[KHONG_PARSE] / tong * 100, 1) if tong else 0.0,
            "bia_so": round(ket_cuc_chung[BIA_SO] / tong * 100, 1) if tong else 0.0,
        },
        "tham_so": {
            "precision_%": round(p * 100, 1),
            "recall_%": round(r_ * 100, 1),
            "f1_%": round(2 * p * r_ / (p + r_) * 100, 1) if (p + r_) else 0.0,
        },
        "hay_nham_nhat": [
            {"dap_an": a, "model_chon": b, "so_lan": n} for (a, b), n in nham_lan.most_common(10)
        ],
    }


def cau_sai(
    dap_an: list[dict[str, Any]], du_doan: dict[str, str], luat: str = LUAT_MAC_DINH
) -> list[dict[str, Any]]:
    """Danh sách câu sai kèm kiểu sai — để đọc tay, đây mới là thứ giúp sửa data."""
    ds = []
    for r in dap_an:
        out = du_doan.get(r["prompt"])
        if out is None:
            continue
        kc = (
            cham_cau_tra_loi(out, r["prompt"])
            if la_mau_tra_loi(r)
            else cham_mot_cau(out, r["completion"], luat)[0]
        )
        if kc != DUNG:
            ds.append(
                {
                    "cau": query_from_call_prompt(r["prompt"]),
                    "kieu_sai": kc,
                    "dap_an": r["completion"].replace("<|im_end|>", ""),
                    "model_sinh": out.replace("<|im_end|>", "")[:120],
                    "fn": r["meta"]["fn"],
                    "nhom_phu": r["meta"].get("nhom_phu", "khac"),
                }
            )
    return ds
