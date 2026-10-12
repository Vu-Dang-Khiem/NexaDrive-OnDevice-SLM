"""ROUTER — nối model, guardrail, xe mô phỏng và câu trả lời thành một lượt hội thoại (PLAN.md Phase 5).

Một câu của người lái đi qua:

    có lời gọi ĐANG CHỜ xác nhận? -- "có" -> thực thi · "không" -> huỷ · câu khác -> huỷ rồi xử lý câu mới
        |
    model (cổng chung) -> parser -> từ chối / đề xuất (hỏi lại) / gọi hàm
        |
    validator (serve/kiem_tra.py) -> luật an toàn (serve/an_toan.py) -> xe mô phỏng -> trả lời

Model chỉ lo HIỂU câu; mọi quyết định an toàn nằm ở code. Model giả hay student GGUF đều đi qua đúng
một luồng.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

from eval.metrics import khong_bia_so
from gen.loc import tu_tieng_anh
from schema.call_format import CallParseError, parse_calls, render_call, render_completion
from schema.loader import CATALOG
from serve import an_toan, bo_nho, dong_nap, tra_loi
from serve.chuan_hoa import chuan_hoa_chu
from serve.contracts import AssistantTurn, CallPayload, ClarifyTurn, StateEvent, UiActionPayload
from serve.kiem_tra import THIEU_MUC_AM_LUONG, kiem_tra
from serve.mo_hinh import LoiGoi, MoHinh
from serve.quy_uoc import ap_quy_uoc_chay
from serve.tach_lenh import tach_lenh
from serve.xe_mo_phong import XeMoPhong

TU_CHOI_HAM = "irrelevant_function"


def _la_tu_choi(tho: str) -> bool:
    """Output thô của model là lời TỪ CHỐI (output hỏng không tính — để router báo "chưa hiểu")."""
    try:
        return parse_calls(tho)[0].name == TU_CHOI_HAM
    except (CallParseError, IndexError):
        return False

# Chỉ coi là "có/không" khi CẢ CÂU là lời đáp (kèm từ đệm) — "dạ, mở cốp đi" là LỆNH MỚI, không phải "dạ".
# "ừ bật đi", "có mở luôn" vẫn là ĐỒNG Ý: sau lời đáp cho phép động từ chung (bật/mở/làm) + từ đệm.
_TU_DEM = r"(\s+(đi|nhé|nha|luôn|ạ|nào|bạn|em|anh|chị|giúp|giùm|với|rồi|làm|bật|mở|cảm ơn|cám ơn))*"
_DONG_Y = re.compile(
    rf"^(có|ừ+m?|ờ|ok+|oke|okay|đồng ý|được|vâng|dạ|dạ có|chắc chắn|đúng rồi|làm đi){_TU_DEM}$"
)
_KHONG = re.compile(rf"^(không|dạ không|thôi|thôi khỏi|khỏi|đừng|ko|không cần){_TU_DEM}$")
# Chọn giữa 2 đề xuất phải nói rõ "cái đầu / cái thứ hai" — "hạ kính sau" là lệnh mới,
# không phải "chọn cái sau".
_CHON = [
    re.compile(r"\b(cái|việc|số|lựa chọn) (đầu|đầu tiên|thứ nhất|1|một)\b"),
    re.compile(r"\b(cái|việc|số|lựa chọn) (sau|thứ hai|2|hai)\b"),
]


# đơn vị đo viết kiểu quốc tế là bình thường (chạy thử: "24°C", "13,9mm", "31,2 psi" từng bị lọc oan)
_DON_VI = {"c", "mm", "psi", "kph", "kwh"}


def _chu_trong(x: Any) -> list[str]:
    """Mọi chuỗi nằm trong GIÁ TRỊ của kết quả hàm (đệ quy), bỏ tên trường."""
    if isinstance(x, str):
        return [x]
    if isinstance(x, dict):
        return [s for v in x.values() for s in _chu_trong(v)]
    if isinstance(x, list | tuple):
        return [s for v in x for s in _chu_trong(v)]
    return []


@dataclass
class DangCho:
    """Lời gọi đang chờ người lái xác nhận."""

    loi_goi: list[LoiGoi]
    loai: Literal["de_xuat", "an_toan"]
    cau_goc: str


@dataclass
class _KetQuaLuot:
    calls: list[CallPayload] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    ui: list[UiActionPayload] = field(default_factory=list)
    cau: list[str] = field(default_factory=list)
    doi_trang_thai: bool = False
    bi_chan: bool = False  # luật an toàn từ chối
    hong: bool = False  # model sinh lời gọi sai giá trị


class TacTu:
    """`luot_2`: "mau" (MẶC ĐỊNH, người dùng chốt 2026-10-05) — câu trả lời bằng câu mẫu: luôn đúng số liệu,
    tức thì (đo model thật: lượt 2 thêm ~1,6 s và có lỗi nghĩa lưới an toàn không bắt được, PLAN.md mục W);
    "model" — model viết câu trả lời, qua lưới an toàn (bật để trình diễn / đánh giá)."""

    def __init__(
        self, mo_hinh: MoHinh, xe: XeMoPhong | None = None, luot_2: Literal["mau", "model"] = "mau"
    ) -> None:
        self.mo_hinh = mo_hinh
        self.xe = xe or XeMoPhong()
        self.luot_2 = luot_2
        self.dang_cho: DangCho | None = None
        self.luot_truoc: bo_nho.LuotTruoc | None = None  # hành động vừa làm (bộ nhớ luật, serve/bo_nho.py)
        self._lan_tro_chuyen = 0  # xoay vòng câu đáp xã giao (vd cảm ơn) cho đỡ lặp

    # ── cửa vào ───────────────────────────────────────────────────────────
    def xu_ly(self, cau: str, nguon: str = "nut") -> list[BaseModel]:
        t0 = time.perf_counter()
        # câu NHIỀU LỆNH nối bằng từ nối rõ ("… xong …", "sau đó") -> làm lần lượt (serve/tach_lenh.py);
        # đang chờ xác nhận thì không tách (câu trả lời "có" / "không" là một ý)
        doan = tach_lenh(cau.strip()) if self.dang_cho is None else [cau.strip()]
        su_kien = self._xu_ly(doan[0], nguon) if len(doan) == 1 else self._nhieu_lenh(doan, nguon)
        for s in su_kien:
            if isinstance(s, AssistantTurn):
                s.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return su_kien

    def _nhieu_lenh(self, doan: list[str], nguon: str) -> list[BaseModel]:
        """Xử lý từng đoạn THEO THỨ TỰ, gộp thành một lượt trả lời. Đoạn nào cần hỏi lại (mơ hồ, xác nhận an
        toàn, thiếu tham số) thì DỪNG ở đó — các đoạn sau chưa làm và được nói rõ — để không việc nào chạy khi
        người lái chưa trả lời."""
        cau_noi: list[str] = []
        calls: list[CallPayload] = []
        results: list[dict[str, Any]] = []
        ui: list[UiActionPayload] = []
        loai: list[str] = []
        trang_thai: StateEvent | None = None
        hoi: ClarifyTurn | None = None
        for i, d in enumerate(doan):
            for s in self._xu_ly(d, nguon):
                if isinstance(s, AssistantTurn):
                    cau_noi.append(s.text)
                    calls += s.calls
                    results += s.results
                    ui += s.ui_actions
                    loai.append(s.loai)
                elif isinstance(s, StateEvent):
                    trang_thai = s  # trạng thái SAU CÙNG
                elif isinstance(s, ClarifyTurn):
                    hoi = s
            if hoi is not None:
                con = doan[i + 1 :]
                # giao diện hiện / đọc câu hỏi lại -> ghép cả việc đã làm + việc chưa làm vào đó
                them = [tra_loi.con_lai(con)] if con else []
                hoi.text = " ".join([*dict.fromkeys(cau_noi), hoi.text, *them])
                break
        ra: list[BaseModel] = []
        if cau_noi:
            ra.append(
                AssistantTurn(
                    text=" ".join(dict.fromkeys(cau_noi)),
                    loai="thuc_thi" if calls else loai[0],
                    calls=calls,
                    results=results,
                    ui_actions=ui,
                )
            )
        return ra + ([trang_thai] if trang_thai else []) + ([hoi] if hoi else [])

    def _xu_ly(self, cau: str, nguon: str) -> list[BaseModel]:
        if self.dang_cho is not None:
            tl = self._tra_loi_dang_cho(cau, nguon)
            if tl is not None:
                return tl
            self.dang_cho = None  # nói sang việc khác -> huỷ lời gọi đang chờ, xử lý như câu mới

        # XÃ GIAO THUẦN ("xin chào", "cảm ơn", "bạn là ai", "bạn làm được gì") không kèm động từ điều khiển:
        # đáp câu mẫu ngay, không qua model (model đôi khi hỏi lại thừa với câu tán gẫu)
        if (dap := tra_loi.xa_giao(cau)) is not None:
            return [AssistantTurn(text=dap, loai="tro_chuyen")]

        # ĐÓNG nắp xăng / nắp sạc: schema không có hàm đóng (model dễ gọi nhầm hàm MỞ) -> luật runtime
        # (serve/dong_nap.py, "cách A"), không hỏi model
        if (nap := dong_nap.nap_can_dong(cau, self.luot_truoc.ten if self.luot_truoc else None)) is not None:
            return self._dong_nap(nap)

        # BỘ NHỚ — hai cách (người dùng chọn "cả hai", PLAN.md): model v6 đã học ngữ cảnh -> đưa lời gọi
        # lượt trước vào prompt, KHÔNG dùng luật; model v5 -> luật: câu nối tiếp chỉ gồm một động từ ("đóng
        # đi", "giảm nữa") suy từ hành động lượt trước, không hỏi model
        nho_model = getattr(self.mo_hinh, "nho_luot_truoc", False)
        if not nho_model and (tiep := bo_nho.noi_tiep(cau, self.luot_truoc, self.xe.s)) is not None:
            return self._thuc_hien(cau, [tiep], nguon)
        truoc = None
        if nho_model and self.luot_truoc:
            truoc = render_call(self.luot_truoc.ten, self.luot_truoc.args)

        # model nhạy chữ hoa / thường ở chữ đầu (dữ liệu train luôn viết hoa) -> đưa đúng hình thức lúc train
        vao = chuan_hoa_chu(cau)
        tho = self.mo_hinh.goi_ham(vao, truoc)
        # NGỮ CẢNH chỉ để GIÚP hiểu câu nối tiếp ("đóng đi"): có lượt trước mà model TỪ CHỐI -> hỏi lại model
        # không kèm lượt trước; bản đó hiểu được (gọi hàm / đề xuất) thì dùng. Người dùng gặp 2026-10-07:
        # "ngoài trời đẹp thật" sau "hạ kính ghế phụ" bị từ chối, đứng riêng thì hỏi mở cửa sổ trời.
        if truoc is not None and _la_tu_choi(tho):
            rieng = self.mo_hinh.goi_ham(vao, None)
            if not _la_tu_choi(rieng):
                tho = rieng
        try:
            calls = parse_calls(tho)
        except CallParseError:
            return [AssistantTurn(text=tra_loi.CHUA_HIEU, loai="chua_hieu")]
        # quy ước gán nhãn của dữ liệu train áp lại lúc chạy (serve/quy_uoc.py): "nữa" ±2 độ, "hạ 2 kính"…
        loi_goi = [(c.name, ap_quy_uoc_chay(c.name, dict(c.args), cau, c.suggested)) for c in calls]

        # chỉnh nhiệt TƯƠNG ĐỐI không kèm số: model v5 chưa học tham số `delta` -> đoán (có lúc TẮT điều
        # hoà). Luật runtime: nhiệt hiện tại ± 2 (16–30); áp khi model chọn điều hoà / từ chối mà KHÔNG tự
        # đưa `delta` (model v6 có `delta` thì để model làm).
        h = tra_loi.huong_nhiet(cau)
        if h and loi_goi[0][0] in ("set_ac", TU_CHOI_HAM) and "delta" not in loi_goi[0][1]:
            moi = min(30, max(16, self.xe.s.ac_temperature + h * tra_loi.BUOC_NHIET))
            return self._thuc_hien(cau, [("set_ac", {"state": "on", "temperature": moi})], nguon)

        if loi_goi[0][0] == TU_CHOI_HAM:
            return [self._tu_choi(cau)]
        if calls[0].suggested:
            return self._de_xuat(cau, loi_goi)
        return self._thuc_hien(cau, loi_goi, nguon)

    def _dong_nap(self, nap: str) -> list[BaseModel]:
        """Đóng nắp theo luật runtime. Lời gọi trả về giao diện mang tên `close_*` (frontend có hoạt cảnh
        đóng) và đánh dấu rõ là luật, không phải model."""
        ket_qua = self.xe.dong_nap(nap)
        ten = dong_nap.TEN_HAM[nap]
        ham_mo = "open_fuel_flap" if nap == dong_nap.XANG else "open_charge_port"
        self.luot_truoc = None  # đã đóng xong; "mở ra" sau đó để model hiểu như câu mới
        luot = AssistantTurn(
            text=tra_loi.bao_dong_nap(nap, ket_qua["da_dong_san"]),
            calls=[CallPayload(name=ten, arguments={}, nexa="luật runtime (không qua model)")],
            results=[ket_qua],
            ui_actions=[UiActionPayload(camera=CATALOG.by_name[ham_mo].ui_action.camera)],
        )
        return [luot, StateEvent(vehicle=self.xe.anh_chup())]

    # ── lời gọi đang chờ xác nhận ─────────────────────────────────────────
    def _tra_loi_dang_cho(self, cau: str, nguon: str) -> list[BaseModel] | None:
        """None = câu này không phải trả lời cho lời gọi đang chờ."""
        assert self.dang_cho is not None
        t = " ".join(re.sub(r"[.,!?]", " ", cau.lower()).split())
        cho = self.dang_cho
        if _KHONG.match(t):
            self.dang_cho = None
            return [AssistantTurn(text=tra_loi.HUY, loai="huy")]
        chon = next((i for i, p in enumerate(_CHON) if p.search(t) and i < len(cho.loi_goi)), None)
        if chon is None and _DONG_Y.match(t):
            if len(cho.loi_goi) > 1:  # "có" nhưng có 2 lựa chọn -> hỏi chọn một, không tự chọn thay
                ten = [n for n, _ in cho.loi_goi]
                return [
                    ClarifyTurn(
                        text=tra_loi.hoi_chon(cho.loi_goi, cho.cau_goc), ly_do="chon", function=ten[0],
                        functions=ten,
                    )
                ]
            chon = 0
        if chon is None:
            return None
        self.dang_cho = None
        return self._thuc_hien(cho.cau_goc, [cho.loi_goi[chon]], nguon, da_xac_nhan=cho.loai == "an_toan")

    # ── ba nhánh ──────────────────────────────────────────────────────────
    def _tu_choi(self, cau: str) -> AssistantTurn:
        """Model từ chối: câu xã giao rõ ràng (chào, cảm ơn…) -> đáp theo loại; còn lại -> xin lỗi như cũ
        (hoặc câu model viết nếu bật lượt 2 bằng model)."""
        text, loai = tra_loi.tra_loi_ngoai_pham_vi(cau, self._lan_tro_chuyen)
        if loai != "khong_lam_duoc":
            self._lan_tro_chuyen += 1
            return AssistantTurn(text=text, loai="tro_chuyen")
        if self.luot_2 == "mau":
            return AssistantTurn(text=tra_loi.TU_CHOI, loai="tu_choi")
        cau_model = self.mo_hinh.tra_loi(cau, render_completion([(TU_CHOI_HAM, {})]), {"handled": True})
        dung = cau_model if cau_model and "<nexa" not in cau_model else tra_loi.TU_CHOI
        return AssistantTurn(text=dung, loai="tu_choi")

    def _de_xuat(self, cau: str, loi_goi: list[LoiGoi]) -> list[BaseModel]:
        """Câu mơ hồ: KHÔNG thực thi — lưu lời gọi chờ và hỏi lại. Không bao giờ tự đề xuất hàm nhạy cảm."""
        giu: list[LoiGoi] = []
        for ten, args in loi_goi:
            spec = CATALOG.by_name[ten]
            kt = kiem_tra(ten, args)
            # than âm lượng mà model không đưa mức -> đoán hướng từ câu (xem tra_loi.huong_am_luong)
            if ten == "set_volume" and THIEU_MUC_AM_LUONG in kt.thieu and (h := tra_loi.huong_am_luong(cau)):
                kt = kiem_tra(ten, {"delta": h * tra_loi.BUOC_AM_LUONG})
            if ten == TU_CHOI_HAM or spec.nhay_cam or not kt.hop_le:
                continue
            giu.append((ten, kt.args))
        if not giu:
            return [AssistantTurn(text=tra_loi.CHUA_RO, loai="chua_hieu")]
        self.dang_cho = DangCho(giu[:2], "de_xuat", cau)
        ten = [n for n, _ in giu[:2]]
        return [
            ClarifyTurn(
                text=tra_loi.hoi_de_xuat(giu[:2], cau, self.xe.s), ly_do="de_xuat", function=ten[0],
                functions=ten,
            )
        ]

    def _thuc_hien(
        self, cau: str, loi_goi: list[LoiGoi], nguon: str, da_xac_nhan: bool = False
    ) -> list[BaseModel]:
        kq = _KetQuaLuot()
        for ten, args in loi_goi:
            if ten == TU_CHOI_HAM:
                continue
            kt = kiem_tra(ten, args)
            if kt.loi:
                kq.cau.append(tra_loi.CHUA_HIEU)
                kq.hong = True
                continue
            if kt.thieu:  # hỏi lại một lần; câu trả lời sau xử lý như câu mới (hạn chế đã ghi)
                return [
                    ClarifyTurn(
                        text=tra_loi.hoi_thieu(ten, kt.args, kt.thieu),
                        ly_do="thieu_tham_so",
                        function=ten,
                        functions=[ten],
                        missing=kt.thieu,
                    )
                ]
            qd = an_toan.xet(ten, kt.args, self.xe.s, nguon)
            if qd.xu_ly == "tu_choi":
                kq.cau.append(qd.ly_do)
                kq.bi_chan = True
                continue
            if qd.xu_ly == "hoi_lai" and not da_xac_nhan:
                self.dang_cho = DangCho([(ten, qd.args)], "an_toan", cau)
                hoi = ClarifyTurn(
                    text=tra_loi.hoi_an_toan(ten, qd.args, qd.ly_do),
                    ly_do="an_toan",
                    function=ten,
                    functions=[ten],
                )
                return self._ket_thuc(kq)[: 2 if kq.calls else 0] + [hoi]
            if qd.xu_ly == "gioi_han":
                kq.cau.append(qd.ly_do)
            self._chay(cau, ten, qd.args, kq, mot_loi_goi=len(loi_goi) == 1)
        return self._ket_thuc(kq)

    def _ket_thuc(self, kq: _KetQuaLuot) -> list[BaseModel]:
        """Lượt trả lời + ảnh chụp trạng thái xe nếu có gì thay đổi (frontend cập nhật 3D/HUD)."""
        return [self._luot(kq)] + ([StateEvent(vehicle=self.xe.anh_chup())] if kq.doi_trang_thai else [])

    # ── thực thi một lời gọi ──────────────────────────────────────────────
    def _chay(self, cau: str, ten: str, args: dict[str, Any], kq: _KetQuaLuot, mot_loi_goi: bool) -> None:
        spec = CATALOG.by_name[ten]
        truoc_khi_chay = self.xe.anh_chup()  # tăng / giảm có chạm giới hạn không ("giảm quạt" khi đã tắt)
        ket_qua = self.xe.thuc_thi(ten, args)
        kq.calls.append(CallPayload(name=ten, arguments=args, nexa=render_call(ten, args)))
        kq.results.append(ket_qua)
        kq.ui.append(
            UiActionPayload(
                camera=spec.ui_action.camera, animation=spec.ui_action.animation, hud=dict(spec.ui_action.hud)
            )
        )
        kq.doi_trang_thai |= not spec.chi_doc
        # câu hỏi (xăng, lốp...) không ghi đè: "mở cốp", "còn xăng không", "đóng đi" vẫn hiểu
        if not spec.chi_doc:
            h = tra_loi.huong_nhiet(cau) or bo_nho.huong_cau(cau, self.luot_truoc)
            self.luot_truoc = bo_nho.LuotTruoc(ten, dict(args), bo_nho.huong_cua(ten, args, h))
        # bật "sưởi" cho câu than lạnh: model lượt 2 sẽ nói "bật điều hoà" -> dùng câu mẫu nói đúng "sưởi"
        suoi = tra_loi.la_suoi(ten, args, cau)
        cau_model = (
            self.mo_hinh.tra_loi(cau, render_completion([(ten, args)]), ket_qua)
            if self.luot_2 == "model" and mot_loi_goi and not suoi
            else None
        )
        kq.cau.append(
            tra_loi.bao_cham_gioi_han(ten, args, truoc_khi_chay, self.xe.anh_chup())
            or (cau_model if self._cau_dung_duoc(cau_model, ket_qua, cau)
                else tra_loi.bao_ket_qua(ten, args, ket_qua, cau))
        )

    @staticmethod
    def _cau_dung_duoc(cau_model: str | None, ket_qua: dict[str, Any], cau: str) -> bool:
        """Lưới an toàn lượt 2: có chữ, không sinh lại lời gọi hàm, không bịa số (cùng luật với bộ chấm),
        không lẫn tiếng Anh (chạy thử model thật: "Không cần phải refuel ngay") — trừ từ có sẵn trong câu
        người lái hoặc trong kết quả (tên bài hát, địa danh)."""
        if not (cau_model and cau_model.strip()) or "<nexa" in cau_model:
            return False
        # chỉ miễn từ có trong GIÁ TRỊ kết quả, không phải tên trường ("needs_refuel" không cho "refuel" qua)
        nguon = (cau + " " + " ".join(_chu_trong(ket_qua))).lower()
        lan_anh = [w for w in tu_tieng_anh(cau_model) if w.lower() not in nguon and w.lower() not in _DON_VI]
        return not lan_anh and khong_bia_so(cau_model, ket_qua, cau)

    @staticmethod
    def _luot(kq: _KetQuaLuot) -> AssistantTurn:
        loai = "thuc_thi" if kq.calls else ("chua_hieu" if kq.hong and not kq.bi_chan else "tu_choi")
        # bỏ câu trùng (vd hai kính sau cùng bị giới hạn -> lời giải thích chỉ nói một lần)
        text = " ".join(dict.fromkeys(kq.cau)) if kq.cau else tra_loi.CHUA_HIEU
        return AssistantTurn(text=text, loai=loai, calls=kq.calls, results=kq.results, ui_actions=kq.ui)
