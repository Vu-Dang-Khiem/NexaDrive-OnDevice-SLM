"""Wake-word "hey nexa" (PLAN.md Phase 6b) — chạy trong env `giong_noi` cùng dịch vụ giọng nói.

Một đường xử lý DUY NHẤT cho train, đo và chạy thật: âm thanh 16 kHz int16 cắt thành khung 80 ms
(1280 mẫu) -> đặc trưng chung của openWakeWord (mel-spectrogram + `speech_embedding` của Google, chạy
kiểu stream) -> 16 khung × 96 chiều gần nhất (~2 s âm thanh) -> bộ phân loại nhỏ của mình
(scikit-learn) -> điểm 0..1. Điểm ≥ ngưỡng thì báo thức rồi nghỉ 2 s (không báo lặp).

Sau khi thức: `ThuLenh` gom âm thanh câu lệnh (giữ 0,3 s trước khung báo thức — người ta hay nói liền
"hey nexa mở cốp"), dừng khi silero VAD thấy im ~0,8 s sau khi đã có tiếng nói, hoặc quá 8 s; không nghe
thấy tiếng nói trong 4 s thì huỷ.
"""

from __future__ import annotations

import re
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

KHUNG = 1280  # 80 ms @ 16 kHz — đúng bước của đặc trưng openWakeWord
SO_KHUNG = 16  # số khung đặc trưng đưa vào bộ phân loại (~2 s âm thanh)
THU_MUC = Path("models/wake")
MO_HINH = THU_MUC / "hey_nexa.joblib"
NGHI_KHUNG = 25  # nghỉ 2 s sau mỗi lần báo


class DacTrung:
    """Đặc trưng openWakeWord kiểu stream. `them(khung)` -> vectơ 16×96 phẳng của ~2 s gần nhất."""

    def __init__(self, thu_muc: Path = THU_MUC) -> None:
        from openwakeword.utils import AudioFeatures

        self.af = AudioFeatures(
            melspec_model_path=str(thu_muc / "melspectrogram.onnx"),
            embedding_model_path=str(thu_muc / "embedding_model.onnx"),
            inference_framework="onnx",
        )
        self.lam_moi()

    def lam_moi(self) -> None:
        np.random.seed(0)  # reset() của openWakeWord điền bộ đệm bằng ồn ngẫu nhiên — cố định để lặp lại được
        self.af.reset()

    def them(self, khung: np.ndarray) -> np.ndarray:
        assert khung.dtype == np.int16 and khung.shape == (KHUNG,), "khung phải là 1280 mẫu int16"
        self.af(khung)
        return self.af.get_features(SO_KHUNG).reshape(-1)

    def chuoi(self, am: np.ndarray) -> np.ndarray:
        """Đặc trưng của MỌI khung trong một đoạn âm thanh (bắt đầu từ bộ đệm mới) — dùng khi train / đo."""
        self.lam_moi()
        n = len(am) // KHUNG
        return (
            np.stack([self.them(am[i * KHUNG : (i + 1) * KHUNG]) for i in range(n)])
            if n
            else np.zeros((0, 1536))
        )


def nap_mo_hinh(tep: Path = MO_HINH) -> dict[str, Any]:
    import joblib

    return joblib.load(tep)


def diem(mo_hinh: Any, vec: np.ndarray) -> np.ndarray:
    """Điểm "hey nexa" cho từng hàng đặc trưng."""
    return mo_hinh.predict_proba(np.atleast_2d(vec))[:, 1]


def dem_bao(cac_diem: np.ndarray, nguong: float, lien: int = 1, nghi: int = NGHI_KHUNG) -> list[int]:
    """Vị trí khung báo thức: `lien` khung LIỀN NHAU cùng ≥ ngưỡng (câu "hey nexa" thật giữ điểm cao qua vài
    khung, báo nhầm thường chỉ là một khung vọt lên); sau mỗi lần báo bỏ qua `nghi` khung."""
    bao, cho, chuoi = [], 0, 0
    for i, d in enumerate(cac_diem):
        chuoi = chuoi + 1 if d >= nguong else 0
        if cho:
            cho -= 1
            continue
        if chuoi >= lien:
            bao.append(i)
            cho, chuoi = nghi, 0
    return bao


class BoDanhThuc:
    """Nghe liên tục từng khung 80 ms, trả về True ở khung báo thức (cùng luật với `dem_bao`)."""

    def __init__(self, tep: Path = MO_HINH, nguong: float | None = None) -> None:
        goi = nap_mo_hinh(tep)
        self.mo_hinh, self.nguong = goi["mo_hinh"], float(nguong if nguong is not None else goi["nguong"])
        self.lien = int(goi.get("lien", 1))
        self.dt = DacTrung()
        self._cho = self._chuoi = 0

    def lam_moi(self) -> None:
        self.dt.lam_moi()
        self._cho = self._chuoi = 0

    def nghe(self, khung: np.ndarray) -> tuple[float, bool]:
        d = float(diem(self.mo_hinh, self.dt.them(khung))[0])
        self._chuoi = self._chuoi + 1 if d >= self.nguong else 0
        if self._cho:
            self._cho -= 1
            return d, False
        if self._chuoi >= self.lien:
            self._cho, self._chuoi = NGHI_KHUNG, 0
            return d, True
        return d, False


class ThuLenh:
    """Gom âm thanh câu lệnh sau khi thức. `them(khung)` -> None (đang nghe) | "xong" | "huy"."""

    def __init__(
        self,
        truoc: list[np.ndarray],
        vad: Any,
        im_khung: int = 10,
        toi_da_khung: int = 100,
        cho_noi_khung: int = 50,
        nguong_vad: float = 0.5,
    ) -> None:
        self.khung = list(truoc)
        self.vad, self.im_khung, self.toi_da, self.cho_noi, self.nguong_vad = (
            vad,
            im_khung,
            toi_da_khung,
            cho_noi_khung,
            nguong_vad,
        )
        self.da_noi, self.im, self.dem, self.dem_noi = False, 0, 0, 0
        vad.reset_states()

    def them(self, khung: np.ndarray) -> str | None:
        self.khung.append(khung)
        self.dem += 1
        if self.vad.predict(khung, frame_size=640) >= self.nguong_vad:
            self.da_noi, self.im = True, 0
        elif self.da_noi:
            self.im += 1
        if self.da_noi:
            self.dem_noi += 1
            if self.im >= self.im_khung:
                return "xong"
            # độ dài tối đa tính TỪ LÚC BẮT ĐẦU NÓI — trước đây tính từ lúc bắt đầu chờ: ở chế độ nghe tiếp
            # nói muộn thì câu bị cắt sớm (bắt đầu ở giây 5 chỉ còn 3 s); chờ quá 8 s thì im lặng bị nhận dạng
            return "xong" if self.dem_noi >= self.toi_da else None
        return "huy" if self.dem >= self.cho_noi else None

    def am_thanh(self) -> np.ndarray:
        return np.concatenate(self.khung) if self.khung else np.zeros(0, np.int16)


def bo_dem_truoc(so_khung: int = 4) -> deque[np.ndarray]:
    """Giữ vài khung gần nhất (~0,3 s) để ghép vào đầu câu lệnh khi thức."""
    return deque(maxlen=so_khung)


# Đầu chữ STT có thể còn đuôi cụm gọi ("nét xa", "nexa ơi"...) vì giữ 0,3 s trước khung báo -> cắt trước
# khi gửi model.
# CHỈ cắt khi có phần giống "nexa"; "hãy mở cốp" giữ nguyên.
_DAU_GOI = re.compile(
    r"^\s*(?:(?:hey|hây|hay|hê|hãy|hai|ê|ây)\s+)?"
    r"(?:nexa\w*|nếch|néc|nét|nếc|lếch|lách|ních)(?:[\s-]*(?:xa|xà|sa|ra|xả))?"
    r"(?:\s+ơi)?\s*[,.!?]*\s*",
    re.IGNORECASE,
)


def bo_cum_goi(chu: str) -> str:
    return _DAU_GOI.sub("", chu, count=1).strip()
