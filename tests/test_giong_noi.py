"""`serve/giong_noi.py` với STT / TTS giả — kiểm đường đi, không cần model."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from serve.danh_thuc import KHUNG, bo_cum_goi
from serve.giong_noi import STT_FT, DocTungDoan, chon_stt, pcm16, tao_app

nhan: list[str] = []


def test_chon_stt_mac_dinh_ban_fine_tune_va_ve_goc_duoc():
    assert chon_stt(None, co_ft=True) == str(STT_FT)
    assert chon_stt("goc", co_ft=True) is None  # NEXA_STT=goc -> PhoWhisper-small gốc
    assert chon_stt(None, co_ft=False) is None  # máy chưa có bản fine-tune -> tự về bản gốc
    assert chon_stt("ft", co_ft=False) is None


def _nghe(tep: str) -> str:
    nhan.append(tep)
    return "bật điều hoà hai hai độ."


client = TestClient(tao_app(_nghe, lambda text: b"RIFF" + text.encode()))


def test_stt_chuan_hoa_chu_nhu_luc_cham_chuoi():
    r = client.post("/stt", content=b"\x1aE\xdf\xa3webm", headers={"Content-Type": "audio/webm"})
    assert r.status_code == 200
    assert r.json()["text"] == "Bật điều hoà 22 độ" and r.json()["chu_stt"] == "bật điều hoà hai hai độ."
    assert nhan[-1].endswith(".webm")


def test_stt_khong_co_am_thanh_thi_bao_loi():
    assert client.post("/stt", content=b"").status_code == 400


def test_tts_tra_wav():
    r = client.post("/tts", json={"text": "Đã mở cốp sau."})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and r.content.startswith(b"RIFF")
    assert client.post("/tts", json={"text": "  "}).status_code == 400


# ── đọc từng đoạn /tts_stream ──────────────────────────────────────────────
def _sinh_gia(text: str):
    """Giả VieNeu infer_stream: ba đoạn float, đoạn giữa rỗng (thư viện có lúc trả đoạn rỗng)."""
    yield np.full(4, 0.5, np.float32)
    yield np.zeros(0, np.float32)
    yield np.full(2, -2.0, np.float32)  # vượt biên -> phải cắt, không tràn số


def test_pcm16_cat_bien():
    ra = np.frombuffer(pcm16(np.array([0.0, 1.0, -1.0, 3.0, -3.0], np.float32)), dtype="<i2")
    assert ra.tolist() == [0, 32767, -32767, 32767, -32767]


def test_tts_stream_gui_tung_doan_va_tan_so():
    doc = DocTungDoan(_sinh_gia, 48000)
    c = TestClient(tao_app(_nghe, lambda text: b"RIFF", doc_dong=doc))
    r = c.post("/tts_stream", json={"text": "Đã mở cốp sau."})
    assert r.status_code == 200 and r.headers["x-tan-so"] == "48000"
    am = np.frombuffer(r.content, dtype="<i2")
    assert am.tolist() == [16383] * 4 + [-32767] * 2  # đoạn rỗng bị bỏ, thứ tự giữ nguyên
    assert c.post("/tts_stream", json={"text": " "}).status_code == 400


def test_cau_moi_thi_cau_cu_dung_tao():
    doc = DocTungDoan(_sinh_gia, 48000)
    _, cu = doc("câu cũ")
    assert next(cu) == pcm16(np.full(4, 0.5, np.float32))
    _, moi = doc("câu mới")  # câu mới tới trong lúc câu cũ còn đang tạo
    assert list(cu) == []  # câu cũ dừng ở đoạn kế tiếp, nhả khoá
    assert len(b"".join(moi)) == 12  # câu mới đọc đủ (4 + 2 mẫu × 2 byte)


def test_khong_bat_doc_tung_doan_thi_404():
    assert client.post("/tts_stream", json={"text": "Đã mở cốp sau."}).status_code == 404


# ── wake-word /wake (Phase 6b) với bộ báo thức + VAD giả ─────────────────────
class _BoGia:
    """Báo thức ở khung có biên độ 5000 (giả "hey nexa")."""

    def __init__(self) -> None:
        self.lam_moi_lan = 0

    def lam_moi(self) -> None:
        self.lam_moi_lan += 1

    def nghe(self, khung: np.ndarray) -> tuple[float, bool]:
        return 1.0, bool(khung[0] == 5000)


class _VadGia:
    """Có tiếng nói khi khung khác 0."""

    def reset_states(self) -> None:
        pass

    def predict(self, khung: np.ndarray, frame_size: int = 640) -> float:
        return 1.0 if khung.any() else 0.0


def _khung(gia_tri: int, so: int = 1) -> bytes:
    return np.full(KHUNG * so, gia_tri, dtype="<i2").tobytes()


def _app_wake(chu_stt: str) -> TestClient:
    return TestClient(tao_app(lambda tep: chu_stt, lambda text: b"RIFF", lambda: (_BoGia(), _VadGia())))


def test_wake_thuc_roi_gom_lenh_den_khi_im():
    with _app_wake("hây nét xa ơi, mở cốp giúp mình.").websocket_connect("/wake") as ws:
        ws.send_bytes(_khung(0, 3) + _khung(5000))
        assert ws.receive_json() == {"type": "thuc"}
        ws.send_bytes(_khung(300, 8) + _khung(0, 10))  # nói 8 khung rồi im 10 khung (0,8 s)
        tin = ws.receive_json()
        assert tin["type"] == "lenh" and tin["text"] == "Mở cốp giúp mình"  # đã cắt cụm gọi + chuẩn hoá
        assert tin["chu_stt"].startswith("hây nét xa")


def test_wake_thuc_ma_khong_ai_noi_thi_chi_goi():
    """Chỉ gọi "hey nexa" rồi im 2 s -> giao diện chào "Dạ, mình nghe đây" rồi nghe tiếp."""
    with _app_wake("").websocket_connect("/wake") as ws:
        ws.send_bytes(_khung(5000))
        assert ws.receive_json() == {"type": "thuc"}
        ws.send_bytes(_khung(0, 25))  # 2 s im
        assert ws.receive_json() == {"type": "chi_goi"}


def test_wake_chi_con_duoi_cum_goi_thi_chi_goi():
    """Đuôi "nexa" lọt vào câu lệnh: STT ra "hây nét xa" (hoặc một chữ thừa) -> không gửi model."""
    with _app_wake("hây nét xa").websocket_connect("/wake") as ws:
        ws.send_bytes(_khung(5000))
        assert ws.receive_json() == {"type": "thuc"}
        ws.send_bytes(_khung(300, 3) + _khung(0, 10))
        assert ws.receive_json() == {"type": "chi_goi", "chu_stt": "hây nét xa"}


def test_nghe_tiep_khong_can_goi_lai():
    """Sau khi đọc xong câu trả lời: nói câu lệnh tiếp theo (kể cả "có") mà không cần "hey nexa"."""
    with _app_wake("có").websocket_connect("/wake") as ws:
        ws.send_text('{"type": "nghe_tiep"}')
        ws.send_bytes(_khung(300, 4) + _khung(0, 10))  # không có khung báo thức nào
        tin = ws.receive_json()
        assert tin["type"] == "lenh" and tin["text"] == "Có" and tin["tiep"] is True


def test_nghe_tiep_ma_im_thi_quay_ve_cho_goi():
    with _app_wake("mở cốp").websocket_connect("/wake") as ws:
        ws.send_text('{"type": "nghe_tiep"}')
        ws.send_bytes(_khung(0, 75))  # 6 s im
        assert ws.receive_json() == {"type": "het_tiep"}
        ws.send_bytes(_khung(5000))  # quay về: lại nghe "hey nexa" như thường
        assert ws.receive_json() == {"type": "thuc"}


def test_wake_tam_dung_thi_khong_nghe():
    with _app_wake("mở cốp").websocket_connect("/wake") as ws:
        ws.send_text('{"type": "tam_dung"}')
        ws.send_bytes(_khung(5000, 2))  # loa đang đọc: không báo
        ws.send_text('{"type": "tiep_tuc"}')
        ws.send_bytes(_khung(5000))
        assert ws.receive_json() == {"type": "thuc"}


def test_wake_tiep_tuc_thua_khong_xoa_lenh_dang_gom():
    """Thức -> giao diện dừng loa -> sự kiện dừng loa gửi "tiep_tuc" khi KHÔNG tạm dừng: phải bỏ qua."""
    with _app_wake("mở cốp").websocket_connect("/wake") as ws:
        ws.send_bytes(_khung(5000))
        assert ws.receive_json() == {"type": "thuc"}
        ws.send_text('{"type": "tiep_tuc"}')
        ws.send_bytes(_khung(300, 5) + _khung(0, 10))
        assert ws.receive_json()["text"] == "Mở cốp"


def test_luu_am_thanh_va_chu_de_chan_doan(tmp_path):
    app = tao_app(lambda tep: "mở cốp", lambda text: b"RIFF", lambda: (_BoGia(), _VadGia()), tmp_path)
    c = TestClient(app)
    c.post("/stt", content=b"\x1aE\xdf\xa3webm", headers={"Content-Type": "audio/webm"})
    with c.websocket_connect("/wake") as ws:
        ws.send_bytes(_khung(5000))
        ws.receive_json()
        ws.send_bytes(_khung(300, 3) + _khung(0, 10))
        ws.receive_json()
    dong = (tmp_path / "nhat_ky.jsonl").read_text(encoding="utf-8").splitlines()
    nhat_ky = [json.loads(x) for x in dong]
    assert [d["nguon"] for d in nhat_ky] == ["mic", "wake"] and nhat_ky[0]["chu_stt"] == "mở cốp"
    assert (tmp_path / nhat_ky[0]["tep"]).read_bytes().endswith(b"webm")
    assert (tmp_path / nhat_ky[1]["tep"]).read_bytes().startswith(b"RIFF")


def test_khong_co_bo_bao_thuc_thi_khong_co_wake():
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/wake"):
        pass


@pytest.mark.parametrize("chu,ra", [
    ("hây nét xa ơi, mở cốp", "mở cốp"),
    ("Nexa ơi bật điều hoà", "bật điều hoà"),
    ("nét xa. hạ kính ghế phụ", "hạ kính ghế phụ"),
    ("hãy mở cốp giúp mình", "hãy mở cốp giúp mình"),  # không có "nexa" -> giữ nguyên
    ("mở cốp", "mở cốp"),
])
def test_bo_cum_goi(chu, ra):
    assert bo_cum_goi(chu) == ra
