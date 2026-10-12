"""Backend Phase 5: health, schema endpoint, WebSocket chạy qua agent (model giả)."""

from fastapi.testclient import TestClient

from serve.app import app

client = TestClient(app)


def test_health_tra_ve_so_ham_dung():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["functions"] == 25
    assert body["special_tokens"] == 27  # 25 hàm + <nexa_end> + <nexa_ask>
    assert body["mo_hinh"] == "MoHinhGia"


def test_endpoint_scene_actions_khop_voi_catalog():
    r = client.get("/schema/actions")
    assert r.status_code == 200
    body = r.json()
    assert "open_trunk" in body["actions"]
    assert body["actions"]["open_trunk"]["camera"] == "rear"
    assert body["actions"]["open_trunk"]["animation"] == "trunk_open"
    assert body["actions"]["open_trunk"]["nexa"] == "<nexa_11>"


def test_ket_noi_nhan_ngay_trang_thai_xe():
    with client.websocket_connect("/ws") as ws:
        dau = ws.receive_json()
    assert dau["type"] == "state" and dau["vehicle"]["trunk_open"] is False


def test_websocket_mo_cop_chay_ca_luong():
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # trạng thái ban đầu
        ws.send_json({"type": "user_message", "text": "mở cốp xe", "lang": "vi"})
        tl = ws.receive_json()
        st = ws.receive_json()
    assert tl["type"] == "assistant_turn" and tl["calls"][0]["name"] == "open_trunk"
    assert st["type"] == "state" and st["vehicle"]["trunk_open"] is True


def test_cong_tac_dang_chay_lam_luat_an_toan_co_hieu_luc():
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "dieu_khien_xe", "dang_chay": True})
        assert ws.receive_json()["vehicle"]["dang_chay"] is True
        ws.send_json({"type": "user_message", "text": "mở cốp xe"})
        tl = ws.receive_json()
    assert tl["loai"] == "tu_choi"


def test_mo_ho_qua_websocket_hoi_lai():
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "user_message", "text": "nóng quá"})
        h = ws.receive_json()
        ws.send_json({"type": "user_message", "text": "có"})
        tl = ws.receive_json()
    assert h["type"] == "clarify" and h["ly_do"] == "de_xuat"
    assert tl["type"] == "assistant_turn" and tl["calls"][0]["name"] == "set_ac"


def test_websocket_payload_sai_thi_bao_loi_nhung_khong_dut_ket_noi():
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "user_message", "lang": 123})  # lang sai kiểu
        err = ws.receive_json()
        assert err["type"] == "error"
        # kết nối vẫn sống, gửi tiếp được
        ws.send_json({"type": "user_message", "text": "kể chuyện cười đi"})
        ok = ws.receive_json()
    assert ok["type"] == "assistant_turn" and ok["loai"] == "tu_choi"
