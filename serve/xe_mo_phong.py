"""Xe MÔ PHỎNG có trạng thái — nơi 24 hàm (tool) thực sự "chạy" (PLAN.md Phase 5).

Khác `gen/simulate.py` (dùng lúc sinh dữ liệu: mỗi lần gọi trả số ngẫu nhiên, không nhớ gì), xe ở đây NHỚ
trạng thái: mở cốp xong thì hỏi tình trạng xe phải thấy cốp đang mở. Kết quả trả về giữ ĐÚNG tên trường
như lúc sinh dữ liệu train (`returns` trong schema) để lượt 2 của model đọc quen.

Ba hàm cần dịch vụ bên ngoài (dẫn đường, tìm địa điểm, thời tiết) chưa nối API thật: mượn `gen/simulate.py`
với hạt giống tăng dần -> chạy lại y hệt, test được.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from gen.simulate import simulate_result
from schema.loader import CATALOG

KINH = ("front_left", "front_right", "rear_left", "rear_right")
CUA_XE = ("driver", "passenger", "rear_left", "rear_right")
KM_MOI_PHAN_TRAM_XANG = 4.2
KM_MOI_PHAN_TRAM_PIN = 3.5
NGUONG_DO_XANG = 15
NGUONG_LOP_THAP = 30.0
AP_SUAT_KHUYEN_NGHI = 32.0


@dataclass
class TrangThaiXe:
    """Toàn bộ trạng thái xe — cũng là thứ gửi cho frontend (StateEvent) sau mỗi lần thay đổi."""

    # công tắc cho demo (luật an toàn đọc hai cờ này)
    dang_chay: bool = False
    khoa_tre_em: bool = False
    # năng lượng / quãng đường
    fuel_percent: int = 62
    battery_percent: int = 80
    charging: bool = False
    odometer_km: float = 23450.0
    tire_psi: dict[str, float] = field(
        default_factory=lambda: {
            "front_left": 32.5,
            "front_right": 32.0,
            "rear_left": 29.5,
            "rear_right": 31.8,
        }
    )
    # cửa / khoang
    doors_locked: bool = True
    doors_open: list[str] = field(default_factory=list)
    trunk_open: bool = False
    charge_port_open: bool = False
    fuel_flap_open: bool = False
    # kính (% đang mở) / cửa sổ trời
    windows: dict[str, int] = field(default_factory=lambda: dict.fromkeys(KINH, 0))
    sunroof_state: str = "close"
    sunroof_percent: int = 0
    # điều hoà
    ac_on: bool = False
    ac_temperature: int = 24
    ac_zone: str = "all"
    ac_mode: str = "auto"  # cool / heat / auto
    fan_level: int = 0
    defrost_on: bool = False
    defrost_zone: str = "front"
    # đèn
    headlights: str = "auto"
    hazards_on: bool = False
    # giải trí
    volume: int = 12
    media_playing: bool = False
    media_source: str = "streaming"
    now_playing: str = ""
    # dẫn đường
    destination: str | None = None
    location: str = "Hà Nội"


Args = dict[str, Any]


class XeMoPhong:
    """24 tool chạy trên một chiếc xe mô phỏng. `thuc_thi` nhận tham số ĐÃ qua validator."""

    def __init__(self, trang_thai: TrangThaiXe | None = None, hat_giong: int = 0) -> None:
        self.s = trang_thai or TrangThaiXe()
        self._hat_giong = hat_giong
        self._bang: dict[str, Callable[[Args], dict[str, Any]]] = {
            f.name: getattr(self, f"_{f.name}") for f in CATALOG.functions
        }

    # ── giao diện ─────────────────────────────────────────────────────────
    def thuc_thi(self, ten: str, args: Args) -> dict[str, Any]:
        if ten not in self._bang:
            raise KeyError(f"không có tool tên {ten!r}")
        return self._bang[ten](args)

    def anh_chup(self) -> dict[str, Any]:
        return asdict(self.s)

    def dong_nap(self, nap: str) -> dict[str, Any]:
        """Đóng nắp bình xăng ("xang") / nắp cổng sạc ("sac") — KHÔNG phải tool của model (schema chỉ có hàm
        mở); router gọi theo luật runtime `serve/dong_nap.py`. Trả về {trường: False, "da_dong_san": ...}."""
        truong = "fuel_flap_open" if nap == "xang" else "charge_port_open"
        da_dong_san = not getattr(self.s, truong)
        setattr(self.s, truong, False)
        return {truong: False, "da_dong_san": da_dong_san}

    def _ngoai(self, ten: str, args: Args) -> dict[str, Any]:
        """Dịch vụ bên ngoài (chưa nối API thật) — hạt giống tăng dần để tái lập được."""
        self._hat_giong += 1
        return simulate_result(ten, args, seed=self._hat_giong)

    # ── tra cứu (chỉ đọc) ─────────────────────────────────────────────────
    def _tam_xang(self) -> float:
        return round(self.s.fuel_percent * KM_MOI_PHAN_TRAM_XANG, 1)

    def _tam_pin(self) -> float:
        return round(self.s.battery_percent * KM_MOI_PHAN_TRAM_PIN, 1)

    def _get_fuel_level(self, a: Args) -> dict[str, Any]:
        return {
            "fuel_percent": self.s.fuel_percent,
            "range_km": self._tam_xang(),
            "needs_refuel": self.s.fuel_percent < NGUONG_DO_XANG,
            "threshold_percent": NGUONG_DO_XANG,
        }

    def _get_battery_range(self, a: Args) -> dict[str, Any]:
        return {
            "battery_percent": self.s.battery_percent,
            "range_km": self._tam_pin(),
            "charging": self.s.charging,
        }

    def _get_tire_pressure(self, a: Args) -> dict[str, Any]:
        banh = [a["wheel"]] if a.get("wheel") else list(KINH)
        ap = {b: self.s.tire_psi[b] for b in banh}
        return {
            "pressures_psi": ap,
            "low": [b for b, p in ap.items() if p < NGUONG_LOP_THAP],
            "recommended_psi": AP_SUAT_KHUYEN_NGHI,
        }

    def _get_vehicle_status(self, a: Args) -> dict[str, Any]:
        return {
            "doors_locked": self.s.doors_locked,
            "doors_open": list(self.s.doors_open),
            "trunk_open": self.s.trunk_open,
            "windows": dict(self.s.windows),
            "lights": self.s.headlights,
            "climate_on": self.s.ac_on,
            "fuel_percent": self.s.fuel_percent,
            "battery_percent": self.s.battery_percent,
        }

    def _get_odometer(self, a: Args) -> dict[str, Any]:
        don_vi = a.get("unit", "km")
        gia_tri = self.s.odometer_km if don_vi == "km" else round(self.s.odometer_km * 0.621371, 1)
        return {"odometer": gia_tri, "unit": don_vi}

    def _get_range_estimate(self, a: Args) -> dict[str, Any]:
        tong = round(self._tam_xang() + self._tam_pin(), 1)
        if a.get("trip_km") is None:
            return {"total_range_km": tong, "reachable": True, "margin_km": tong}
        du = round(tong - float(a["trip_km"]), 1)
        return {"total_range_km": tong, "reachable": du >= 0, "margin_km": du}

    # ── điều hoà ──────────────────────────────────────────────────────────
    def _set_ac(self, a: Args) -> dict[str, Any]:
        self.s.ac_on = a["state"] == "on"
        if "temperature" in a:
            self.s.ac_temperature = a["temperature"]
        elif "delta" in a:  # tương đối, kẹp trong 16–30
            self.s.ac_temperature = min(30, max(16, self.s.ac_temperature + a["delta"]))
        if "mode" in a:  # không nói chế độ -> giữ chế độ đang có
            self.s.ac_mode = a["mode"]
        self.s.ac_zone = a.get("zone", "all")
        if self.s.ac_on and self.s.fan_level == 0:
            self.s.fan_level = 3  # bật điều hoà mà quạt đang tắt thì chẳng có gió
        return {"ac_on": self.s.ac_on, "temperature": self.s.ac_temperature, "zone": self.s.ac_zone,
                "mode": self.s.ac_mode}

    def _set_fan_speed(self, a: Args) -> dict[str, Any]:
        # `level` tuyệt đối hoặc `delta` tương đối (v6), kẹp 0–7
        self.s.fan_level = a["level"] if "level" in a else min(7, max(0, self.s.fan_level + a["delta"]))
        return {"fan_level": self.s.fan_level}

    def _set_defrost(self, a: Args) -> dict[str, Any]:
        self.s.defrost_on = a["state"] == "on"
        self.s.defrost_zone = a.get("zone", "front")
        return {"defrost_on": self.s.defrost_on, "zone": self.s.defrost_zone}

    # ── cửa / khoang ──────────────────────────────────────────────────────
    def _lock_doors(self, a: Args) -> dict[str, Any]:
        self.s.doors_locked = True
        return {"doors_locked": True}

    def _unlock_doors(self, a: Args) -> dict[str, Any]:
        cua = a.get("door", "all")
        self.s.doors_locked = False
        return {"doors_locked": False, "unlocked": list(CUA_XE) if cua == "all" else [cua]}

    def _open_trunk(self, a: Args) -> dict[str, Any]:
        self.s.trunk_open = True
        return {"trunk_open": True}

    def _close_trunk(self, a: Args) -> dict[str, Any]:
        self.s.trunk_open = False
        return {"trunk_open": False}

    def _open_charge_port(self, a: Args) -> dict[str, Any]:
        self.s.charge_port_open = True
        return {"charge_port_open": True}

    def _open_fuel_flap(self, a: Args) -> dict[str, Any]:
        self.s.fuel_flap_open = True
        return {"fuel_flap_open": True}

    # ── kính / cửa sổ trời ────────────────────────────────────────────────
    def _control_window(self, a: Args) -> dict[str, Any]:
        """`amount` = % hành trình (mặc định 100 = mở/đóng hết).

        Hạ -> kính mở `amount`%; nâng -> kính còn mở 100-amount%."""
        nhom = {"all": list(KINH), "front": ["front_left", "front_right"],
                "rear": ["rear_left", "rear_right"]}
        kinh = nhom.get(a["window"], [a["window"]])
        muc = a.get("amount", 100)
        mo = muc if a["position"] == "down" else 100 - muc
        for k in kinh:
            self.s.windows[k] = mo
        return {"window": a["window"], "position": a["position"], "opening_percent": mo}

    def _operate_sunroof(self, a: Args) -> dict[str, Any]:
        hanh_dong = a["action"]
        mac_dinh = {"open": 100, "tilt": 30, "close": 0}[hanh_dong]
        self.s.sunroof_state = hanh_dong
        self.s.sunroof_percent = 0 if hanh_dong == "close" else a.get("intensity", mac_dinh)
        return {"sunroof_state": hanh_dong, "intensity_percent": self.s.sunroof_percent}

    # ── đèn ───────────────────────────────────────────────────────────────
    def _set_headlights(self, a: Args) -> dict[str, Any]:
        self.s.headlights = a["mode"]
        return {"headlights": self.s.headlights}

    def _set_hazard_lights(self, a: Args) -> dict[str, Any]:
        self.s.hazards_on = a["state"] == "on"
        return {"hazards_on": self.s.hazards_on}

    # ── giải trí ──────────────────────────────────────────────────────────
    def _set_volume(self, a: Args) -> dict[str, Any]:
        moi = a["set_value"] if "set_value" in a else self.s.volume + a["delta"]
        self.s.volume = max(0, min(40, moi))
        return {"volume": self.s.volume}

    def _play_media(self, a: Args) -> dict[str, Any]:
        self.s.media_playing = True
        self.s.media_source = a.get("source", "streaming")
        self.s.now_playing = a.get("query", "Playlist đề xuất")
        return {"playing": True, "source": self.s.media_source, "now_playing": self.s.now_playing}

    # ── dẫn đường / thông tin (dịch vụ ngoài, mô phỏng) ───────────────────
    def _navigate_to(self, a: Args) -> dict[str, Any]:
        self.s.destination = a["destination"]
        return self._ngoai("navigate_to", a)

    def _find_nearby(self, a: Args) -> dict[str, Any]:
        return self._ngoai("find_nearby", a)

    def _get_weather(self, a: Args) -> dict[str, Any]:
        return self._ngoai("get_weather", {"location": self.s.location, **a})

    def _irrelevant_function(self, a: Args) -> dict[str, Any]:
        return {"handled": True}
