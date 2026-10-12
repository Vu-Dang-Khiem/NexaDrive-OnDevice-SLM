"""Mô phỏng kết quả trả về của từng hàm — dùng để tạo dữ liệu "loại B" (câu trả lời)
ở Phase 2, và làm bản nháp cho `serve/vehicle_state.py` thật ở Phase 5.

Không gọi model, không cần GPU — thuần logic + số ngẫu nhiên có seed để tái lập được.

    from gen.simulate import simulate_result
    simulate_result("get_fuel_level", {}, seed=42)
    # -> {"fuel_percent": 63, "range_km": 264.6, "needs_refuel": False, "threshold_percent": 15}
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from schema.loader import CATALOG

Args = dict[str, Any]
Rng = random.Random
Simulator = Callable[[Args, Rng], dict[str, Any]]


# ── nhóm truy vấn trạng thái ─────────────────────────────────────────────────
def _get_fuel_level(args: Args, rng: Rng) -> dict:
    pct = rng.randint(3, 98)
    return {
        "fuel_percent": pct,
        "range_km": round(pct * 4.2, 1),
        "needs_refuel": pct < 15,
        "threshold_percent": 15,
    }


def _get_battery_range(args: Args, rng: Rng) -> dict:
    pct = rng.randint(3, 98)
    return {"battery_percent": pct, "range_km": round(pct * 3.5, 1), "charging": rng.random() < 0.15}


def _get_tire_pressure(args: Args, rng: Rng) -> dict:
    wheels = ["front_left", "front_right", "rear_left", "rear_right"]
    only = args.get("wheel")
    target = [only] if only else wheels
    pressures = {w: round(rng.uniform(26, 36), 1) for w in target}
    low = [w for w, p in pressures.items() if p < 30]
    return {"pressures_psi": pressures, "low": low, "recommended_psi": 32.0}


def _get_vehicle_status(args: Args, rng: Rng) -> dict:
    doors = ["driver", "passenger", "rear_left", "rear_right"]
    open_doors = rng.sample(doors, k=rng.randint(0, 1))
    return {
        "doors_locked": len(open_doors) == 0 and rng.random() < 0.8,
        "doors_open": open_doors,
        "trunk_open": rng.random() < 0.1,
        "windows": {"front_left": 0, "front_right": 0, "rear_left": 0, "rear_right": 0},
        "lights": rng.choice(["off", "auto", "low_beam"]),
        "climate_on": rng.random() < 0.5,
        "fuel_percent": rng.randint(10, 95),
        "battery_percent": rng.randint(10, 95),
    }


def _get_odometer(args: Args, rng: Rng) -> dict:
    unit = args.get("unit", "km")
    km = rng.randint(1_000, 120_000)
    value = km if unit == "km" else round(km * 0.621371, 1)
    return {"odometer": value, "unit": unit}


def _get_range_estimate(args: Args, rng: Rng) -> dict:
    total = round(rng.uniform(80, 550), 1)
    trip = args.get("trip_km")
    if trip is None:
        return {"total_range_km": total, "reachable": True, "margin_km": total}
    margin = round(total - float(trip), 1)
    return {"total_range_km": total, "reachable": margin >= 0, "margin_km": margin}


# ── nhóm điều hoà ────────────────────────────────────────────────────────────
def _set_ac(args: Args, rng: Rng) -> dict:
    return {
        "ac_on": args.get("state", "on") == "on",
        "temperature": args.get("temperature", min(30, max(16, 24 + args.get("delta", 0)))),
        "zone": args.get("zone", "all"),
        "mode": args.get("mode", "auto"),
    }


def _set_fan_speed(args: Args, rng: Rng) -> dict:
    return {"fan_level": args.get("level", min(7, max(0, 3 + args.get("delta", 0))))}


def _set_defrost(args: Args, rng: Rng) -> dict:
    return {"defrost_on": args.get("state", "on") == "on", "zone": args.get("zone", "front")}


# ── nhóm cửa / khoang ────────────────────────────────────────────────────────
def _lock_doors(args: Args, rng: Rng) -> dict:
    return {"doors_locked": True}


def _unlock_doors(args: Args, rng: Rng) -> dict:
    door = args.get("door", "all")
    tat_ca = ["driver", "passenger", "rear_left", "rear_right"]
    return {"doors_locked": False, "unlocked": tat_ca if door == "all" else [door]}


def _open_trunk(args: Args, rng: Rng) -> dict:
    return {"trunk_open": True}


def _close_trunk(args: Args, rng: Rng) -> dict:
    return {"trunk_open": False}


def _open_charge_port(args: Args, rng: Rng) -> dict:
    return {"charge_port_open": True}


def _open_fuel_flap(args: Args, rng: Rng) -> dict:
    return {"fuel_flap_open": True}


# ── nhóm cửa sổ / nóc ────────────────────────────────────────────────────────
def _control_window(args: Args, rng: Rng) -> dict:
    position = args.get("position", "up")
    amount = args.get("amount", 100)
    pct = amount if position == "down" else 100 - amount
    return {"window": args.get("window", "all"), "position": position, "opening_percent": pct}


def _operate_sunroof(args: Args, rng: Rng) -> dict:
    action = args.get("action", "open")
    intensity = args.get("intensity", 100 if action == "open" else 0)
    return {"sunroof_state": action, "intensity_percent": intensity}


# ── nhóm đèn ─────────────────────────────────────────────────────────────────
def _set_headlights(args: Args, rng: Rng) -> dict:
    return {"headlights": args.get("mode", "auto")}


def _set_hazard_lights(args: Args, rng: Rng) -> dict:
    return {"hazards_on": args.get("state", "on") == "on"}


# ── nhóm media ───────────────────────────────────────────────────────────────
def _set_volume(args: Args, rng: Rng) -> dict:
    if "set_value" in args:
        return {"volume": args["set_value"]}
    current = rng.randint(5, 30)
    delta = args.get("delta", 0)
    return {"volume": max(0, min(40, current + delta))}


def _play_media(args: Args, rng: Rng) -> dict:
    query = args.get("query", "Playlist đề xuất")
    return {"playing": True, "source": args.get("source", "streaming"), "now_playing": query}


# ── nhóm dẫn đường / thông tin ───────────────────────────────────────────────
def _navigate_to(args: Args, rng: Rng) -> dict:
    dist = round(rng.uniform(1.5, 45), 1)
    # Kết quả phải phản ánh yêu cầu tránh đường, không thì câu trả lời sinh ra sẽ bỏ qua
    # đúng cái người dùng dặn (thẩm định chéo chê 22 mẫu vì lý do này).
    tranh = {"tolls": ", tránh trạm thu phí", "highways": ", tránh cao tốc"}.get(
        str(args.get("avoid", "none")), ""
    )
    return {
        "destination": args.get("destination", ""),
        "distance_km": dist,
        "eta_min": round(dist / 35 * 60, 1),
        "route_summary": f"Đi theo tuyến chính{tranh}, {dist} km",
    }


def _find_nearby(args: Args, rng: Rng) -> dict:
    """LỖI ĐÃ SỬA: trước đây bỏ qua `radius_km` nên trả về địa điểm NGOÀI bán kính
    người dùng xin ("trong bán kính 1km" -> trả về 4.9 km). Câu trả lời sinh từ kết quả
    đó dạy model nói ngược với yêu cầu — thẩm định chéo bằng Llama bắt được lỗi này."""
    cat = args.get("category", "gas_station")
    ban_kinh = float(args.get("radius_km", 5))
    n = rng.randint(1, 4)
    ket_qua = [
        f"{cat}_{i + 1} — {round(rng.uniform(0.3, max(0.4, ban_kinh)), 1)} km" for i in range(n)
    ]
    return {"category": cat, "results": ket_qua}


def _get_weather(args: Args, rng: Rng) -> dict:
    desc = rng.choice(["nắng", "nhiều mây", "mưa nhẹ", "mưa to", "âm u"])
    return {
        "location": args.get("location", "vị trí hiện tại"),
        "date": f"+{args.get('day', 0)}d",
        "description": desc,
        "temp_c": round(rng.uniform(18, 36), 1),
        "precipitation_mm": round(rng.uniform(0, 20), 1) if "mưa" in desc else 0.0,
    }


def _irrelevant_function(args: Args, rng: Rng) -> dict:
    return {"handled": True}


_SIMULATORS: dict[str, Simulator] = {
    "get_fuel_level": _get_fuel_level,
    "get_battery_range": _get_battery_range,
    "get_tire_pressure": _get_tire_pressure,
    "get_vehicle_status": _get_vehicle_status,
    "get_odometer": _get_odometer,
    "get_range_estimate": _get_range_estimate,
    "set_ac": _set_ac,
    "set_fan_speed": _set_fan_speed,
    "set_defrost": _set_defrost,
    "lock_doors": _lock_doors,
    "unlock_doors": _unlock_doors,
    "open_trunk": _open_trunk,
    "close_trunk": _close_trunk,
    "open_charge_port": _open_charge_port,
    "open_fuel_flap": _open_fuel_flap,
    "control_window": _control_window,
    "operate_sunroof": _operate_sunroof,
    "set_headlights": _set_headlights,
    "set_hazard_lights": _set_hazard_lights,
    "set_volume": _set_volume,
    "play_media": _play_media,
    "navigate_to": _navigate_to,
    "find_nearby": _find_nearby,
    "get_weather": _get_weather,
    "irrelevant_function": _irrelevant_function,
}


def simulate_result(name: str, args: Args, seed: int | None = None) -> dict[str, Any]:
    """Trả về kết quả giả lập hợp lý cho 1 lời gọi hàm — dùng để sinh data loại B
    (Phase 2) và làm bản nháp logic cho Vehicle State Service thật (Phase 5).
    """
    if name not in CATALOG.by_name:
        raise KeyError(f"không có hàm tên {name!r}")
    handler = _SIMULATORS.get(name)
    if handler is None:  # phòng khi thêm hàm mới vào schema mà quên viết simulator
        raise NotImplementedError(f"chưa có simulator cho hàm {name!r}")
    return handler(args, random.Random(seed))


def all_functions_have_simulator() -> list[str]:
    """Trả về danh sách tên hàm trong catalog CHƯA có simulator — rỗng nghĩa là đủ."""
    return [f.name for f in CATALOG.functions if f.name not in _SIMULATORS]
