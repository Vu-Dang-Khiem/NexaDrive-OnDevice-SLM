"""Đọc functions.yaml và suy ra mọi artifact phần còn lại của hệ thống cần.

Đây là nơi DUY NHẤT đọc functions.yaml. Import từ đây:

    from schema.loader import CATALOG

    CATALOG.functions                 # list[FunctionSpec], sắp theo index
    CATALOG.by_name["open_trunk"]     # FunctionSpec
    CATALOG.nexa_to_name              # {"<nexa_11>": "open_trunk", ...}
    CATALOG.special_tokens            # token mới cần thêm vào tokenizer
    CATALOG.render_call_prompt(query)                 # Prompt A (lượt 1)
    CATALOG.render_reply_prompt(query, call, result)  # Prompt B (lượt 2)
    CATALOG.function_description_block()              # doc dài, chỉ đưa cho teacher
    CATALOG.json_schema("set_ac")                     # JSON Schema cho tham số
    CATALOG.scene_actions()                           # id camera / hoạt cảnh / HUD (endpoint /schema/actions)

Chạy như script để in tóm tắt (map token, mẫu prompt):

    python schema/loader.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SCHEMA_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCHEMA_DIR.parent
YAML_PATH = SCHEMA_DIR / "functions.yaml"

_PY_TO_JSON_TYPE = {
    "string": "string",
    "int": "integer",
    "number": "number",
    "bool": "boolean",
}


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: str
    required: bool = False
    enum: list[Any] | None = None
    minimum: float | None = None
    maximum: float | None = None
    default: Any | None = None
    desc: str = ""

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> ParamSpec:
        return cls(
            name=raw["name"],
            type=raw["type"],
            required=bool(raw.get("required", False)),
            enum=raw.get("enum"),
            minimum=raw.get("min"),
            maximum=raw.get("max"),
            default=raw.get("default"),
            desc=raw.get("desc", ""),
        )

    def signature_fragment(self) -> str:
        """Render `name` hoặc `name=default` cho chữ ký hàm dạng người đọc."""
        if self.default is not None:
            return f"{self.name}={self.default!r}"
        if not self.required:
            return f"{self.name}=None"
        return self.name

    def json_schema(self) -> dict[str, Any]:
        node: dict[str, Any] = {"type": _PY_TO_JSON_TYPE[self.type]}
        if self.enum is not None:
            node["enum"] = list(self.enum)
        if self.minimum is not None:
            node["minimum"] = self.minimum
        if self.maximum is not None:
            node["maximum"] = self.maximum
        if self.default is not None:
            node["default"] = self.default
        if self.desc:
            node["description"] = self.desc
        return node


DIEU_KIEN_AN_TOAN = frozenset({"dang_chay", "khoa_tre_em", "wake_word"})
XU_LY_AN_TOAN = frozenset({"tu_choi", "hoi_lai", "gioi_han"})


@dataclass(frozen=True)
class QuyTacAnToan:
    """Một luật an toàn lúc chạy thật — xem giải thích trường `an_toan` đầu functions.yaml."""

    khi: tuple[str, ...]  # MỌI điều kiện cùng đúng thì luật mới áp
    xu_ly: str
    ly_do: str
    neu_tham_so: dict[str, list[Any]] = field(default_factory=dict)
    tham_so: str | None = None
    toi_da: float | None = None

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> QuyTacAnToan:
        neu = {k: (v if isinstance(v, list) else [v]) for k, v in (raw.get("neu_tham_so") or {}).items()}
        khi = raw["khi"]
        return cls(
            khi=tuple(khi) if isinstance(khi, list) else (khi,),
            xu_ly=raw["xu_ly"],
            ly_do=raw["ly_do"],
            neu_tham_so=neu,
            tham_so=raw.get("tham_so"),
            toi_da=raw.get("toi_da"),
        )


@dataclass(frozen=True)
class UiAction:
    camera: str
    animation: str | None
    hud: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FunctionSpec:
    index: int
    name: str
    category: str
    description: str
    params: list[ParamSpec]
    returns: dict[str, Any]
    ui_action: UiAction
    examples: list[str] = field(default_factory=list)
    implicit_examples: list[str] = field(default_factory=list)
    # Các cách người Việt gọi bộ phận/chức năng này. Đưa THẲNG vào prompt sinh data và
    # yêu cầu teacher dùng đều, vì nếu chỉ trông vào `examples` thì teacher lặp lại đúng
    # từ trong ví dụ mồi. Đo thật: 267/267 câu của control_window dùng "cửa sổ", 0 câu
    # dùng "kính" — trong khi set_defrost chiếm 100% chữ "kính" -> model học luật sạch
    # "kính ⟹ sấy kính", nên "hạ kính bên tài" gọi nhầm hàm. Xem gen/ra_soat_tu_vung.py.
    tu_dong_nghia: list[str] = field(default_factory=list)
    # Hàm CHỈ ĐỌC: không đổi trạng thái xe -> làm sai cũng vô hại -> KHÔNG hỏi lại.
    # Hỏi lại là cơ chế chống làm sai; đọc số liệu thì không có gì để làm sai, hỏi thêm
    # chỉ làm phiền người lái ("Bạn có muốn tôi kiểm tra áp suất lốp không?").
    chi_doc: bool = False
    # Hàm mở lối vào xe: runtime không bao giờ tự đề xuất từ câu than thở.
    nhay_cam: bool = False
    # Luật an toàn lúc chạy thật (serve/an_toan.py) — không ảnh hưởng dữ liệu train.
    an_toan: list[QuyTacAnToan] = field(default_factory=list)

    @property
    def nexa_token(self) -> str:
        return f"<nexa_{self.index}>"

    def signature(self) -> str:
        inner = ", ".join(p.signature_fragment() for p in self.params)
        return f"{self.name}({inner})"

    def json_schema(self) -> dict[str, Any]:
        props = {p.name: p.json_schema() for p in self.params}
        required = [p.name for p in self.params if p.required]
        schema: dict[str, Any] = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": self.name,
            "type": "object",
            "properties": props,
            "additionalProperties": False,
        }
        if required:
            schema["required"] = required
        return schema

    def teacher_doc(self) -> str:
        """Khối docstring kiểu Python, CHỈ đưa cho teacher lúc sinh data."""
        lines = [f"def {self.signature()}:", f'    """{self.description.strip()}']
        if self.params:
            lines.append("    Tham số:")
            for p in self.params:
                bits = [p.type]
                if not p.required:
                    bits.append("tuỳ chọn")
                if p.enum:
                    bits.append("một trong: " + ", ".join(map(str, p.enum)))
                if p.minimum is not None or p.maximum is not None:
                    bits.append(f"khoảng {p.minimum}..{p.maximum}")
                lines.append(f"    - {p.name} ({', '.join(bits)}): {p.desc}")
        lines.append('    """')
        return "\n".join(lines)


@dataclass(frozen=True)
class Catalog:
    raw_meta: dict[str, Any]
    functions: list[FunctionSpec]
    camera_presets: list[str]
    hud_widgets: list[str]

    # ── tra cứu suy ra ────────────────────────────────────────────────────
    @property
    def by_name(self) -> dict[str, FunctionSpec]:
        return {f.name: f for f in self.functions}

    @property
    def by_index(self) -> dict[int, FunctionSpec]:
        return {f.index: f for f in self.functions}

    @property
    def nexa_to_name(self) -> dict[str, str]:
        return {f.nexa_token: f.name for f in self.functions}

    @property
    def name_to_nexa(self) -> dict[str, str]:
        return {f.name: f.nexa_token for f in self.functions}

    @property
    def nexa_end(self) -> str:
        return self.raw_meta["nexa_end_token"]

    @property
    def nexa_ask(self) -> str:
        """Token đánh dấu lời gọi chỉ là ĐỀ XUẤT (câu mơ hồ) — runtime hỏi lại trước khi làm."""
        return self.raw_meta["nexa_ask_token"]

    @property
    def special_tokens(self) -> list[str]:
        """Toàn bộ token cần thêm vào tokenizer, theo thứ tự cố định.
        Token mới luôn nối vào CUỐI để id của token cũ không bị xê dịch."""
        return [f.nexa_token for f in self.functions] + [self.nexa_end, self.nexa_ask]

    @property
    def system_prompt(self) -> str:
        return self.raw_meta["system_prompt"]

    # ── render prompt (bọc ChatML vì student là Instruct) ─────────────────
    def render_call_prompt(self, query: str, luot_truoc: str | None = None) -> str:
        """Prompt A — lượt 1: câu hỏi người dùng → model sinh lời gọi hàm.

        `luot_truoc` (v6): lời gọi đã THỰC THI ở lượt trước (vd `<nexa_11>()`) -> thêm một dòng
        "Lượt trước: ..." trước câu; không có thì prompt Y HỆT v5 (chấm được trên test cũ)."""
        noi_dung = query.strip() if not luot_truoc else f"Lượt trước: {luot_truoc.strip()}\n{query.strip()}"
        return (
            self.raw_meta["call_prompt"]
            .replace("{system}", self.system_prompt)
            .replace("{query}", noi_dung)
        )

    def render_reply_prompt(self, query: str, call: str, result: Any) -> str:
        """Prompt B — lượt 2: câu hỏi + lời gọi + kết quả → model viết câu trả lời."""
        result_str = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        return (
            self.raw_meta["reply_prompt"]
            .replace("{system}", self.system_prompt)
            .replace("{query}", query.strip())
            .replace("{call}", call.strip())
            .replace("{result}", result_str)
        )

    def function_description_block(self) -> str:
        """Mô tả toàn bộ hàm — chỉ dùng lúc sinh data, KHÔNG dùng lúc infer."""
        return "\n\n".join(f.teacher_doc() for f in self.functions)

    def json_schema(self, name: str) -> dict[str, Any]:
        return self.by_name[name].json_schema()

    # ── xuất cho frontend ────────────────────────────────────────────────
    def scene_actions(self) -> dict[str, Any]:
        return {
            "cameraPresets": self.camera_presets,
            "hudWidgets": self.hud_widgets,
            "actions": {
                f.name: {
                    "nexa": f.nexa_token,
                    "camera": f.ui_action.camera,
                    "animation": f.ui_action.animation,
                    "hud": f.ui_action.hud,
                }
                for f in self.functions
            },
        }

    # ── kiểm tra toàn vẹn ────────────────────────────────────────────────
    def validate(self) -> None:
        indices = [f.index for f in self.functions]
        assert indices == list(range(len(indices))), f"index phải liên tục từ 0, đang là {indices}"
        names = [f.name for f in self.functions]
        assert len(names) == len(set(names)), "có tên hàm bị trùng"
        cams = set(self.camera_presets)
        huds = set(self.hud_widgets)
        for f in self.functions:
            assert f.ui_action.camera in cams, f"{f.name}: camera preset lạ {f.ui_action.camera!r}"
            widget = f.ui_action.hud.get("widget", "none")
            assert widget in huds, f"{f.name}: hud widget lạ {widget!r}"
        for key in ("call_prompt", "reply_prompt", "system_prompt", "nexa_end_token", "nexa_ask_token"):
            assert key in self.raw_meta, f"thiếu meta.{key} trong functions.yaml"
        assert not self.by_name["irrelevant_function"].implicit_examples, (
            "irrelevant_function không được có implicit_examples (không có gì để đề xuất)"
        )
        for f in self.functions:
            ten_tham_so = {p.name for p in f.params}
            assert not (f.nhay_cam and f.implicit_examples), f"{f.name}: hàm nhạy cảm không được có câu gợi ý"
            for q in f.an_toan:
                assert q.khi and set(q.khi) <= DIEU_KIEN_AN_TOAN, f"{f.name}: điều kiện an toàn lạ {q.khi!r}"
                assert q.xu_ly in XU_LY_AN_TOAN, f"{f.name}: cách xử lý an toàn lạ {q.xu_ly!r}"
                assert set(q.neu_tham_so) <= ten_tham_so, f"{f.name}: luật an toàn nhắc tham số lạ"
                if q.xu_ly == "gioi_han":
                    assert q.tham_so in ten_tham_so and q.toi_da is not None, (
                        f"{f.name}: luật gioi_han cần `tham_so` có thật và `toi_da`"
                    )


def load(path: Path = YAML_PATH) -> Catalog:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    functions = [
        FunctionSpec(
            index=raw["index"],
            name=raw["name"],
            category=raw["category"],
            description=raw["description"],
            params=[ParamSpec.from_raw(p) for p in (raw.get("params") or [])],
            returns=raw.get("returns", {}),
            ui_action=UiAction(
                camera=raw["ui_action"]["camera"],
                animation=raw["ui_action"].get("animation"),
                hud=raw["ui_action"].get("hud", {}),
            ),
            examples=list(raw.get("examples", [])),
            implicit_examples=list(raw.get("implicit_examples", [])),
            tu_dong_nghia=list(raw.get("tu_dong_nghia", [])),
            chi_doc=bool(raw.get("chi_doc", False)),
            nhay_cam=bool(raw.get("nhay_cam", False)),
            an_toan=[QuyTacAnToan.from_raw(q) for q in (raw.get("an_toan") or [])],
        )
        for raw in data["functions"]
    ]
    functions.sort(key=lambda f: f.index)
    catalog = Catalog(
        raw_meta=data["meta"],
        functions=functions,
        camera_presets=list(data["camera_presets"]),
        hud_widgets=list(data["hud_widgets"]),
    )
    catalog.validate()
    return catalog


CATALOG = load()


if __name__ == "__main__":
    import sys

    # Console Windows mặc định cp1252 — ép UTF-8 để in được tiếng Việt.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print(f"{len(CATALOG.functions)} hàm, {len(CATALOG.special_tokens)} token đặc biệt\n")

    print("map functional token:")
    for tok, name in CATALOG.nexa_to_name.items():
        print(f"  {tok:>11}  {name}")

    print("\n--- Prompt A (lượt 1) ---")
    print(CATALOG.render_call_prompt("Bật điều hòa 22 độ ghế lái"))
    print("--- Prompt B (lượt 2) ---")
    print(
        CATALOG.render_reply_prompt(
            "Bật điều hòa 22 độ ghế lái",
            '<nexa_6>(state="on", temperature=22, zone="driver")<nexa_end>',
            {"ac_on": True, "temperature": 22, "zone": "driver"},
        )
    )
