import json
import re
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from gen.records import query_from_call_prompt as q
from schema.call_format import parse_calls


def doc(p):
    return [json.loads(x) for x in open(p, encoding="utf-8")]


def calls(r):
    return [(c.name, dict(c.args)) for c in parse_calls(r["completion"].replace("<|im_end|>", ""))]


V5 = doc("data/processed/v5/v5_hoi_lai_hc_gop.jsonl")
CU = [r for r in doc("data/processed/final_v4_doi_chung/train_vi.jsonl") if r["meta"]["type"] == "ask"]

NGOAI = re.compile(r"bên ngoài|ngoài đường|ngoài kia|ngoài trời|hàng xóm|xe cộ|công trình|khoan")
ON = re.compile(r"ồn|inh|ầm|to quá|nhức")
MO = re.compile(r"mờ|đọng|hơi nước|hấp hơi|sương|nhòe|nhoè|đổ mồ hôi|bám hơi")
DEN_DA_BAT = re.compile(
    r"(đèn pha|đèn xe|đèn (của )?(mình|tao|tôi|em)|đèn nhà mình)[^.,]{0,20}(không sáng|yếu|mờ|tối|chưa đủ)"
    r"|không sáng bằng|đủ sáng không|có đủ sáng"
)

LUAT = {
    "A_on_ngoai_ma_am_luong": lambda c, s: c[0][0] == "set_volume" and NGOAI.search(s) and ON.search(s),
    "B_quat_level0": lambda c, s: any(n == "set_fan_speed" and a.get("level") == 0 for n, a in c),
    "C_say_kinh_khong_mo": lambda c, s: any(n == "set_defrost" for n, _ in c) and not MO.search(s),
    "D_den_da_bat_ma_yeu": lambda c, s: any(n == "set_headlights" for n, _ in c) and DEN_DA_BAT.search(s),
    "E_am_luong_khong_huong": lambda c, s: any(n == "set_volume" and not a for n, a in c),
}
for ten, ds in (("v5 mới (1.751)", V5), (f"hỏi lại train cũ ({len(CU)})", CU)):
    print(f"\n== {ten}")
    for k, f in LUAT.items():
        hit = [r for r in ds if f(calls(r), q(r["prompt"]).lower())]
        print(f"  {k:26} {len(hit):4d}")
        for r in hit[:3]:
            print(f"       {q(r['prompt'])[:80]} -> {[n for n, _ in calls(r)]}")
fan = [r for r in V5 if any(n == "set_fan_speed" for n, _ in calls(r))]
print("\nquạt gió v5 theo level:", {lv: sum(1 for r in fan for n, a in calls(r) if n == "set_fan_speed" and a.get("level") == lv) for lv in range(8)})
fan_cu = [r for r in CU if any(n == "set_fan_speed" for n, _ in calls(r))]
print("quạt gió train cũ theo level:", {lv: sum(1 for r in fan_cu for n, a in calls(r) if n == "set_fan_speed" and a.get("level") == lv) for lv in range(8)})
