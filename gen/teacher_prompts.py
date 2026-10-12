"""Prompt gửi cho TEACHER (Qwen2.5-72B-Instruct) + parser đọc kết quả — Phase 2.

Lưu ý thiết kế quan trọng: **teacher KHÔNG biết functional token** (`<nexa_i>` là
token mới chỉ student học lúc fine-tune, teacher vanilla chưa từng thấy). Vì vậy
teacher luôn được yêu cầu trả lời bằng **tên hàm dạng chữ + JSON tham số** (quen
thuộc, ít lỗi cú pháp hơn với teacher) — sau đó CODE (không phải teacher) tự
chuyển sang định dạng `<nexa_i>(...)<nexa_end>` bằng `schema.call_format`, vốn đã
test kỹ ở Phase 0.

Mọi hàm ở đây chỉ **build prompt (str)** hoặc **parse response (str) -> dữ liệu
Python** — không tự gọi model. Tách khỏi phần gọi vLLM (gen/modal_app.py, viết
sau) để test được ở local bằng dữ liệu giả lập, không cần GPU/Modal.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from schema.loader import CATALOG, FunctionSpec, ParamSpec

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class TeacherResponseError(ValueError):
    """Ném ra khi không parse được JSON hợp lệ từ output của teacher."""


def _extract_json(text: str) -> Any:
    """Bóc JSON ra khỏi output của teacher — teacher hay bọc trong ```json ... ```
    hoặc thêm vài câu giải thích trước/sau, nên không thể `json.loads` thẳng."""
    t = text.strip()
    m = _JSON_FENCE.search(t)
    candidate = m.group(1) if m else t
    start_candidates = [i for i in (candidate.find("{"), candidate.find("[")) if i != -1]
    if start_candidates:
        start = min(start_candidates)
        end = max(candidate.rfind("}"), candidate.rfind("]"))
        if end > start:
            candidate = candidate[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise TeacherResponseError(f"không parse được JSON từ output teacher: {exc}\n---\n{text}") from exc


# ── 1. Sinh câu hỏi tích cực (loại A), theo từng hàm ────────────────────────
def _yeu_cau_tu_vung(fn: FunctionSpec, n: int) -> str:
    """Ép teacher chia đều các cách gọi tên bộ phận, thay vì để nó tự do.

    Không có đoạn này thì teacher lặp lại đúng từ trong `examples`: đo trên bộ dữ liệu
    vòng 1, cả 267/267 câu của control_window đều dùng "cửa sổ" và 0 câu dùng "kính".
    Nói "hãy đa dạng cách diễn đạt" là chưa đủ — teacher hiểu thành đa dạng ở cấu trúc
    câu và mức lịch sự, chứ không đổi danh từ chỉ bộ phận.
    """
    if not fn.tu_dong_nghia:
        return ""
    ds = fn.tu_dong_nghia
    moi_tu = max(1, n // len(ds))
    liet_ke = ", ".join(f'"{t}"' for t in ds)
    return f"""
- QUAN TRỌNG — cách gọi tên: người Việt gọi thứ này bằng {len(ds)} cách: {liet_ke}.
  Hãy DÙNG ĐỀU tất cả các cách đó, mỗi cách khoảng {moi_tu} câu. Đừng chỉ dùng một từ.
  Đếm lại trước khi trả lời: mỗi cách gọi phải xuất hiện ít nhất vài câu."""


def build_query_gen_prompt(fn: FunctionSpec, n: int, lang: str = "vi") -> str:
    lang_name = "tiếng Việt" if lang == "vi" else "English"
    examples = "\n".join(f'- "{e}"' for e in fn.examples)
    tu_vung = _yeu_cau_tu_vung(fn, n)
    return f"""Bạn là người tạo dữ liệu huấn luyện cho một tác tử điều khiển xe ô tô bằng giọng nói.

Hàm cần tạo câu hỏi:
{fn.teacher_doc()}

Hãy viết {n} câu người dùng có thể nói bằng {lang_name} để yêu cầu thực hiện ĐÚNG hàm này.
Yêu cầu:
- Đa dạng cách diễn đạt: lịch sự, thân mật, ra lệnh trực tiếp, hỏi gián tiếp, có/không có từ đệm
- Khoảng một nửa số câu đủ thông tin để điền hết tham số; nửa còn lại thiếu tham số
  optional (sẽ dùng giá trị mặc định)
- Không lặp lại cấu trúc câu, không đánh số thứ tự trong câu
- Không nhắc tên hàm hay bất kỳ từ khoá kỹ thuật nào, chỉ dùng ngôn ngữ tự nhiên đời thường{tu_vung}

Câu ví dụ tham khảo (đừng lặp lại y hệt):
{examples}

Chỉ trả về đúng JSON, không thêm chữ nào khác:
{{"queries": ["câu 1", "câu 2", ...]}}"""


_MANG_QUERIES = re.compile(r'"queries"\s*:\s*\[')
_CHUOI_HOAN_CHINH = re.compile(r'"(?:[^"\\]|\\.)*"')


def _cuu_van_queries_bi_cat(text: str) -> list[str]:
    """Vớt các câu HOÀN CHỈNH từ JSON bị cắt giữa chừng (do chạm max_tokens).

    Khi model bị ngắt đột ngột, JSON thiếu dấu đóng `"`, `]`, `}` nên `json.loads`
    hỏng — nhưng hàng trăm câu phía trước vẫn nguyên vẹn. Không vớt thì vứt sạch
    cả lô dù đã trả tiền GPU sinh ra chúng (đã gặp thật: mỗi lô mất ~250 câu).

    Câu cuối cùng (đang viết dở, chưa có dấu nháy đóng) tự động bị loại vì regex
    chỉ khớp chuỗi có đủ cặp nháy.
    """
    m = _MANG_QUERIES.search(text)
    if not m:
        return []
    ket_qua: list[str] = []
    for tok in _CHUOI_HOAN_CHINH.findall(text[m.end() :]):
        try:
            s = json.loads(tok)  # json.loads lo phần giải escape (\" \\ \n ...)
        except json.JSONDecodeError:
            continue
        if isinstance(s, str) and s.strip():
            ket_qua.append(s.strip())
    return ket_qua


def parse_query_gen_response(text: str) -> list[str]:
    """Đọc danh sách câu hỏi teacher sinh ra.

    Nếu JSON hỏng (thường do output bị cắt vì chạm max_tokens) thì KHÔNG vứt cả
    lô — cố vớt các câu hoàn chỉnh phía trước đã.
    """
    try:
        data = _extract_json(text)
    except TeacherResponseError:
        cuu_duoc = _cuu_van_queries_bi_cat(text)
        if cuu_duoc:
            print(f"    [cảnh báo] JSON hỏng (nhiều khả năng bị cắt), vớt được {len(cuu_duoc)} câu")
            return cuu_duoc
        raise

    if not isinstance(data, dict) or not isinstance(data.get("queries"), list):
        cuu_duoc = _cuu_van_queries_bi_cat(text)
        if cuu_duoc:
            print(f"    [cảnh báo] thiếu khoá 'queries' hợp lệ, vớt được {len(cuu_duoc)} câu")
            return cuu_duoc
        raise TeacherResponseError(f"thiếu khoá 'queries' dạng list: {data!r}")

    return [str(q).strip() for q in data["queries"] if str(q).strip()]


# ── 2. Sinh câu hỏi ngoài phạm vi (loại negative) ───────────────────────────
def build_negative_query_prompt(n: int, lang: str = "vi") -> str:
    lang_name = "tiếng Việt" if lang == "vi" else "English"
    return f"""Bạn là người tạo dữ liệu huấn luyện cho một tác tử điều khiển xe ô tô bằng giọng nói.
Tác tử này CHỈ hỗ trợ các việc: kiểm tra trạng thái xe (xăng, pin, lốp, quãng đường),
điều hoà, khoá/mở cửa, cốp, cửa sổ, cửa sổ trời, đèn, âm lượng/nhạc, dẫn đường, thời tiết.

Hãy viết {n} câu bằng {lang_name} mà người dùng CÓ THỂ nói với 1 trợ lý giọng nói trên xe,
nhưng KHÔNG liên quan tới bất kỳ chức năng nào ở trên. Trộn đều các loại:
- Tán gẫu / hỏi thăm ("hôm nay bạn thế nào")
- Hỏi kiến thức chung không liên quan xe ("thủ đô nước Pháp là gì")
- Yêu cầu xe KHÔNG hỗ trợ (tự lái hoàn toàn, đổi màu xe, sửa xe...)
- Câu mơ hồ không đủ ý nghĩa để hiểu là yêu cầu gì

Chỉ trả về đúng JSON, không thêm chữ nào khác:
{{"queries": ["câu 1", "câu 2", ...]}}"""


# ── 2b. Câu từ chối KHÓ: ra lệnh, đúng chủ đề xe, nhưng xe không hỗ trợ ─────
#
# Các nhóm tính năng dưới đây là thứ xe thật CÓ mà 24 hàm của mình KHÔNG có. Suy ra từ
# `VehiclePropertyIds` của Android Automotive (nguồn đã ghi ở PLAN.md B3) trừ đi bộ hàm
# hiện tại — KHÔNG lấy từ tập test, vì sửa dữ liệu theo câu test thì tập test hết còn là
# dữ liệu chưa từng thấy.
TINH_NANG_KHONG_HO_TRO = [
    "sưởi/làm mát ghế (HVAC_SEAT_TEMPERATURE)",
    "sưởi vô lăng (HVAC_STEERING_WHEEL_HEAT)",
    "lấy gió trong / lấy gió ngoài (HVAC_RECIRC_ON)",
    "hướng gió điều hoà, cửa gió (HVAC_FAN_DIRECTION)",
    "cần gạt nước, phun nước rửa kính (WINDSHIELD_WIPERS_*)",
    "gương chiếu hậu: gập, chỉnh, sưởi gương (MIRROR_*)",
    "chỉnh ghế: ngả lưng, tiến lùi, nhớ vị trí (SEAT_*)",
    "chế độ lái: thể thao, tiết kiệm, off-road, địa hình",
    "ga tự động / giữ tốc độ (cruise control)",
    "camera lùi, camera 360, cảm biến đỗ xe",
    "phanh tay điện tử, giữ phanh tự động",
    "đèn nội thất, đèn viền, đèn trần",
    "rèm che nắng, rèm cửa sổ trời",
    "cốp trước (frunk), nắp capo",
    "kết nối Bluetooth, ghép điện thoại, gọi điện",
    "chuyển bài hát tiếp theo / bài trước, tua nhạc",
    "huỷ lộ trình đang dẫn, hỏi giờ tới nơi",
    "khởi động máy từ xa, tắt máy",
]


def build_hard_negative_prompt(n: int, lang: str = "vi", nhom: str | None = None) -> str:
    """Sinh câu RA LỆNH mà tác tử phải TỪ CHỐI — loại negative khó nhất.

    Vì sao cần riêng: đo trên dữ liệu vòng 1, mẫu từ chối chỉ có **2,1%** là câu mệnh
    lệnh, còn mẫu gọi hàm có 37,1%. Tức trong train, hễ gặp câu mệnh lệnh thì 94,7% số
    lần đáp án là gọi hàm — model học tắt theo DẠNG CÂU thay vì theo nghĩa. Đến lúc gặp
    "Bật sưởi ghế lái lên" (xe không có tính năng đó) là nó gọi bừa `set_ac`.

    Thêm mẫu từ chối dạng câu hỏi lịch sự KHÔNG chữa được: nhóm đó model vốn đã làm tốt
    (87–100%). Phải thêm đúng loại câu mệnh lệnh này để phá luật tắt kia.
    """
    lang_name = "tiếng Việt" if lang == "vi" else "English"
    ds = [nhom] if nhom else TINH_NANG_KHONG_HO_TRO
    liet_ke = "\n".join(f"- {t}" for t in ds)
    return f"""Bạn là người tạo dữ liệu huấn luyện cho một tác tử điều khiển xe ô tô bằng giọng nói.

Tác tử này CHỈ làm được: kiểm tra trạng thái xe (xăng, pin, lốp, số km, tầm hoạt động),
bật/tắt điều hoà và đặt nhiệt độ, tốc độ quạt gió, sấy kính, khoá/mở khoá cửa, mở/đóng cốp,
mở nắp sạc, mở nắp bình xăng, hạ/nâng kính cửa sổ, cửa sổ trời, đèn pha, đèn khẩn cấp,
âm lượng, phát nhạc, dẫn đường tới địa chỉ, tìm địa điểm gần đây, xem thời tiết.

Hãy viết {n} câu bằng {lang_name} mà người ngồi trong xe RA LỆNH cho tác tử làm những việc
sau — đây đều là tính năng xe thật có nhưng tác tử này KHÔNG làm được:
{liet_ke}

Yêu cầu bắt buộc:
- Phải là câu RA LỆNH, mở đầu bằng động từ sai khiến (bật, mở, tắt, đóng, hạ, kéo, gập,
  chuyển, chỉnh, tăng, giảm, cài, huỷ, bấm...) — ĐÂY LÀ ĐIỂM QUAN TRỌNG NHẤT
- Nghe phải y như một câu lệnh thật trong xe: cộc lốc, tự nhiên, có thể kèm "giúp tôi",
  "hộ cái", "đi", "lên", "xuống"
- TUYỆT ĐỐI KHÔNG viết câu rơi vào danh sách tác tử LÀM ĐƯỢC ở trên. Ví dụ "bật điều hoà",
  "hạ kính", "mở cốp" là SAI vì tác tử làm được — đừng viết những câu như vậy.
- Không viết câu hỏi lịch sự kiểu "bạn có thể giúp tôi..."; chỉ viết câu ra lệnh
- Không lặp lại cấu trúc câu, không đánh số thứ tự trong câu

Chỉ trả về đúng JSON, không thêm chữ nào khác:
{{"queries": ["câu 1", "câu 2", ...]}}"""


# ── 2c. Câu từ chối KỂ HOÀN CẢNH (vòng 4 — PLAN.md E0b "Hướng 2") ──────────
#
# Vì sao: câu từ chối khó ở 2b là mệnh lệnh ngắn một khuôn. Đo trên tập dev người viết: câu
# "kể hoàn cảnh rồi mới nhờ" việc xe không làm được -> model chỉ từ chối đúng 20–33%, vì phần
# hoàn cảnh chứa từ khoá của hàm thật (mưa/kính -> sấy kính) kéo model gọi bừa.
#
# Cách làm theo các paper làm đa dạng dữ liệu — ĐỔI ĐẦU VÀO mỗi lần sinh thay vì xin "viết nhiều":
#   - vai người nói (Persona Hub, Ge et al. 2024)
#   - câu có hoàn cảnh, vòng vo (Evol-Instruct, Xu et al. 2023)
#   - mỗi lô một bộ (việc, vai, bẫy) khác nhau, không câu ví dụ nào để teacher khỏi chép khuôn.
# Nguồn việc cần từ chối độc lập với tập dev/test: tính năng xe từ VehiclePropertyIds (2b) và
# việc ngoài xe theo 18 lĩnh vực của MASSIVE (FitzGerald et al. 2022).
VAI_NGUOI_NOI = [
    "tài xế taxi công nghệ chạy cả ngày, nói cộc, hay dùng từ đệm",
    "người mẹ đang chở con nhỏ đi học, vội và hơi bực",
    "người lớn tuổi mới dùng xe có trợ lý giọng nói, nói chậm, lễ phép, dài dòng",
    "sinh viên mới có bằng lái, hơi lúng túng, hay giải thích",
    "nhân viên văn phòng kẹt xe giờ tan tầm, mệt mỏi",
    "tài xế chạy đường dài trên cao tốc ban đêm",
    "người đang chở cả gia đình đi du lịch xa",
    "người đang trễ giờ họp, nói nhanh, gấp gáp",
    "người miền Nam nói giọng tự nhiên, dùng từ địa phương",
    "người miền Trung nói giọng tự nhiên, dùng từ địa phương",
    "người vừa đi nhậu về đang ngồi ghế phụ, nói lè nhè nhưng vẫn rõ ý",
    "người yêu xe, rành công nghệ, gọi đúng tên từng tính năng",
]
BAY_TU_KHOA = [
    "nóng bức hoặc lạnh buốt trong xe",
    "trời mưa, kính bị mờ hoặc đọng nước",
    "trời tối, đường thiếu ánh sáng",
    "tiếng ồn trong hoặc ngoài xe",
    "cửa xe, an ninh, sợ mất đồ",
    "sắp hết xăng hoặc hết pin, đường còn xa",
    "đường lạ, tắc đường, sợ lạc",
    "đang nghe nhạc hoặc nghe đài",
    "gió, bụi, mùi trong xe",
    "đang tìm chỗ đỗ hoặc vừa đỗ xe",
]
VIEC_NGOAI_XE = [
    "điều khiển đồ trong NHÀ: đèn phòng, điều hoà phòng, khoá cửa nhà, rèm, robot hút bụi",
    "nhắn tin, gửi email, đăng mạng xã hội",
    "lịch hẹn, báo thức, nhắc việc, danh sách việc cần làm",
    "đặt đồ ăn, mua sắm online, đặt vé",
    "ngân hàng, chuyển tiền, thanh toán hoá đơn",
    "hỏi kiến thức, tin tức, tỉ giá, kết quả bóng đá",
    "nấu ăn, công thức, sức khoẻ",
]


def build_contextual_negative_prompt(n: int, viec: str, vai: str, bay: str, lang: str = "vi") -> str:
    """Sinh `n` câu KỂ HOÀN CẢNH rồi NHỜ một việc tác tử KHÔNG làm được (việc = `viec`)."""
    lang_name = "tiếng Việt" if lang == "vi" else "English"
    return f"""Bạn là người tạo dữ liệu huấn luyện cho một tác tử điều khiển xe ô tô bằng giọng nói.

Tác tử này CHỈ làm được: kiểm tra trạng thái xe (xăng, pin, lốp, số km, tầm hoạt động),
bật/tắt điều hoà và đặt nhiệt độ, tốc độ quạt gió, sấy kính, khoá/mở khoá cửa XE, mở/đóng cốp sau,
mở nắp sạc, mở nắp bình xăng, hạ/nâng kính cửa sổ, cửa sổ trời, đèn pha, đèn khẩn cấp,
âm lượng, phát nhạc, dẫn đường tới địa chỉ, tìm địa điểm gần đây, xem thời tiết.

Hãy viết {n} câu bằng {lang_name}, mỗi câu là lời của MỘT người đang ngồi trong xe nói với tác tử.
Người nói: {vai}.

Mỗi câu gồm HAI phần, đúng thứ tự:
1. Kể hoàn cảnh hoặc lý do dẫn tới lời nhờ. Nếu ghép được tự nhiên, cho hoàn cảnh nhắc tới
   chuyện: {bay}. Không ghép được tự nhiên thì dùng hoàn cảnh khác cho hợp lý.
2. Rồi NHỜ RÕ RÀNG một việc thuộc nhóm: {viec}.
   Tác tử KHÔNG làm được việc này — đó chính là điều câu cần kiểm tra.

Yêu cầu bắt buộc:
- Hoàn cảnh phải HỢP LÝ dẫn tới lời nhờ: người nghe thấy ngay vì sao người nói muốn việc đó.
  Không ghép gượng hai chuyện chẳng liên quan gì nhau.
- Lời nhờ ở phần 2 phải rõ ràng và cụ thể. Chỉ than thở mà không nhờ gì là SAI.
- Phần 1 CHỈ KỂ, không chứa lời nhờ hay câu ra lệnh nào khác. Mỗi câu nhờ ĐÚNG MỘT việc.
- Viết đúng chính tả tiếng Việt; không chèn từ tiếng Anh, trừ tên gọi quen thuộc như Bluetooth, camera.
- Lời nhờ TUYỆT ĐỐI không được là việc tác tử làm được ở danh sách trên (kể cả khi phần 1 nhắc
  tới chủ đề đó). Phần 1 được nhắc tới nóng lạnh, kính, đèn, tiếng ồn… nhưng phần 2 thì không.
- Nói đúng giọng và cách xưng hô của người nói ở trên; tự nhiên như lời nói, có thể lủng củng,
  có từ đệm.
- {n} câu phải là {n} tình huống KHÁC HẲN nhau — khác hoàn cảnh, khác cách mở đầu, khác cách nhờ
  (ra lệnh thẳng, hỏi được không, nhờ khéo, nói trống không). Không viết lại cùng một câu đổi vài
  chữ. Độ dài 10–35 chữ, câu ngắn câu dài xen kẽ.
- Không đánh số, không lặp khuôn câu.

Chỉ trả về đúng JSON, không thêm chữ nào khác:
{{"queries": ["câu 1", "câu 2", ...]}}"""


def build_contextual_positive_prompt(n: int, fn: FunctionSpec, vai: str, bay: str, lang: str = "vi") -> str:
    """Sinh `n` câu KỂ HOÀN CẢNH rồi NHỜ RÕ một việc tác tử LÀM ĐƯỢC (hàm `fn`) — cặp đối chứng của
    `build_contextual_negative_prompt` (PLAN.md mục P, kết quả vòng 4).

    Vì sao cần: vòng 4 thêm 1.651 câu kể hoàn cảnh mà TOÀN là từ chối -> model học tắt "câu kể hoàn cảnh
    ⟹ từ chối" (dev: câu làm được cùng kiểu bị từ chối 40–80%). Cùng khung (vai, bẫy, hoàn cảnh hợp lý) ở
    CẢ HAI phía nhãn thì hình thức câu không còn đoán được nhãn — model phải đọc nghĩa lời nhờ.
    """
    lang_name = "tiếng Việt" if lang == "vi" else "English"
    cach_goi = f" (người ta hay gọi: {', '.join(fn.tu_dong_nghia)})" if fn.tu_dong_nghia else ""
    return f"""Bạn là người tạo dữ liệu huấn luyện cho một tác tử điều khiển xe ô tô bằng giọng nói.

Việc tác tử LÀM ĐƯỢC cần nhắm tới: {fn.description.strip()}{cach_goi}

Hãy viết {n} câu bằng {lang_name}, mỗi câu là lời của MỘT người đang ngồi trong xe nói với tác tử.
Người nói: {vai}.

Mỗi câu gồm HAI phần, đúng thứ tự:
1. Kể hoàn cảnh hoặc lý do dẫn tới lời nhờ. Nếu ghép được tự nhiên, cho hoàn cảnh nhắc tới
   chuyện: {bay}. Không ghép được tự nhiên thì dùng hoàn cảnh khác cho hợp lý.
2. Rồi NHỜ RÕ RÀNG tác tử làm đúng việc ở trên (có thể kèm chi tiết như mức nhiệt độ, cửa nào,
   đi đâu… nếu hợp với việc đó).

Yêu cầu bắt buộc:
- Hoàn cảnh phải HỢP LÝ dẫn tới lời nhờ: người nghe thấy ngay vì sao người nói muốn việc đó.
- Phải có LỜI NHỜ RÕ RÀNG ở phần 2 (ra lệnh thẳng, hỏi được không, nhờ khéo đều được). Chỉ than thở,
  chỉ kể khó chịu mà không nhờ gì là SAI.
- Phần 1 CHỈ KỂ, không chứa lời nhờ nào khác. Mỗi câu nhờ ĐÚNG MỘT việc — việc ở trên.
- Nói đúng giọng và cách xưng hô của người nói ở trên; tự nhiên như lời nói, có thể lủng củng, có từ đệm.
- Viết đúng chính tả tiếng Việt; không chèn từ tiếng Anh, trừ tên gọi quen thuộc như Bluetooth, camera.
- {n} câu phải là {n} tình huống KHÁC HẲN nhau — khác hoàn cảnh, khác cách mở đầu, khác cách nhờ. Không
  viết lại cùng một câu đổi vài chữ. Độ dài 10–35 chữ, câu ngắn câu dài xen kẽ.
- Không đánh số, không lặp khuôn câu.

Chỉ trả về đúng JSON, không thêm chữ nào khác:
{{"queries": ["câu 1", "câu 2", ...]}}"""


# Bẫy HỢP với từng hàm hỏi lại. Chạy thử 1 (PLAN.md mục U) bốc bẫy ngẫu nhiên -> câu lạc đề: hàm
# "đóng cốp" + bẫy "trời tối" -> "Trời tối quá… cẩn thận khi lấy đồ ở cốp" -> người gán nhãn hiểu
# thành bật đèn; 39/107 câu lệch hàm đích. Với câu hỏi lại, hoàn cảnh CHÍNH LÀ manh mối của nhu cầu,
# nên phải hợp với hàm (khác câu từ chối, nơi bẫy được cố ý đặt LỆCH để làm khó).
#
# Chạy thử 2: bẫy MÂU THUẪN với `HUONG_DAN_NGAM` của chính hàm thì câu bị kéo sang hàm khác — cửa sổ
# trời ("KHÔNG than nóng lạnh") lại được bẫy "nóng bức" -> 0/10 câu khớp (gán điều hoà); quạt gió + bẫy
# "nóng lạnh" -> gán điều hoà; tìm quanh đây + bẫy "sợ lạc" -> gán dẫn đường. Bỏ các bẫy mâu thuẫn,
# thêm vài bẫy riêng đúng manh mối của hàm.
BAY_THEO_HAM: dict[str, list[str]] = {
    "set_ac": [BAY_TU_KHOA[0], BAY_TU_KHOA[8]],
    "set_fan_speed": [BAY_TU_KHOA[3], "luồng gió điều hoà thổi quá mạnh hoặc quá yếu"],
    "set_defrost": [BAY_TU_KHOA[1], BAY_TU_KHOA[0]],
    "lock_doors": [BAY_TU_KHOA[4], BAY_TU_KHOA[9]],
    "close_trunk": [BAY_TU_KHOA[1], BAY_TU_KHOA[4], BAY_TU_KHOA[9]],
    "control_window": [BAY_TU_KHOA[3], BAY_TU_KHOA[8], BAY_TU_KHOA[1]],
    "operate_sunroof": [
        "bầu trời, nắng, trăng sao nhìn qua phía trên đầu",
        "mưa, lá hoặc bụi lọt vào từ khe phía trên đầu",
        "muốn thoáng phía trên đầu khi ngồi lâu trong xe",
    ],
    "set_headlights": [BAY_TU_KHOA[2], BAY_TU_KHOA[1]],
    "set_hazard_lights": [BAY_TU_KHOA[5], BAY_TU_KHOA[2], BAY_TU_KHOA[1]],
    "set_volume": [BAY_TU_KHOA[3], BAY_TU_KHOA[7]],
    "play_media": [BAY_TU_KHOA[7], BAY_TU_KHOA[6]],
    "navigate_to": [BAY_TU_KHOA[6], BAY_TU_KHOA[5]],
    "find_nearby": [BAY_TU_KHOA[5], BAY_TU_KHOA[9], "đói, khát, buồn ngủ hoặc cần nghỉ chân giữa đường"],
}


def build_contextual_ask_prompt(n: int, fn: FunctionSpec, vai: str, bay: str, lang: str = "vi") -> str:
    """Sinh `n` câu KỂ HOÀN CẢNH mà KHÔNG nhờ gì — tác tử nên HỎI LẠI có muốn dùng hàm `fn` không.

    Cặp đối chứng THỨ BA (PLAN.md mục U). Vòng 4 có câu kể hoàn cảnh ở phía từ chối
    (`build_contextual_negative_prompt`) và phía gọi hàm (`build_contextual_positive_prompt`) nhưng
    KHÔNG có ở phía hỏi lại: train có 3.254 câu từ chối dài vs 102 câu hỏi lại dài -> trên bộ mơ hồ
    130 câu, student từ chối nhầm 13–33/81 câu dài. Cùng khung (vai, bẫy) ở cả BA phía nhãn thì độ dài
    / hình thức câu không còn đoán được nhãn: nhờ việc được -> gọi; nhờ việc không được -> từ chối;
    chỉ kể mà một chức năng giúp được -> hỏi lại.

    Không đưa câu ví dụ (như hai prompt kể hoàn cảnh kia) để teacher khỏi chép khuôn.
    """
    lang_name = "tiếng Việt" if lang == "vi" else "English"
    huong_dan = HUONG_DAN_NGAM.get(fn.name)
    khoi_huong_dan = f"\nĐÚNG TÌNH HUỐNG — quan trọng:\n{huong_dan}\n" if huong_dan else ""
    return f"""Bạn là người tạo dữ liệu huấn luyện cho một tác tử điều khiển xe ô tô bằng giọng nói.

Chức năng của xe cần nhắm tới: {fn.description.strip()}
{khoi_huong_dan}
Hãy viết {n} câu bằng {lang_name}, mỗi câu là lời của MỘT người đang ngồi trong xe.
Người nói: {vai}.

Mỗi câu KỂ một hoàn cảnh, cảm giác hoặc vấn đề đang gặp — KHÔNG nhờ, KHÔNG ra lệnh gì cả — nhưng một
tác tử tinh ý nghe xong sẽ nghĩ tới việc HỎI LẠI xem họ có muốn dùng chức năng ở trên không.
Nếu ghép được tự nhiên, cho hoàn cảnh nhắc tới chuyện: {bay}. Không ghép được tự nhiên thì dùng hoàn
cảnh khác cho hợp lý với chức năng trên.

Yêu cầu bắt buộc:
- TUYỆT ĐỐI không có lời nhờ hay câu ra lệnh: không dùng "bật", "tắt", "mở", "đóng", "khoá", "chỉnh",
  "tăng", "giảm", "hạ", "nâng", "tìm", "phát", "dẫn đường", "giúp", "nhờ", "cho tôi"; không hỏi "được
  không". Chỉ kể, than, cảm thán, nói bâng quơ hoặc nói với người ngồi cạnh.
- Không nhắc thẳng tên chức năng. Nhưng câu phải đủ manh mối để người nghe đoán ra nhu cầu đó.
- Hoàn cảnh phải HỢP LÝ và thường gặp ngoài đời ở Việt Nam (không tuyết rơi, không tình huống hiếm,
  không ghép gượng).
- Mỗi câu chỉ nói về MỘT chuyện — chuyện dẫn tới đúng nhu cầu ở trên; không kể thêm chuyện thứ hai.
- Nói đúng giọng và cách xưng hô của người nói ở trên; tự nhiên như lời nói, có thể có từ đệm.
- Viết hoàn toàn bằng tiếng Việt có dấu; không chèn bất kỳ từ tiếng Anh nào (kể cả "mood", "relax",
  "meeting"), trừ tên gọi quen thuộc như Bluetooth.
- {n} câu phải là {n} tình huống KHÁC HẲN nhau — khác hoàn cảnh, khác cách mở đầu. Độ dài 12–35 chữ.
- Không đánh số, không lặp khuôn câu.

Chỉ trả về đúng JSON, không thêm chữ nào khác:
{{"queries": ["câu 1", "câu 2", ...]}}"""


# ── 3. Trích tham số từ câu hỏi (biết trước đúng hàm) ───────────────────────
def build_call_gen_prompt(query: str, fn: FunctionSpec) -> str:
    return f"""Cho câu hỏi của người dùng và mô tả hàm dưới đây, hãy trích ra đúng tham số để gọi hàm.

Câu hỏi: "{query}"

{fn.teacher_doc()}

Chỉ điền tham số suy ra được rõ ràng từ câu hỏi; bỏ qua tham số optional không được nhắc tới
(đừng tự bịa giá trị mặc định). Không thêm tham số ngoài danh sách trên.

Chỉ trả về đúng JSON, không thêm chữ nào khác:
{{"arguments": {{"tên_tham_số": giá_trị, ...}}}}"""


def _coerce_to_declared_type(value: Any, p: ParamSpec) -> Any:
    """Ép giá trị teacher trả về đúng kiểu đã khai trong schema.

    LLM rất hay trả số dưới dạng chuỗi (`"22"` thay vì `22`) hoặc số thực cho
    tham số int (`22.0`). Nếu để nguyên, cùng 1 tham số sẽ có nhiều định dạng
    khác nhau trong dataset -> model học bị rối, và validator lúc chạy thật sẽ
    từ chối. Ép kiểu ngay tại đây để mọi nhãn sinh ra đều nhất quán.
    """
    if p.type == "string":
        if isinstance(value, bool):  # tránh True -> "True"
            raise TeacherResponseError(f"{p.name}: mong đợi chuỗi, nhận bool {value!r}")
        return str(value)

    if p.type == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        raise TeacherResponseError(f"{p.name}: mong đợi bool, nhận {value!r}")

    # int / number — bool là subclass của int trong Python nên phải chặn riêng
    if isinstance(value, bool):
        raise TeacherResponseError(f"{p.name}: mong đợi số, nhận bool {value!r}")

    if p.type == "int":
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(float(value.strip()))  # "22" và "22.0" đều về 22
            except ValueError as exc:
                raise TeacherResponseError(f"{p.name}: không đổi được {value!r} sang int") from exc
        raise TeacherResponseError(f"{p.name}: mong đợi int, nhận {value!r}")

    if p.type == "number":
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError as exc:
                raise TeacherResponseError(f"{p.name}: không đổi được {value!r} sang number") from exc
        raise TeacherResponseError(f"{p.name}: mong đợi number, nhận {value!r}")

    raise TeacherResponseError(f"{p.name}: kiểu khai báo lạ {p.type!r}")


def _check_rang_buoc(value: Any, p: ParamSpec) -> None:
    """Kiểm tra giá trị có thoả enum / min / max khai trong schema không.

    Ném lỗi (loại CẢ MẪU) thay vì bỏ riêng tham số sai — vì bỏ tham số sẽ tạo nhãn
    lệch nghĩa. Ví dụ thật đã gặp: câu hỏi "thời tiết hôm qua" -> teacher trả
    day=-1 (ngoài khoảng 0..6); nếu chỉ bỏ `day` thì nhãn thành get_weather() =
    thời tiết HÔM NAY, không khớp câu hỏi -> dạy model sai.
    """
    if p.enum is not None and value not in p.enum:
        raise TeacherResponseError(f"{p.name}={value!r} không thuộc enum {p.enum}")
    if p.minimum is not None and isinstance(value, (int, float)) and value < p.minimum:
        raise TeacherResponseError(f"{p.name}={value} nhỏ hơn min {p.minimum}")
    if p.maximum is not None and isinstance(value, (int, float)) and value > p.maximum:
        raise TeacherResponseError(f"{p.name}={value} lớn hơn max {p.maximum}")


def parse_call_gen_response(text: str, fn: FunctionSpec) -> dict[str, Any]:
    """Đọc JSON tham số teacher trả về → lọc tham số lạ → ép đúng kiểu → kiểm tra
    ràng buộc enum/min/max → kiểm tra đủ tham số bắt buộc.

    Ném `TeacherResponseError` nếu có bất kỳ vi phạm nào — mẫu đó bị pipeline loại,
    tốt hơn là âm thầm đưa nhãn sai vào dataset (validator lúc chạy thật sẽ từ chối
    đúng những giá trị đó, tức model học ra lời gọi không dùng được).
    """
    data = _extract_json(text)
    if not isinstance(data, dict) or not isinstance(data.get("arguments"), dict):
        raise TeacherResponseError(f"thiếu khoá 'arguments' dạng dict: {data!r}")
    return _kiem_tra_tham_so(data["arguments"], fn)


def _kiem_tra_tham_so(raw_args: dict[str, Any], fn: FunctionSpec) -> dict[str, Any]:
    """Lọc tham số lạ/null → ép kiểu → check enum/min/max → check đủ required.
    Dùng chung cho mọi chỗ teacher trả tham số (trích tham số, đề xuất hỏi lại)."""
    by_name = {p.name: p for p in fn.params}
    out: dict[str, Any] = {}
    for k, v in raw_args.items():
        if k not in by_name or v is None:  # bỏ tham số lạ teacher tự bịa, bỏ null
            continue
        p = by_name[k]
        gia_tri = _coerce_to_declared_type(v, p)
        _check_rang_buoc(gia_tri, p)
        out[k] = gia_tri

    thieu = [p.name for p in fn.params if p.required and p.name not in out]
    if thieu:
        raise TeacherResponseError(f"thiếu tham số bắt buộc: {thieu}")
    return out


# ── 4. Sinh câu trả lời tự nhiên (loại B) ───────────────────────────────────
def build_reply_gen_prompt(query: str, call_str: str, result: dict[str, Any], lang: str = "vi") -> str:
    """Dùng ĐÚNG khung Prompt B đã định (schema/functions.yaml meta.reply_prompt) —
    vì teacher cũng là Qwen-Instruct nên hiểu thẳng ChatML, không cần hướng dẫn thêm.
    Chèn 1 dòng chỉ dẫn ngôn ngữ + không bịa số ngay trước khi model bắt đầu trả lời."""
    base = CATALOG.render_reply_prompt(query, call_str, result)
    lang_name = "tiếng Việt" if lang == "vi" else "tiếng Anh"
    head, _, _ = base.rpartition("<|im_start|>assistant")
    instruction = (
        f"(Trả lời bằng {lang_name}, 1-2 câu, thân thiện, CHỈ dùng đúng số liệu có trong kết quả ở trên, "
        "không bịa thêm.)\n"
    )
    return head + instruction + "<|im_start|>assistant"


def build_refusal_reply_prompt(query: str, lang: str = "vi") -> str:
    """Prompt RIÊNG cho lượt 2 của câu ngoài phạm vi.

    LỖI ĐÃ GẶP (652/1128 mẫu): dùng chung prompt reply thường thì teacher chỉ thấy kết
    quả `{"handled": true}` nên tưởng đã xử lý xong và TRẢ LỜI THẲNG câu hỏi — "máy ảnh
    nào tốt nhất?" được tư vấn máy ảnh thật. Lượt 1 từ chối đúng nhưng lượt 2 lại trả
    lời, mà người dùng chỉ nghe lượt 2 -> toàn bộ việc dạy từ chối thành vô nghĩa.
    """
    lang_name = "tiếng Việt" if lang == "vi" else "tiếng Anh"
    return f"""Bạn là trợ lý giọng nói trên xe ô tô. Bạn CHỈ làm được: điều khiển và tra cứu
trạng thái xe (xăng, pin, lốp, quãng đường), điều hoà, khoá/mở cửa, cốp, nắp sạc, nắp xăng,
cửa sổ, cửa sổ trời, đèn, âm lượng, phát nhạc, dẫn đường, tìm địa điểm gần, xem thời tiết.

Người dùng vừa nói: "{query}"

Câu này NẰM NGOÀI phạm vi của bạn. Hãy viết câu đáp bằng {lang_name}, 1-2 câu:
- Nói ngắn gọn rằng bạn không hỗ trợ việc này
- Có thể nhắc bạn giúp được gì trên xe
- TUYỆT ĐỐI KHÔNG trả lời nội dung câu hỏi, không tư vấn, không hướng dẫn, dù chỉ một phần
- Giọng thân thiện, tự nhiên như người nói; đa dạng cách diễn đạt, đừng lặp một khuôn câu

Chỉ viết câu đáp, không thêm gì khác."""


def parse_reply_gen_response(text: str) -> str:
    reply = text.split("<|im_end|>", 1)[0].strip()
    if not reply:
        raise TeacherResponseError("teacher trả lời rỗng")
    return reply


# ── 5. Verify — teacher tự chấm nhị phân ────────────────────────────────────
@dataclass
class VerifyResult:
    valid: bool
    reason: str


def build_verify_prompt(
    query: str,
    fn: FunctionSpec,
    args: dict[str, Any],
    reply: str | None = None,
    result: dict[str, Any] | None = None,
) -> str:
    """Prompt chấm cho mẫu hàm thường.

    QUAN TRỌNG: đưa lời gọi ở dạng **tên hàm đọc được** (`get_fuel_level()`), KHÔNG
    phải `<nexa_0>()` — teacher chưa từng thấy functional token nên nhìn token rỗng
    sẽ tưởng là vô nghĩa và loại sạch mẫu của các hàm không tham số (đã gặp thật:
    hàm 0 tham số bị loại 89-100%, hàm có tham số chỉ bị loại 0-12%).
    """
    from schema.call_format import render_readable

    call_doc = render_readable(fn.name, args)
    reply_block = (
        f'\nCâu trả lời đã sinh: "{reply}"\nKết quả hàm trả về: {json.dumps(result, ensure_ascii=False)}'
        if reply is not None
        else ""
    )
    reply_check = (
        "\n- Câu trả lời có tự nhiên và dùng ĐÚNG số liệu trong kết quả hàm không (không bịa số)?"
        if reply is not None
        else ""
    )
    return f"""Bạn là người kiểm duyệt dữ liệu huấn luyện cho tác tử điều khiển xe ô tô.

Câu hỏi người dùng: "{query}"

Hàm được chọn: {call_doc}
Mô tả hàm này: {fn.description.strip()}{reply_block}

Tham số của hàm này:
{_liet_ke_tham_so(fn)}

Kiểm tra:
- Hàm được chọn có đúng là hàm cần gọi cho câu hỏi này không?
- Tham số đã đúng chưa — không sai giá trị, không bịa thêm chi tiết mà câu không nhắc tới?
  LƯU Ý QUAN TRỌNG: chỉ tham số BẮT BUỘC mới cần có mặt. Tham số TUỲ CHỌN mà câu hỏi
  không nhắc tới thì **bỏ trống là ĐÚNG** — hệ thống sẽ dùng giá trị mặc định; đừng coi
  đó là thiếu sót. Hàm không có tham số nào thì gọi rỗng `{fn.name}()` cũng là ĐÚNG.{reply_check}

Chỉ trả về đúng JSON, không thêm chữ nào khác:
{{"valid": true hoặc false, "reason": "lý do ngắn gọn, tiếng Việt"}}"""


def _liet_ke_tham_so(fn: FunctionSpec) -> str:
    """Liệt kê rõ tham số nào BẮT BUỘC, tham số nào TUỲ CHỌN — cho người/model chấm.

    Cần vì giám khảo khác họ (Llama) chấm nghiêm hơn Qwen: đã gặp thật, nó chê
    `get_odometer()` là "thiếu tham số" trong khi `unit` vốn tuỳ chọn.
    """
    if not fn.params:
        return "- (hàm này không có tham số nào)"
    dong = []
    for p in fn.params:
        loai = "BẮT BUỘC" if p.required else "tuỳ chọn"
        them = f", mặc định {p.default!r}" if (not p.required and p.default is not None) else ""
        dong.append(f"- {p.name} ({loai}{them})")
    return "\n".join(dong)


# Ví dụ mẫu trong prompt chấm từ chối: (câu, giải thích, valid). Mỗi cặp cùng chủ đề, một
# làm được một không — dạy giám khảo rằng cùng chủ đề chưa đủ. Để thành hằng số riêng để
# test kiểm được không câu nào trùng tập test (đã có lần lỡ dùng đúng câu "Mở cốp sau ra").
VI_DU_GIAM_KHAO_TU_CHOI: list[tuple[str, str, bool]] = [
    ("Bật điều hoà lên", 'set_ac(state="on") làm được', False),
    ("Bật chế độ lấy gió trong", "set_ac không có tham số nào về lấy gió", True),
    ("Mở cốp sau cho tôi để đồ", "open_trunk() làm được", False),
    ("Mở cốp trước ra", "open_trunk chỉ mở cốp SAU", True),
    ("Phát nhạc Trịnh Công Sơn", "play_media(query=...) làm được", False),
    ("Nhảy sang bài sau", "play_media không có tham số chuyển bài", True),
]


def _render_vi_du_giam_khao() -> str:
    rong = max(len(c) for c, _, _ in VI_DU_GIAM_KHAO_TU_CHOI) + 2
    return "\n".join(
        f'- {chr(34) + c + chr(34):{rong}} -> {g} -> valid = {"true" if v else "false"}'
        for c, g, v in VI_DU_GIAM_KHAO_TU_CHOI
    )


def build_verify_negative_prompt(query: str) -> str:
    """Prompt chấm RIÊNG cho câu ngoài phạm vi.

    Không thể hỏi "lời gọi hàm có đúng ý câu hỏi không" như hàm thường — vì
    `irrelevant_function()` về bản chất KHÔNG BAO GIỜ trả lời được câu hỏi (đó
    chính là ý nghĩa của nó) -> hỏi kiểu đó thì teacher loại sạch (đã gặp thật:
    chỉ giữ 4.9%). Câu hỏi đúng phải là: "câu này có THỰC SỰ ngoài phạm vi không?"

    SỬA Ở VÒNG 2 — bản cũ chỉ liệt kê NHÓM chức năng ("điều hoà", "cốp", "phát nhạc").
    Chạy thử sinh câu từ chối khó: teacher loại oan 6/6 câu sát nách một hàm có thật —
    "bật lấy gió trong" (nó thấy thuộc nhóm "điều hoà"), "mở cốp trước" (có chữ "cốp"),
    "chuyển bài hát tiếp theo" (thuộc "phát nhạc"), "huỷ lộ trình đang dẫn"... Tức teacher
    72B mắc đúng bệnh gọi bừa của student 1,5B, và nếu để nguyên thì nó lọc sạch đúng
    những mẫu cần nhất. Sửa: đưa CHỮ KÝ CHÍNH XÁC từng hàm + luật "cùng chủ đề chưa đủ,
    phải viết được lời gọi làm đúng việc đó". Đây là sửa cho giám khảo hiểu đúng luật (như
    đã làm với giám khảo Llama), KHÔNG phải nới tay chấm: ví dụ mẫu có đủ cả hai chiều.
    """
    return f"""Bạn là người kiểm duyệt dữ liệu huấn luyện cho tác tử điều khiển xe ô tô.

Tác tử này có ĐÚNG các chức năng dưới đây (tên, tham số, giá trị cho phép; dấu ? là tuỳ
chọn). Ngoài những gì liệt kê ở đây, nó KHÔNG làm được gì khác:
{_catalog_ngan()}

Câu người dùng: "{query}"

Câu này đang được gán nhãn "NGOÀI PHẠM VI HỖ TRỢ". Hãy kiểm tra nhãn đó.

CÁCH KIỂM: thử viết một lời gọi hàm từ danh sách trên để thực hiện ĐÚNG yêu cầu này.
- Viết được, và lời gọi đó làm ĐÚNG việc người dùng muốn -> nhãn SAI  -> valid = false
- Không viết được vì không hàm nào, không tham số nào diễn đạt được việc đó
                                                         -> nhãn ĐÚNG -> valid = true

QUAN TRỌNG: CÙNG CHỦ ĐỀ CHƯA ĐỦ để coi là làm được. Phải có tham số diễn đạt đúng việc đó.
{_render_vi_du_giam_khao()}

Chỉ trả về đúng JSON, không thêm chữ nào khác:
{{"valid": true hoặc false, "reason": "lời gọi thử viết (nếu có) + lý do ngắn, tiếng Việt"}}"""


# Bộ HIỆU CHỈNH giám khảo từ chối: (câu, đáp án đúng valid). Không lấy từ tập test.
# Có đủ cả hai chiều — chỉ đo chiều "giữ câu ngoài phạm vi" thì một giám khảo gật đầu với
# mọi thứ cũng đạt 100%, mà giám khảo như thế sẽ để lọt "bật điều hoà" vào nhãn từ chối.
BO_HIEU_CHINH_TU_CHOI: list[tuple[str, bool]] = [
    # ngoài phạm vi, sát nách một hàm có thật -> phải GIỮ (valid = true)
    ("Bật sưởi ghế phụ lên", True),
    ("Chuyển điều hoà sang chế độ lấy gió ngoài", True),
    ("Chỉnh cửa gió điều hoà hướng xuống chân", True),
    ("Mở nắp capo ra", True),
    ("Mở cốp trước giúp tôi", True),
    ("Qua bài khác đi", True),
    ("Tua bài này tới đoạn điệp khúc", True),
    ("Thôi khỏi dẫn đường nữa, tắt đi", True),
    ("Gập gương hai bên vào", True),
    ("Bật gạt mưa tự động", True),
    # làm được -> phải LOẠI khỏi nhãn từ chối (valid = false)
    ("Bật điều hoà 23 độ", False),
    ("Tăng quạt gió lên mức 4", False),
    ("Sấy kính sau đi", False),
    ("Bật cốp sau lên cho tôi bỏ vali", False),
    ("Phát bài Hà Nội mùa thu", False),
    ("Dẫn đường tới sân bay Nội Bài", False),
    ("Hạ kính bên phụ xuống", False),
    ("Bật đèn khẩn cấp", False),
    ("Vặn nhỏ tiếng loa bớt 4 mức", False),
    ("Mở cửa sổ trời", False),
]


def parse_verify_response(text: str) -> VerifyResult:
    data = _extract_json(text)
    if not isinstance(data, dict) or "valid" not in data:
        raise TeacherResponseError(f"thiếu khoá 'valid': {data!r}")
    return VerifyResult(valid=bool(data["valid"]), reason=str(data.get("reason", "")))


# ── 6. Câu NGẦM HIỂU → đề xuất rồi hỏi lại (nhãn <nexa_ask>) ────────────────
# Hướng dẫn tình huống cho các hàm hay bị teacher "trôi" sang hàm hàng xóm khi sinh câu than phiền.
# Mỗi dòng dựa trên kiểu trôi ĐO ĐƯỢC ở vòng 2 (PLAN.md VÒNG 2 — D): tỉ lệ câu sinh ra được giữ đúng
# hàm gốc chỉ 13–51%. Viết ở mức KHÁI QUÁT, không câu ví dụ — người viết prompt đã đọc 28 câu test
# bổ sung, nên không đưa bất kỳ cách nói cụ thể nào vào đây (câu mồi dẫn dắt teacher mạnh nhất).
HUONG_DAN_NGAM: dict[str, str] = {
    "set_hazard_lights": (
        "Tình huống: xe CỦA CHÍNH người nói gặp trục trặc, hỏng hóc, hoặc buộc phải dừng tạm ở vị trí "
        "nguy hiểm trên đường mà xe khác có thể không nhìn thấy.\n"
        "KHÔNG viết về xe KHÁC hay đèn của xe khác (vòng trước teacher viết nhầm hàng loạt kiểu này). "
        "KHÔNG viết về nỗi lo an ninh khi đỗ xe (đó là khoá cửa)."
    ),
    "lock_doors": (
        "Tình huống: lo ngại về an ninh của xe hoặc đồ đạc trong xe — khi sắp rời xe, khi đỗ ở nơi lạ, "
        "khi có người lạ.\n"
        "KHÔNG viết về trời tối hay đường tối (đó là đèn). "
        "KHÔNG viết về sự cố trên đường (đó là đèn khẩn cấp)."
    ),
    "control_window": (
        "Tình huống: liên quan tới thứ đi qua CỬA KÍNH HAI BÊN — không khí, mùi, tiếng, mưa, bụi từ bên "
        "ngoài; hoặc cần giao tiếp, đưa nhận đồ với người đứng bên ngoài xe.\n"
        "KHÔNG than nóng lạnh chung chung (đó là điều hoà). KHÔNG than âm thanh trong xe (đó là âm lượng)."
    ),
    "operate_sunroof": (
        "Tình huống: liên quan tới PHÍA TRÊN ĐẦU, trần/nóc xe — bầu trời, ánh sáng hay thời tiết chiếu "
        "hoặc rơi từ trên xuống, cảm giác muốn thoáng phía trên.\n"
        "KHÔNG than nóng lạnh chung chung (đó là điều hoà). KHÔNG viết về cửa kính hai bên."
    ),
    "set_fan_speed": (
        "Tình huống: điều hoà ĐANG CHẠY, người nói chỉ than về LUỒNG GIÓ — thổi mạnh quá, yếu quá, "
        "ồn, phả vào người khó chịu.\n"
        "KHÔNG than nóng hay lạnh chung chung (đó là bật/tắt điều hoà — vòng trước 77% câu sinh cho quạt "
        "bị trôi sang kiểu này)."
    ),
    "navigate_to": (
        "Tình huống: người nói đã có sẵn MỘT ĐIỂM ĐẾN CỤ THỂ (nêu tên nơi đó trong câu) nhưng không biết "
        "hoặc không chắc đường đi.\n"
        "KHÔNG viết kiểu cần tìm 'một chỗ nào đó gần đây' như quán ăn, cây xăng bất kỳ (đó là tìm địa điểm)."
    ),
}


def build_implicit_query_prompt(fn: FunctionSpec, n: int, lang: str = "vi") -> str:
    """Sinh câu than thở / mô tả tình huống mà trợ lý nên HỎI LẠI có muốn dùng hàm này không.

    Lý do cần riêng: soi data full cho thấy câu positive có chữ "nóng" gần như đều đã
    ra lệnh thẳng ("Bật điều hoà lên, nóng quá") — prompt sinh câu thường không tạo ra
    câu ngầm hiểu thật, nên model sẽ không học được cách xử lý "Ôi nóng quá".
    """
    lang_name = "tiếng Việt" if lang == "vi" else "English"
    examples = "\n".join(f'- "{e}"' for e in fn.implicit_examples)
    huong_dan = HUONG_DAN_NGAM.get(fn.name)
    khoi_huong_dan = f"\nĐÚNG TÌNH HUỐNG — quan trọng nhất:\n{huong_dan}\n" if huong_dan else ""
    return f"""Bạn là người tạo dữ liệu huấn luyện cho một trợ lý giọng nói trên xe ô tô.

Chức năng của xe:
{fn.teacher_doc()}
{khoi_huong_dan}
Hãy viết {n} câu bằng {lang_name} mà người ngồi trong xe có thể buột miệng nói, KHÔNG ra lệnh
trực tiếp, nhưng một trợ lý tinh ý sẽ nghĩ tới việc HỎI LẠI xem họ có muốn dùng chức năng trên không.
Yêu cầu:
- Chỉ mô tả cảm giác, tình huống hoặc nhu cầu (nóng, lạnh, mệt, đói, ồn, tối, không nhìn rõ...)
- KHÔNG dùng động từ ra lệnh như: bật, tắt, mở, đóng, chỉnh, tăng, giảm, hạ, nâng, tìm, phát
- Đa dạng: than thở, cảm thán, nói bâng quơ, nói với người ngồi cạnh; có/không có từ đệm
- Không lặp lại cấu trúc câu, không đánh số thứ tự trong câu

Câu ví dụ tham khảo (đừng lặp lại y hệt):
{examples}

Chỉ trả về đúng JSON, không thêm chữ nào khác:
{{"queries": ["câu 1", "câu 2", ...]}}"""


def _catalog_ngan() -> str:
    """Mỗi hàm 1 dòng (tên, tham số rút gọn, mô tả) — đủ ngắn để cả catalog vừa context
    teacher, dùng khi teacher phải TỰ CHỌN hàm (không biết trước hàm đúng)."""
    lines = []
    for f in CATALOG.functions:
        if f.name == "irrelevant_function":
            continue
        params = []
        for p in f.params:
            s = p.name
            if p.enum:
                s += "∈{" + "|".join(map(str, p.enum)) + "}"
            elif p.minimum is not None or p.maximum is not None:
                s += f"[{p.minimum}..{p.maximum}]"
            if not p.required:
                s += "?"
            params.append(s)
        lines.append(f"- {f.name}({', '.join(params)}): {' '.join(f.description.split())}")
    return "\n".join(lines)


LOAI_CAU = ("ro_rang", "ngam_hieu", "ngoai_pham_vi")
_SO_DE_XUAT_TOI_DA = {"ro_rang": 1, "ngam_hieu": 2, "ngoai_pham_vi": 0}

# Hàm KHÔNG được phép xuất hiện trong lời đề xuất, dù teacher có chọn (xem PLAN.md,
# dòng "Luật an toàn"). Câu mơ hồ không đủ căn cứ để mở khoá/mở cốp — đợt chạy thử đã
# thấy thật: "bạn có thể mở cửa hàng khi tôi đang xem phim không" -> đề xuất unlock_doors.
HAM_KHONG_DE_XUAT = frozenset(
    {"unlock_doors", "open_trunk", "open_charge_port", "open_fuel_flap", "irrelevant_function"}
)

# Giá trị bị thay khi ĐỀ XUẤT (câu ngầm hiểu). Không áp dụng cho lệnh rõ ràng.
#
# Đợt chạy 200 câu/hàm: 97/123 mẫu đèn đề xuất `high_beam` cho câu kiểu "trời tối quá,
# không nhìn rõ đường". Đèn pha xa làm loá xe ngược chiều (kể cả trong hầm), nên không
# phải thứ nên GỢI Ý mặc định. `auto` cũng không hợp: người dùng muốn sáng NGAY, không
# phải giao quyền cho cảm biến. -> gợi ý đèn chiếu gần; ai cần pha xa thì nói thẳng
# ("bật pha xa lên"), câu đó thuộc nhóm lệnh rõ ràng và được thực hiện nguyên vẹn.
DOI_GIA_TRI_KHI_DE_XUAT: dict[tuple[str, str, Any], Any] = {
    ("set_headlights", "mode", "high_beam"): "low_beam",
}


# Đề xuất vô lý, loại luôn cả mẫu. Đã gặp thật: "Trời hôm nay thật đẹp, muốn tận hưởng"
# và "Nắng nhẹ nhàng lắm" -> set_headlights(mode="off"). Tắt đèn không bao giờ là việc
# nên GỢI Ý từ một câu than vãn; ai muốn tắt thì nói thẳng.
DE_XUAT_BI_LOAI: frozenset[tuple[str, str, Any]] = frozenset({("set_headlights", "mode", "off")})


# Tham số TUỲ CHỌN được GIỮ LẠI trong lời đề xuất khi câu nói đã nêu rõ.
#
# Quy ước ban đầu ("bỏ sạch tham số tuỳ chọn") chặn được việc teacher bịa số
# (`temperature=22` cho câu "nóng quá"), nhưng bỏ nhầm cả thông tin người dùng ĐÃ nói:
# "Kính sau đọng hơi" -> set_defrost(state="on") -> runtime dùng mặc định `front` ->
# sấy nhầm kính. Quy ước đúng: bỏ tham số phải BỊA, giữ tham số câu NÓI RÕ.
TU_KHOA_THAM_SO: dict[tuple[str, str], dict[Any, tuple[str, ...]]] = {
    ("set_defrost", "zone"): {
        "both": ("cả hai kính", "cả kính trước lẫn kính sau", "cả trước lẫn sau"),
        "rear": ("kính sau", "kính hậu", "kính phía sau"),
        "front": ("kính trước", "kính lái", "kính chắn gió", "kính phía trước"),
    },
    ("set_ac", "zone"): {
        "rear": ("hàng ghế sau", "ghế sau", "băng sau"),
        "passenger": ("ghế phụ", "bên phụ", "ghế bên cạnh"),
        "driver": ("ghế lái", "bên tài", "ghế tài"),
        "all": ("cả xe", "toàn xe"),
    },
    ("control_window", "window"): {
        "all": ("tất cả cửa sổ", "hết cửa sổ", "bốn kính", "4 kính"),
        "rear_left": ("sau bên trái", "sau trái"),
        "rear_right": ("sau bên phải", "sau phải"),
        "front_right": ("ghế phụ", "bên phụ"),
        "front_left": ("ghế lái", "bên tài"),
    },
    ("get_tire_pressure", "wheel"): {
        "front_left": ("trước bên trái", "trước trái"),
        "front_right": ("trước bên phải", "trước phải"),
        "rear_left": ("sau bên trái", "sau trái"),
        "rear_right": ("sau bên phải", "sau phải"),
    },
}

# Đề xuất SAI NGỮ CẢNH — loại theo nội dung câu, không theo giá trị tham số.
# Đã gặp thật: "Ngồi ở phía sau, kính mờ quá" -> operate_sunroof(action="open").
# Kính mờ là việc của sấy kính, mở cửa sổ trời không giải quyết gì.
_KINH_MO = re.compile(r"kính[^.]{0,20}(mờ|đọng|hơi nước)|(mờ|đọng|hơi nước)[^.]{0,20}kính", re.I)
DE_XUAT_SAI_NGU_CANH: tuple[tuple[re.Pattern[str], frozenset[str]], ...] = (
    (_KINH_MO, frozenset({"operate_sunroof", "control_window"})),
)


def tham_so_duoc_noi_ro(fn_name: str, query: str) -> dict[str, Any]:
    """Các tham số tuỳ chọn suy được TRỰC TIẾP từ câu nói (không phải bịa)."""
    thap = " ".join(query.split()).lower()
    ket_qua: dict[str, Any] = {}
    for (ten_ham, ten_tham_so), bang in TU_KHOA_THAM_SO.items():
        if ten_ham != fn_name:
            continue
        for gia_tri, tu_khoa in bang.items():  # thứ tự khai báo = mức ưu tiên
            if any(t in thap for t in tu_khoa):
                ket_qua[ten_tham_so] = gia_tri
                break
    return ket_qua


def loc_tham_so_de_xuat(fn_name: str, args: dict[str, Any], query: str = "") -> dict[str, Any]:
    """Giữ tham số bắt buộc + tham số câu nói rõ; bỏ phần teacher tự bịa."""
    fn = CATALOG.by_name[fn_name]
    bat_buoc = {p.name for p in fn.params if p.required}
    giu = {k: v for k, v in args.items() if k in bat_buoc}
    giu.update(tham_so_duoc_noi_ro(fn_name, query))
    return giu


def ap_dung_chinh_sach_de_xuat(fn_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Thay các giá trị không nên gợi ý mặc định. Dùng cho cả data mới lẫn data đã sinh."""
    return {k: DOI_GIA_TRI_KHI_DE_XUAT.get((fn_name, k, v), v) for k, v in args.items()}


def de_xuat_hop_le(fn_name: str, args: dict[str, Any], query: str = "") -> bool:
    """False nếu lời đề xuất này vô lý -> bỏ riêng đề xuất đó (hết đề xuất thì bỏ cả mẫu)."""
    if any((fn_name, k, v) in DE_XUAT_BI_LOAI for k, v in args.items()):
        return False
    return not any(mau.search(query) and fn_name in ds_ham for mau, ds_ham in DE_XUAT_SAI_NGU_CANH)


def build_ask_label_prompt(query: str) -> str:
    """Teacher xếp 1 câu vào 3 loại + đề xuất lời gọi. Dùng cho CẢ câu ngầm hiểu mới sinh
    LẪN negative cũ (tìm câu ngầm hiểu bị gán nhầm "ngoài phạm vi")."""
    return f"""Bạn là người gán nhãn dữ liệu cho trợ lý giọng nói trên xe ô tô.

Các chức năng trợ lý làm được (dấu ? là tham số tuỳ chọn):
{_catalog_ngan()}

Câu người trong xe vừa nói: "{query}"

Xếp câu này vào ĐÚNG 1 trong 3 loại:
- "ro_rang": người nói YÊU CẦU hoặc HỎI THẲNG một việc thuộc các chức năng trên (kể cả khi
  thiếu chi tiết) → trợ lý làm luôn, không cần hỏi lại.
- "ngam_hieu": người nói KHÔNG yêu cầu thẳng, chỉ than thở / mô tả cảm giác, tình huống,
  nhưng có 1–2 chức năng ở trên giúp được → trợ lý nên HỎI LẠI trước khi làm.
- "ngoai_pham_vi": không chức năng nào ở trên giúp được.

Trường de_xuat:
- "ro_rang": đúng 1 chức năng cần gọi
- "ngam_hieu": **thường chỉ 1 chức năng** phù hợp nhất. Chỉ ghi 2 khi thật sự có HAI CÁCH
  KHÁC NHAU để giải quyết (ví dụ "trong xe bí quá" -> bật điều hoà HOẶC hạ kính). Đừng ghi
  2 chức năng cùng nhóm chỉ để cho đủ
- "ngoai_pham_vi": danh sách rỗng
Tham số: chỉ điền tham số BẮT BUỘC và tham số suy ra rõ ràng từ câu; không tự bịa nhiệt độ,
mức, phần trăm hay địa điểm mà câu không nhắc tới. (Với loại "ngam_hieu", tham số tuỳ chọn
sẽ bị bỏ tự động — cứ để trống.)

Chỉ trả về đúng JSON, không thêm chữ nào khác:
{{"loai": "ro_rang | ngam_hieu | ngoai_pham_vi",
 "de_xuat": [{{"ten_ham": "...", "tham_so": {{}}}}],
 "ly_do": "ngắn gọn, tiếng Việt"}}"""


@dataclass
class AskLabel:
    loai: str
    calls: list[tuple[str, dict[str, Any]]]
    ly_do: str


def parse_ask_label_response(text: str, query: str = "") -> AskLabel:
    """Đọc nhãn 3 loại. Đề xuất có tên hàm lạ / tham số sai bị bỏ RIÊNG đề xuất đó
    (các đề xuất còn lại vẫn là gợi ý hợp lệ); không còn đề xuất nào thì loại cả mẫu."""
    data = _extract_json(text)
    if not isinstance(data, dict) or data.get("loai") not in LOAI_CAU:
        raise TeacherResponseError(f"'loai' phải thuộc {LOAI_CAU}: {data!r}")
    loai = data["loai"]
    raw = data.get("de_xuat") or []
    if not isinstance(raw, list):
        raise TeacherResponseError(f"'de_xuat' phải là list: {raw!r}")

    calls: list[tuple[str, dict[str, Any]]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        ten = item.get("ten_ham")
        fn = CATALOG.by_name.get(ten) if isinstance(ten, str) else None
        if fn is None or any(ten == n for n, _ in calls):
            continue
        if loai == "ngam_hieu" and fn.name in HAM_KHONG_DE_XUAT:
            continue  # chặn cứng bằng code, không tin vào prompt
        if fn.name == "irrelevant_function":
            continue
        tham_so = item.get("tham_so") or {}
        if not isinstance(tham_so, dict):
            continue
        try:
            args = _kiem_tra_tham_so(tham_so, fn)
        except TeacherResponseError:
            continue
        if loai == "ngam_hieu":
            # Bỏ tham số TUỲ CHỌN: câu mơ hồ không nói ra con số nào, teacher tự bịa
            # (đã gặp thật: "nóng quá" -> temperature=22, zone="all"). Giữ lại chỉ làm
            # nhãn mâu thuẫn nhau và dạy model đoán bừa. Người dùng xác nhận xong thì
            # runtime dùng giá trị mặc định trong schema.
            args = loc_tham_so_de_xuat(fn.name, args, query)
            args = ap_dung_chinh_sach_de_xuat(fn.name, args)
            if not de_xuat_hop_le(fn.name, args, query):
                continue
        calls.append((fn.name, args))

    # LUẬT "CHỈ ĐỌC KHÔNG HỎI LẠI": hỏi lại là cơ chế chống làm sai. Hàm chỉ đọc số liệu
    # (xem xăng, pin, áp suất lốp...) thì không có gì để làm sai — hỏi thêm chỉ làm phiền
    # người đang lái. Câu "hình như xe non hơi một bánh" nên kiểm tra rồi báo luôn, chứ
    # không phải "bạn có muốn tôi kiểm tra áp suất lốp không?".
    if loai == "ngam_hieu" and calls and all(CATALOG.by_name[n].chi_doc for n, _ in calls):
        loai = "ro_rang"

    calls = calls[: _SO_DE_XUAT_TOI_DA[loai]]
    if loai != "ngoai_pham_vi" and not calls:
        raise TeacherResponseError(f"loại {loai} nhưng không còn đề xuất hợp lệ nào: {raw!r}")
    return AskLabel(loai=loai, calls=calls, ly_do=str(data.get("ly_do", "")))


def build_verify_ask_prompt(query: str, calls: list[tuple[str, dict[str, Any]]]) -> str:
    """Chấm riêng mẫu "hỏi lại": câu có thật sự KHÔNG ra lệnh thẳng không (ra lệnh thẳng
    mà hỏi lại là sai — phiền user), và đề xuất có phải việc HỢP LÝ NHẤT không.

    SỬA Ở VÒNG 2 — bản cũ chỉ cho giám khảo thấy ĐÚNG hàm được đề xuất và hỏi "có giúp được
    không". Kết quả: giữ 517/518 mẫu, kể cả "Chỗ đỗ xe này trông không an toàn" -> bật đèn
    khẩn cấp, "Điếc tai với tiếng ồn bên ngoài" -> chỉnh âm lượng. Nhìn một hàm đơn lẻ thì
    gần như hàm nào cũng "giúp được một chút". Sửa: đưa CẢ danh sách hàm và hỏi có hàm nào
    RÕ RÀNG hợp hơn không (khoá cửa, đóng kính) — như cách đã sửa giám khảo từ chối.
    """
    from schema.call_format import render_readable

    de_xuat = "\n".join(f"- {render_readable(name, args)}" for name, args in calls)
    return f"""Bạn là người kiểm duyệt dữ liệu huấn luyện cho trợ lý giọng nói trên xe ô tô.

Toàn bộ chức năng của trợ lý (dấu ? là tham số tuỳ chọn):
{_catalog_ngan()}

Câu người trong xe vừa nói: "{query}"

Trợ lý định KHÔNG làm ngay mà HỎI LẠI người dùng có muốn làm việc sau không:
{de_xuat}

QUY TẮC PHÂN BIỆT (của hệ thống này, hãy theo đúng, đừng dùng tiêu chuẩn riêng của bạn):
- "YÊU CẦU THẲNG" = câu có động từ ra lệnh nhắm vào chức năng của xe: bật, tắt, mở, đóng,
  chỉnh, tăng, giảm, hạ, nâng, tra, dẫn đường, phát... hoặc câu hỏi trực tiếp về số liệu xe
  ("còn bao nhiêu xăng"). Loại này phải LÀM LUÔN, hỏi lại là SAI.
- "MÔ TẢ TÌNH HUỐNG" = câu chỉ than vãn, kể cảm giác, nêu vấn đề mà KHÔNG ra lệnh:
  "nóng quá", "kính mờ hết rồi", "khát nước quá", "đường tối thế". Loại này phải HỎI LẠI —
  kể cả khi người nghe đoán được ý, vì đoán sai thì tự ý làm việc người dùng không yêu cầu.

Kiểm tra THEO ĐÚNG THỨ TỰ, bước 1 sai thì dừng luôn:

BƯỚC 1 — câu có phải "YÊU CẦU THẲNG" không? Nếu câu có động từ ra lệnh nhắm vào xe (bật,
  tắt, mở, đóng, chỉnh, tìm, chỉ đường, phát...) thì người dùng ĐÃ NÓI RÕ muốn gì -> trợ lý
  phải làm luôn -> hỏi lại là SAI -> valid = false, KHÔNG xét bước 2. Dù hàm đề xuất có đúng
  đến đâu cũng vậy. Ví dụ: "Mở nhạc lên" -> yêu cầu thẳng -> valid = false.

BƯỚC 2 — chỉ khi câu là "MÔ TẢ TÌNH HUỐNG": việc được đề xuất có phải việc HỢP LÝ NHẤT trong
  danh sách chức năng ở trên không? Nếu có một chức năng KHÁC rõ ràng hợp hơn với tình huống
  thì valid = false. "Giúp được một chút" là CHƯA ĐỦ. Ví dụ:
    · thấy chỗ đỗ không an toàn -> khoá cửa, KHÔNG phải bật đèn khẩn cấp (đèn khẩn cấp chỉ
      dùng khi xe dừng/hỏng trên đường)
    · ồn vì tiếng bên ngoài -> đóng kính, KHÔNG phải chỉnh âm lượng

BƯỚC 3 — tham số có bịa chi tiết mà câu không nhắc tới không? Chỉ tham số BẮT BUỘC mới cần có mặt;
  tham số tuỳ chọn bỏ trống là ĐÚNG, hàm không có tham số thì gọi rỗng cũng ĐÚNG.

Chỉ trả về đúng JSON, không thêm chữ nào khác:
{{"valid": true hoặc false, "reason": "bước nào sai (nếu có) + lý do ngắn, tiếng Việt"}}"""


def build_chon_de_xuat_prompt(query: str) -> str:
    """Giám khảo TỰ CHỌN hàm cho câu than phiền — KHÔNG được xem nhãn (Zheng 2023,
    reference-guided judging: tự giải trước rồi mới so; so sánh làm bằng CODE trong
    gen/cham_ask.py). Thay cho `build_verify_ask_prompt`, vốn hỏi "đề xuất X có hợp lý
    không?" nên giám khảo gật đầu với 517/518 mẫu. Xem PLAN.md mục E0b.

    Không nhắc gì tới "câu ra lệnh" hay "hỏi lại": việc đó code đã làm ở lớp trước; giám
    khảo chỉ làm đúng một việc (ToolACE: mỗi câu hỏi một việc).
    """
    return f"""Bạn là trợ lý giọng nói trên xe ô tô. Đây là toàn bộ chức năng bạn có (dấu ? là tuỳ chọn):
{_catalog_ngan()}

Người trong xe vừa nói: "{query}"

Họ không ra lệnh gì cụ thể. Nếu bạn được phép GỢI Ý một việc để giúp họ, bạn sẽ gợi ý
chức năng nào? Chọn 1 chức năng hợp lý nhất. Chỉ chọn thêm chức năng thứ 2 nếu nó hợp
lý NGANG BẰNG chức năng thứ nhất (không phải "cũng giúp được chút").

Chỉ trả về đúng JSON, không thêm chữ nào khác:
{{"ham": ["tên_hàm"], "ly_do": "ngắn gọn, tiếng Việt"}}"""


# Bộ HIỆU CHỈNH giám khảo hỏi lại: (câu, đề xuất, đáp án đúng valid). KHÔNG lấy từ tập test.
# Ba loại, để đo đủ các cách giám khảo có thể sai:
#   đúng        — câu tình huống + đề xuất hợp nhất                      -> phải GIỮ
#   sai hàm     — cùng câu nhưng đề xuất hàm "giúp được chút" chứ không hợp nhất -> phải LOẠI
#   ra lệnh     — câu ra lệnh thẳng mà bị gán hỏi lại                     -> phải LOẠI
BO_HIEU_CHINH_HOI_LAI: list[tuple[str, list[tuple[str, dict[str, Any]]], bool]] = [
    ("Nóng như cái lò trong này", [("set_ac", {"state": "on"})], True),
    ("Kính trước mờ hết rồi, chẳng thấy gì", [("set_defrost", {"state": "on", "zone": "front"})], True),
    ("Tối thui chẳng thấy đường đâu", [("set_headlights", {"mode": "low_beam"})], True),
    ("Bãi này vắng hoe, để xe ở đây thấy lo lo", [("lock_doors", {})], True),
    ("Ngồi im lặng suốt buồn tẻ ghê", [("play_media", {})], True),
    ("Xe tự dưng khựng lại giữa làn, phải dừng gấp", [("set_hazard_lights", {"state": "on"})], True),
    ("Tiếng xe cộ bên ngoài ồn ào quá", [("control_window", {"window": "all", "position": "up"})], True),
    ("Bụng đói meo rồi", [("find_nearby", {"category": "restaurant"})], True),
    # sai hàm — đúng những kiểu nhãn sai đã thấy trong dữ liệu vòng 2
    ("Bãi này vắng hoe, để xe ở đây thấy lo lo", [("set_hazard_lights", {"state": "on"})], False),
    ("Tiếng xe cộ bên ngoài ồn ào quá", [("set_volume", {"delta": 3})], False),
    ("Nóng như cái lò trong này", [("set_headlights", {"mode": "auto"})], False),
    ("Bụng đói meo rồi", [("set_ac", {"state": "on"})], False),
    ("Rét cóng hết tay chân rồi", [("operate_sunroof", {"action": "open"})], False),
    # ra lệnh thẳng — phải làm luôn, hỏi lại là sai
    ("Bật điều hoà lên 22 độ", [("set_ac", {"state": "on", "temperature": 22})], False),
    ("Khoá hết cửa lại cho tôi", [("lock_doors", {})], False),
    ("Chỉ cho tôi trạm đổ xăng nào quanh đây", [("find_nearby", {"category": "gas_station"})], False),
]
