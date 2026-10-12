"""Pipeline sinh dữ liệu trên Modal — chạy teacher Qwen2.5-72B-Instruct bằng vLLM.

Đây là file DUY NHẤT tốn tiền GPU. Mọi logic soạn prompt / đọc kết quả / ghép mẫu
đều nằm ở gen/teacher_prompts.py, gen/simulate.py, gen/records.py (đã test local).

CÁCH CHẠY (theo đúng nguyên tắc an toàn đã chốt):

  # 1. Tải model về Volume TRƯỚC — máy rẻ, không GPU, tránh trả tiền A100 lúc ngồi tải
  modal run gen/modal_app.py::download_teacher

  # 2. Chạy thử mẫu nhỏ (2 hàm × 5 câu) — kiểm tra prompt/parser ổn trước khi chạy full
  modal run gen/modal_app.py::generate --smoke-test

  # 3. Chạy thật, --detach để đóng terminal/tắt máy vẫn chạy tiếp
  modal run --detach gen/modal_app.py::generate --queries-per-fn 300

  # 4. Sinh bổ sung nhóm câu NGẦM HIỂU (nhãn <nexa_ask>) — chạy riêng, không đụng dataset cũ
  modal run --detach gen/modal_app.py::generate_ask --queries-per-fn 50 --so-negative-cham-lai 500

  # Xem tiến độ / tải kết quả về
  modal run gen/modal_app.py::show_progress
  modal volume get nexa-drive /data/vi/train_vi.jsonl ./data/processed/

AN TOÀN:
  • timeout đặt trần rõ ràng (6h cho sinh data) — chặn vòng lặp vô hạn ngốn tiền
  • mỗi stage ghi ngay ra Volume -> đứt giữa chừng chạy lại không mất phần đã xong
  • retry có trần (MAX_RETRY) -> không lặp vô hạn khi teacher liên tục trả sai
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal

# ── cấu hình ────────────────────────────────────────────────────────────────
TEACHER_MODEL = "Qwen/Qwen2.5-72B-Instruct-AWQ"  # bản 4-bit ~40GB, vừa A100 80GB
TEACHER_GPU = "A100-80GB"  # tier BẮT BUỘC cho teacher 72B (xem PLAN.md B5)
VLLM_VERSION = "0.29.0"  # ghim bản cụ thể để tái lập được; vLLM tự kéo torch/transformers khớp nhau

# Giám khảo cho bước thẩm định chéo: PHẢI khác họ với teacher. Qwen vừa sinh data vừa
# tự chấm thì con số chất lượng không dùng làm bằng chứng được (đã thấy thật: cho Qwen
# chấm lại 500 negative, nó đổi nhãn 78 câu và phần lớn sai). Jafarzadeh Asl et al. 2026
# §A.7 dùng model khác họ làm giám khảo và chỉ coi kết quả là chẩn đoán, không tự xoá mẫu.
# Chọn Llama-3.3-70B: khác họ Meta, ngang tầm teacher 72B, tiếng Việt khá. Bản AWQ cộng
# đồng (không bị khoá quyền tải như repo gốc của Meta).
JUDGE_MODEL = "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4"

VOL_NAME = "nexa-drive"
VOL_PATH = "/vol"
MODEL_DIR = f"{VOL_PATH}/models/teacher"
JUDGE_DIR = f"{VOL_PATH}/models/judge"
DATA_DIR = f"{VOL_PATH}/data"

TIMEOUT_DOWNLOAD = 60 * 60 * 2  # 2h — tải ~40GB
TIMEOUT_GENERATE = 60 * 60 * 6  # 6h — trần an toàn, chạy thật dự kiến 2-4h
MAX_RETRY = 3  # trần số lần sinh lại 1 mẫu hỏng

app = modal.App("nexa-drive-gen")
volume = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Dùng image CUDA bản "devel" (có nvcc) thay vì debian_slim: vLLM 0.29 cần trình biên
# dịch CUDA cả ở bước compile model LẪN bước warmup kernel sampler — thiếu nvcc là crash
# ngay lúc khởi tạo engine, không né được bằng enforce_eager.
# Chọn CUDA 13.x cho khớp với các gói nvidia-*-cu13 mà vLLM 0.29 tự kéo về.
image = (
    modal.Image.from_registry("nvidia/cuda:13.3.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install(
        f"vllm=={VLLM_VERSION}",
        "huggingface_hub[hf_transfer]>=0.26",
        "pyyaml>=6.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})  # tải nhanh (biến HF_HUB_ENABLE_HF_TRANSFER đã lỗi thời)
    # Đưa code local lên container: cần cả .py lẫn functions.yaml
    .add_local_dir(REPO_ROOT / "schema", "/root/schema")
    .add_local_dir(REPO_ROOT / "gen", "/root/gen")
    .add_local_dir(REPO_ROOT / "eval", "/root/eval")
)


# ── bước 1: tải model (KHÔNG GPU — rẻ) ──────────────────────────────────────
@app.function(image=image, volumes={VOL_PATH: volume}, timeout=TIMEOUT_DOWNLOAD)
def download_teacher() -> None:
    """Tải trọng số teacher về Volume. Chạy trên máy CPU thường, không bật A100."""
    from huggingface_hub import snapshot_download

    print(f"Tải {TEACHER_MODEL} về {MODEL_DIR} ...")
    t0 = time.time()
    snapshot_download(TEACHER_MODEL, local_dir=MODEL_DIR)
    volume.commit()
    print(f"Xong sau {time.time() - t0:.0f}s")


@app.function(image=image, volumes={VOL_PATH: volume}, timeout=TIMEOUT_DOWNLOAD)
def download_judge(model: str = JUDGE_MODEL) -> None:
    """Tải model GIÁM KHẢO về Volume. Máy CPU, không bật GPU."""
    from huggingface_hub import snapshot_download

    print(f"Tải {model} về {JUDGE_DIR} ...")
    t0 = time.time()
    snapshot_download(model, local_dir=JUDGE_DIR)
    volume.commit()
    print(f"Xong sau {time.time() - t0:.0f}s")


# ── tiện ích đọc/ghi Volume (resumable) ─────────────────────────────────────
def _stage_path(lang: str, ten: str) -> str:
    return f"{DATA_DIR}/{lang}/{ten}.jsonl"


def _ghi_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    volume.commit()  # đẩy ngay ra Volume — đứt giữa chừng vẫn còn


def _doc_jsonl(path: str) -> list[dict[str, Any]] | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ── nạp teacher (dùng chung cho mọi hàm sinh data) ────────────────────────
def _khoi_tao_teacher():
    """Nạp teacher lên GPU, trả về (sinh, chat). Chỉ gọi BÊN TRONG hàm chạy trên A100."""
    from vllm import LLM, SamplingParams

    # Không ép quantization="awq" — để vLLM tự chọn kernel awq_marlin (nhanh hơn hẳn
    # trên A100/Ampere; bản awq thường bị cảnh báo "not fully optimized").
    #
    # enforce_eager=True: tắt torch.compile/CUDA graph — bớt một lớp biên dịch dễ lỗi.
    # Chấp nhận được với workload này: sinh data theo LÔ LỚN (GPU vốn đã bận tính toán,
    # chi phí khởi động kernel không đáng kể) — khác với serving cần độ trễ thấp.
    # Kernel Marlin vẫn hoạt động bình thường, eager chỉ tắt lớp tối ưu đồ thị.
    llm = LLM(
        model=MODEL_DIR,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        enforce_eager=True,
    )

    def sinh(prompts: list[str], temperature: float, max_tokens: int) -> list[str]:
        """Gọi vLLM theo LÔ — đây là chỗ vLLM nhanh hơn gọi lần lượt hàng chục lần."""
        if not prompts:
            return []
        params = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=max_tokens)
        outs = llm.generate(prompts, params)
        return [o.outputs[0].text for o in outs]

    def chat(noi_dung: str) -> str:
        """Bọc 1 yêu cầu dạng chat vào ChatML cho teacher (teacher cũng là Qwen-Instruct)."""
        return f"<|im_start|>user\n{noi_dung}<|im_end|>\n<|im_start|>assistant\n"

    return sinh, chat


# ── bước 2: sinh dữ liệu (CẦN A100 80GB — tốn tiền) ────────────────────────
@app.function(
    image=image,
    gpu=TEACHER_GPU,
    volumes={VOL_PATH: volume},
    timeout=TIMEOUT_GENERATE,
)
def generate(
    queries_per_fn: int = 300,
    lang: str = "vi",
    smoke_test: bool = False,
    force_restart: bool = False,
) -> dict[str, Any]:
    """Sinh dataset đầy đủ theo 4 stage, mỗi stage lưu ngay ra Volume.

    smoke_test=True: chỉ 2 hàm × 5 câu — dùng để kiểm tra pipeline trước khi
    chạy full (rẻ, vài phút, bắt lỗi prompt/parser sớm).
    """
    from gen.loc import cau_hop_le, tham_so_hop_le
    from gen.records import CallRecord, ReplyRecord, datasheet, negative_record
    from gen.simulate import simulate_result
    from gen.teacher_prompts import (
        TeacherResponseError,
        build_call_gen_prompt,
        build_negative_query_prompt,
        build_query_gen_prompt,
        build_reply_gen_prompt,
        build_verify_negative_prompt,
        build_verify_prompt,
        parse_call_gen_response,
        parse_query_gen_response,
        parse_reply_gen_response,
        parse_verify_response,
    )
    from schema.loader import CATALOG

    if smoke_test:
        queries_per_fn, lang = 5, lang
        ham_can_lam = [f for f in CATALOG.functions if f.name in ("set_ac", "open_trunk")]
        print(f"[SMOKE TEST] chỉ {len(ham_can_lam)} hàm × {queries_per_fn} câu")
    else:
        ham_can_lam = [f for f in CATALOG.functions if f.name != "irrelevant_function"]

    sinh, chat = _khoi_tao_teacher()

    thong_ke: dict[str, Any] = {"bo_qua": {}}

    # ── STAGE 1: sinh câu hỏi ────────────────────────────────────────────
    p1 = _stage_path(lang, "stage1_queries")
    cached = None if force_restart else _doc_jsonl(p1)
    if cached:
        print(f"[stage 1] dùng lại {len(cached)} câu đã sinh trước đó")
        queries = cached
    else:
        # QUAN TRỌNG — chia lô: KHÔNG xin 1000 câu trong 1 prompt.
        # max_tokens của 1 lượt sinh là 4096; xin quá nhiều thì output bị cắt giữa
        # chừng -> JSON hỏng -> mất TRẮNG cả lô. Đã gặp thật ở lần chạy 1000/hàm:
        # 17/24 hàm mất sạch mẫu vì lý do này. Xin 50 câu/lô là mức an toàn.
        CAU_MOI_LO = 50
        prompts, nhan = [], []
        for fn in ham_can_lam:
            so_lo = max(1, -(-queries_per_fn // CAU_MOI_LO))  # làm tròn lên
            xin_moi_lo = min(queries_per_fn, CAU_MOI_LO)
            for _ in range(so_lo):
                prompts.append(chat(build_query_gen_prompt(fn, xin_moi_lo, lang)))
                nhan.append(fn.name)
        if not smoke_test:  # câu ngoài phạm vi, số lượng ~ tổng positive (tỉ lệ 1:1)
            so_negative = queries_per_fn * len(ham_can_lam)
            for _ in range(max(1, so_negative // CAU_MOI_LO)):
                prompts.append(chat(build_negative_query_prompt(CAU_MOI_LO, lang)))
                nhan.append("irrelevant_function")

        print(f"[stage 1] sinh câu hỏi cho {len(prompts)} lô ...")
        outs = sinh(prompts, temperature=0.95, max_tokens=4096)

        queries = []
        bo_vi_loc = 0
        for fn_name, out in zip(nhan, outs, strict=True):
            try:
                for q in parse_query_gen_response(out):
                    # Lọc chương trình: chữ Trung/Nhật/Hàn (Qwen hay chèn) + lộ tên hàm.
                    # Rẻ hơn và đáng tin hơn nhờ teacher chấm (xem gen/loc.py).
                    ok, _ = cau_hop_le(q)
                    if not ok:
                        bo_vi_loc += 1
                        continue
                    queries.append({"query": q, "fn": fn_name})
            except TeacherResponseError as e:
                thong_ke["bo_qua"][f"stage1:{fn_name}"] = str(e)[:200]
        thong_ke["stage1_bo_vi_loc"] = bo_vi_loc
        print(f"[stage 1] bộ lọc chương trình loại {bo_vi_loc} câu (chữ Đông Á / lộ tên hàm)")

        # Khử trùng lặp NGAY tại đây, trước khi stage 2/4 xử lý — tiết kiệm GPU thật:
        # phần negative dùng chung 1 prompt lặp lại hàng trăm lô nên trùng rất nhiều,
        # mỗi câu trùng nếu để lọt vẫn bị stage 4 chấm (tốn tiền vô ích).
        # Đây là khử trùng khớp chính xác; khử gần-trùng (cosine embedding) làm sau ở local.
        truoc_khu = len(queries)
        da_thay: set[tuple[str, str]] = set()
        khong_trung = []
        for q in queries:
            khoa = (q["fn"], " ".join(q["query"].split()).lower())
            if khoa in da_thay:
                continue
            da_thay.add(khoa)
            khong_trung.append(q)
        queries = khong_trung
        thong_ke["trung_lap_da_bo"] = truoc_khu - len(queries)

        _ghi_jsonl(p1, queries)
        print(f"[stage 1] xong: {len(queries)} câu (đã bỏ {truoc_khu - len(queries)} câu trùng)")

        # In số câu THEO TỪNG HÀM + cảnh báo hàm thiếu — nếu chỉ in tổng thì một
        # hàm chết sạch vẫn trông "bình thường" (đã gặp thật: log báo 15.510 câu
        # nhưng 17/24 hàm không có mẫu nào).
        dem_theo_ham: dict[str, int] = {}
        for q in queries:
            dem_theo_ham[q["fn"]] = dem_theo_ham.get(q["fn"], 0) + 1
        print("[stage 1] số câu theo hàm:")
        thieu_nghiem_trong = []
        for fn in ham_can_lam:
            n = dem_theo_ham.get(fn.name, 0)
            canh_bao = ""
            if n < queries_per_fn * 0.5:
                canh_bao = "  <-- THIẾU NGHIÊM TRỌNG"
                thieu_nghiem_trong.append(fn.name)
            print(f"           {fn.name:24} {n:>6}{canh_bao}")
        print(f"           {'irrelevant_function':24} {dem_theo_ham.get('irrelevant_function', 0):>6}")
        if thieu_nghiem_trong:
            print(
                f"[stage 1] CẢNH BÁO: {len(thieu_nghiem_trong)} hàm có <50% số câu "
                f"yêu cầu: {thieu_nghiem_trong}"
            )
        thong_ke["ham_thieu_o_stage1"] = thieu_nghiem_trong

    # ── STAGE 2: trích tham số ───────────────────────────────────────────
    p2 = _stage_path(lang, "stage2_calls")
    cached = None if force_restart else _doc_jsonl(p2)
    if cached:
        print(f"[stage 2] dùng lại {len(cached)} lời gọi đã sinh trước đó")
        calls = cached
    else:
        can_trich = [q for q in queries if q["fn"] != "irrelevant_function"]
        prompts = [chat(build_call_gen_prompt(q["query"], CATALOG.by_name[q["fn"]])) for q in can_trich]
        print(f"[stage 2] trích tham số cho {len(prompts)} câu ...")
        outs = sinh(prompts, temperature=0.1, max_tokens=512)  # cần chính xác, không sáng tạo

        calls = []
        for q, out in zip(can_trich, outs, strict=True):
            try:
                args = parse_call_gen_response(out, CATALOG.by_name[q["fn"]])
                ok, ly_do = tham_so_hop_le(args)  # đã gặp thật: destination="火車站"
                if not ok:
                    thong_ke["bo_qua"][f"stage2:{q['query'][:40]}"] = ly_do
                    continue
                calls.append({"query": q["query"], "fn": q["fn"], "args": args})
            except TeacherResponseError as e:
                thong_ke["bo_qua"][f"stage2:{q['query'][:40]}"] = str(e)[:200]
        # câu ngoài phạm vi: nhãn cố định, không cần hỏi teacher
        for q in queries:
            if q["fn"] == "irrelevant_function":
                calls.append({"query": q["query"], "fn": "irrelevant_function", "args": {}})
        _ghi_jsonl(p2, calls)
        print(f"[stage 2] xong: {len(calls)} lời gọi ({len(thong_ke['bo_qua'])} mẫu bị loại)")

    # ── STAGE 3: sinh câu trả lời (chỉ ~18% số mẫu, đúng tỉ lệ đã chốt) ──
    p3 = _stage_path(lang, "stage3_replies")
    cached = None if force_restart else _doc_jsonl(p3)
    if cached:
        print(f"[stage 3] dùng lại {len(cached)} câu trả lời đã sinh trước đó")
        replies = cached
    else:
        buoc = max(1, round(1 / 0.18))
        chon = calls[::buoc]
        prompts, kem = [], []
        for c in chon:
            ket_qua = simulate_result(c["fn"], c["args"], seed=abs(hash(c["query"])) % 10**6)
            call_str = CallRecord(c["query"], c["fn"], c["args"], lang).call_string()
            prompts.append(build_reply_gen_prompt(c["query"], call_str, ket_qua, lang))
            kem.append({**c, "result": ket_qua})

        print(f"[stage 3] sinh câu trả lời cho {len(prompts)} mẫu ...")
        outs = sinh(prompts, temperature=0.7, max_tokens=256)

        replies = []
        for c, out in zip(kem, outs, strict=True):
            try:
                replies.append({**c, "reply": parse_reply_gen_response(out)})
            except TeacherResponseError as e:
                thong_ke["bo_qua"][f"stage3:{c['query'][:40]}"] = str(e)[:200]
        _ghi_jsonl(p3, replies)
        print(f"[stage 3] xong: {len(replies)} câu trả lời")

    # ── STAGE 4: teacher tự chấm, loại mẫu sai ───────────────────────────
    p4 = _stage_path(lang, f"train_{lang}")
    # Câu ngoài phạm vi phải dùng prompt chấm RIÊNG: hỏi "câu này có thật sự ngoài
    # phạm vi không?" thay vì "lời gọi hàm có đúng ý không" (irrelevant_function
    # không bao giờ 'trả lời' được câu hỏi — hỏi sai kiểu thì bị loại sạch).
    prompts = []
    for c in calls:
        if c["fn"] == "irrelevant_function":
            prompts.append(chat(build_verify_negative_prompt(c["query"])))
        else:
            prompts.append(chat(build_verify_prompt(c["query"], CATALOG.by_name[c["fn"]], c["args"])))
    for r in replies:
        if r["fn"] == "irrelevant_function":
            prompts.append(chat(build_verify_negative_prompt(r["query"])))
        else:
            prompts.append(
                chat(
                    build_verify_prompt(
                        r["query"], CATALOG.by_name[r["fn"]], r["args"], r["reply"], r["result"]
                    )
                )
            )

    print(f"[stage 4] chấm {len(prompts)} mẫu ...")
    outs = sinh(prompts, temperature=0.0, max_tokens=256)  # chấm điểm phải ổn định

    rows: list[dict[str, Any]] = []
    n_calls = len(calls)
    for i, out in enumerate(outs):
        try:
            if not parse_verify_response(out).valid:
                thong_ke["bo_qua"][f"stage4:#{i}"] = "teacher chấm không hợp lệ"
                continue
        except TeacherResponseError:
            pass  # không đọc được kết quả chấm -> giữ lại, để check chương trình lọc sau
        if i < n_calls:
            c = calls[i]
            rec = (
                negative_record(c["query"], lang)
                if c["fn"] == "irrelevant_function"
                else CallRecord(c["query"], c["fn"], c["args"], lang)
            )
        else:
            r = replies[i - n_calls]
            rec = ReplyRecord(r["query"], r["fn"], r["args"], r["result"], r["reply"], lang)
        rows.append(rec.to_row())

    _ghi_jsonl(p4, rows)
    bang = datasheet(rows)
    print(f"[stage 4] xong: giữ lại {len(rows)}/{len(prompts)} mẫu")
    print(json.dumps(bang, ensure_ascii=False, indent=2))
    return {"datasheet": bang, "so_mau_bi_loai": len(thong_ke["bo_qua"])}


# ── bước 2b: câu NGẦM HIỂU -> nhãn <nexa_ask> (CẦN A100 80GB — tốn tiền) ────
@app.function(
    image=image,
    gpu=TEACHER_GPU,
    volumes={VOL_PATH: volume},
    timeout=TIMEOUT_GENERATE,
)
def generate_ask(
    queries_per_fn: int = 200,
    so_negative_cham_lai: int = 0,
    lang: str = "vi",
    force_restart: bool = False,
    chi_ham: str = "",
    tien_to: str = "",
) -> dict[str, Any]:
    """Sinh bổ sung nhóm câu ngầm hiểu (PLAN.md Phase 2a bước 10). Tách khỏi generate()
    để không đụng dataset đã có. Việc gộp vào tập train (và bỏ các negative cũ đã được
    gán lại) làm ở local SAU KHI soi chất lượng — không tự gộp ở đây.

      A1. gom câu: câu ngầm hiểu mới sinh cho các hàm có `implicit_examples`.
          `so_negative_cham_lai` MẶC ĐỊNH 0 — đợt chạy thử cho thấy để teacher chấm lại
          negative cũ gây hại nhiều hơn lợi: trong 500 câu nó đổi nhãn 78 câu và phần lớn
          sai ("mở cửa hàng" -> unlock_doors, "máy lạnh nhà tôi hỏng" -> set_ac). Những câu
          đó là negative KHÓ, cần giữ nguyên. Chỉ bật lại khi muốn khảo sát thủ công.
      A2. teacher xếp mỗi câu vào 3 loại (rõ ràng / ngầm hiểu / ngoài phạm vi) + đề xuất
      A3. teacher chấm lại riêng các câu "ngầm hiểu" -> ghi mẫu <nexa_ask>

    `chi_ham`: danh sách tên hàm cách nhau bởi dấu phẩy — chỉ sinh cho các hàm này.
    `tien_to`: thêm vào đầu tên MỌI file stage (vd "v2_") để KHÔNG ghi đè dữ liệu vòng
    trước — kho vòng 1 còn cần cho thí nghiệm quy mô dữ liệu và để đối chiếu.
    """
    import random

    from gen.loc import cau_hop_le
    from gen.records import SuggestRecord, query_from_call_prompt
    from gen.teacher_prompts import (
        TeacherResponseError,
        build_ask_label_prompt,
        build_implicit_query_prompt,
        build_verify_ask_prompt,
        parse_ask_label_response,
        parse_query_gen_response,
        parse_verify_response,
    )
    from schema.loader import CATALOG

    sinh, chat = _khoi_tao_teacher()

    # ── A1: gom câu cần phân loại ────────────────────────────────────────
    p1 = _stage_path(lang, f"{tien_to}ask_stage1_queries")
    nguon = None if force_restart else _doc_jsonl(p1)
    if nguon:
        print(f"[ask 1] dùng lại {len(nguon)} câu đã gom trước đó")
    else:
        chon = {t.strip() for t in chi_ham.split(",") if t.strip()}
        ham_ngam = [f for f in CATALOG.functions if f.implicit_examples and (not chon or f.name in chon)]
        if chon and len(ham_ngam) != len(chon):
            thieu = chon - {f.name for f in ham_ngam}
            raise ValueError(f"các hàm này không có implicit_examples (hoặc sai tên): {sorted(thieu)}")
        CAU_MOI_LO = 50  # cùng lý do chia lô như stage 1 của generate()
        prompts, nhan = [], []
        for fn in ham_ngam:
            so_lo = max(1, -(-queries_per_fn // CAU_MOI_LO))
            xin_moi_lo = min(queries_per_fn, CAU_MOI_LO)
            for _ in range(so_lo):
                prompts.append(chat(build_implicit_query_prompt(fn, xin_moi_lo, lang)))
                nhan.append(fn.name)
        print(f"[ask 1] sinh câu ngầm hiểu cho {len(ham_ngam)} hàm ({len(prompts)} lô) ...")
        outs = sinh(prompts, temperature=0.95, max_tokens=4096)

        nguon, da_thay = [], set()
        for fn_name, out in zip(nhan, outs, strict=True):
            try:
                cau = parse_query_gen_response(out)
            except TeacherResponseError:
                continue
            for q in cau:
                khoa = " ".join(q.split()).lower()
                if khoa in da_thay or not cau_hop_le(q)[0]:
                    continue
                da_thay.add(khoa)
                nguon.append({"query": q, "nguon": "sinh_moi", "fn_goc": fn_name})

        # Negative cũ bị gán nhầm là có thật: "hôm nay trời nóng quá!", "Mở cửa sổ khi trời oi bức."
        cu = _doc_jsonl(_stage_path(lang, f"train_{lang}")) or []
        negatives = [
            query_from_call_prompt(r["prompt"])
            for r in cu
            if r["meta"]["fn"] == "irrelevant_function" and r["meta"]["type"] == "call"
        ]
        random.Random(42).shuffle(negatives)
        for q in negatives[:so_negative_cham_lai]:
            nguon.append({"query": q, "nguon": "negative_cu", "fn_goc": "irrelevant_function"})
        _ghi_jsonl(p1, nguon)
        print(f"[ask 1] xong: {len(nguon)} câu ({min(len(negatives), so_negative_cham_lai)} negative cũ)")

    # ── A2: teacher xếp loại + đề xuất ───────────────────────────────────
    p2 = _stage_path(lang, f"{tien_to}ask_stage2_labels")
    nhan_rows = None if force_restart else _doc_jsonl(p2)
    if nhan_rows:
        print(f"[ask 2] dùng lại {len(nhan_rows)} nhãn đã có")
    else:
        print(f"[ask 2] phân loại {len(nguon)} câu ...")
        outs = sinh(
            [chat(build_ask_label_prompt(x["query"])) for x in nguon], temperature=0.0, max_tokens=512
        )
        nhan_rows, loi = [], 0
        for x, out in zip(nguon, outs, strict=True):
            try:
                lab = parse_ask_label_response(out, x["query"])
            except TeacherResponseError:
                loi += 1
                continue
            nhan_rows.append(
                {**x, "loai": lab.loai, "calls": [[n, a] for n, a in lab.calls], "ly_do": lab.ly_do}
            )
        _ghi_jsonl(p2, nhan_rows)
        print(f"[ask 2] xong: {len(nhan_rows)} nhãn ({loi} câu không đọc được kết quả)")

    ma_tran: dict[str, dict[str, int]] = {}
    for x in nhan_rows:
        o = ma_tran.setdefault(x["nguon"], {})
        o[x["loai"]] = o.get(x["loai"], 0) + 1
    print("[ask 2] nguồn × loại:", json.dumps(ma_tran, ensure_ascii=False))

    # ── A3: chấm lại riêng các câu ngầm hiểu ─────────────────────────────
    ngam = [x for x in nhan_rows if x["loai"] == "ngam_hieu"]
    print(f"[ask 3] chấm {len(ngam)} mẫu ngầm hiểu ...")
    outs = sinh(
        [chat(build_verify_ask_prompt(x["query"], [(n, a) for n, a in x["calls"]])) for x in ngam],
        temperature=0.0,
        max_tokens=256,
    )
    rows: list[dict[str, Any]] = []
    bi_loai: list[dict[str, Any]] = []
    for x, out in zip(ngam, outs, strict=True):
        # Khác stage 4 của generate(): không đọc được kết quả chấm thì LOẠI (nhãn loại mới,
        # chưa có check chương trình nào phía sau đỡ cho).
        try:
            kq = parse_verify_response(out)
        except TeacherResponseError:
            bi_loai.append({**x, "ly_do_loai": "không đọc được kết quả chấm"})
            continue
        if kq.valid:
            rec = SuggestRecord(x["query"], tuple((n, a) for n, a in x["calls"]), lang)
            rows.append({**rec.to_row(), "nguon": x["nguon"]})
        else:
            bi_loai.append({**x, "ly_do_loai": kq.reason})

    _ghi_jsonl(_stage_path(lang, f"{tien_to}ask_train_{lang}"), rows)
    _ghi_jsonl(_stage_path(lang, f"{tien_to}ask_bi_loai"), bi_loai)

    theo_ham: dict[str, int] = {}
    for r in rows:
        theo_ham[r["meta"]["fn"]] = theo_ham.get(r["meta"]["fn"], 0) + 1
    bao_cao = {
        "nguon_x_loai": ma_tran,
        "ngam_hieu_giu_lai": len(rows),
        "ngam_hieu_bi_loai": len(bi_loai),
        "giu_lai_tu_negative_cu": sum(1 for r in rows if r["nguon"] == "negative_cu"),
        "de_xuat_2_ham": sum(1 for r in rows if len(r["meta"]["fns"]) == 2),
        "theo_ham_de_xuat_dau": dict(sorted(theo_ham.items())),
    }
    print(json.dumps(bao_cao, ensure_ascii=False, indent=2))
    return bao_cao


# ── sinh lại câu trả lời TỪ CHỐI cho nhóm ngoài phạm vi (CẦN A100 80GB) ────
@app.function(
    image=image,
    gpu=TEACHER_GPU,
    volumes={VOL_PATH: volume},
    timeout=TIMEOUT_GENERATE,
)
def sinh_lai_tu_choi(lang: str = "vi", so_mau: int = 0) -> dict[str, Any]:
    """Viết lại lượt 2 cho câu ngoài phạm vi bằng prompt từ chối riêng.

    LỖI ĐANG SỬA: 652/1128 mẫu reply của nhóm này đang TRẢ LỜI THẲNG câu hỏi ngoài phạm
    vi ("máy ảnh nào tốt nhất?" -> tư vấn máy ảnh thật) thay vì từ chối. Lượt 1 từ chối
    đúng nhưng người dùng chỉ nghe lượt 2, nên toàn bộ việc dạy từ chối thành vô nghĩa.

    Đọc câu hỏi từ `data/{lang}/train_{lang}.jsonl` (mẫu reply có fn=irrelevant_function),
    ghi mẫu mới ra `data/{lang}/tu_choi_reply.jsonl`.
    """
    from gen.records import ReplyRecord, query_from_call_prompt
    from gen.teacher_prompts import (
        TeacherResponseError,
        build_refusal_reply_prompt,
        parse_reply_gen_response,
    )

    cu = _doc_jsonl(_stage_path(lang, f"train_{lang}")) or []
    can_sinh = [
        query_from_call_prompt(r["prompt"])
        for r in cu
        if r["meta"]["type"] == "reply" and r["meta"]["fn"] == "irrelevant_function"
    ]
    if so_mau:
        can_sinh = can_sinh[:so_mau]
    if not can_sinh:
        raise RuntimeError("không tìm thấy mẫu reply nào của irrelevant_function")
    print(f"[từ chối] sinh lại {len(can_sinh)} câu đáp ...")

    sinh, chat = _khoi_tao_teacher()
    # temperature cao hơn stage 3 cũ (0.7): câu từ chối rất dễ rập khuôn một mẫu,
    # mà data toàn "Xin lỗi, tôi không hỗ trợ việc này" thì model học thành vẹt.
    outs = sinh([chat(build_refusal_reply_prompt(q, lang)) for q in can_sinh], 0.9, 128)

    rows, loi = [], 0
    for q, out in zip(can_sinh, outs, strict=True):
        try:
            tra_loi = parse_reply_gen_response(out)
        except TeacherResponseError:
            loi += 1
            continue
        rows.append(ReplyRecord(q, "irrelevant_function", {}, {"handled": True}, tra_loi, lang).to_row())
    _ghi_jsonl(_stage_path(lang, "tu_choi_reply"), rows)

    print(f"[từ chối] xong: {len(rows)} mẫu ({loi} mẫu lỗi)")
    print("\n15 mẫu đầu để soi nhanh:")
    for r in rows[:15]:
        q = r["prompt"].split("<|im_start|>user\n")[1].split("<|im_end|>")[0]
        print(f"  {q[:52]:54} -> {r['completion'].replace('<|im_end|>', '')[:80]}")
    return {"so_mau": len(rows), "loi": loi}


# ── vòng 2: sinh BỔ SUNG theo chẩn đoán vòng 1 (CẦN A100 80GB) ─────────────
@app.function(
    image=image,
    gpu=TEACHER_GPU,
    volumes={VOL_PATH: volume},
    timeout=TIMEOUT_GENERATE,
)
def sinh_vong_2(
    cau_moi_ham: int = 200,
    so_tu_choi_kho: int = 1500,
    lang: str = "vi",
    smoke_test: bool = False,
    force_restart: bool = False,
    nguong_hieu_chinh: float = 0.8,
) -> dict[str, Any]:
    """Sinh đúng hai loại dữ liệu mà vòng 1 thiếu, ghi ra file `v2_*` — không đè vòng 1.

    Chẩn đoán vòng 1 (xem gen/ra_soat_tu_vung.py, đo chỉ trên dữ liệu train):

      1. 15/24 hàm KHOÁ TỪ VỰNG — vd 267/267 câu control_window dùng "cửa sổ", 0 câu dùng
         "kính", trong khi set_defrost chiếm 100% chữ "kính". Test viết tay: control_window
         chỉ đúng 8,3%. -> sinh lại câu cho các hàm có `tu_dong_nghia`, prompt ép chia đều.

      2. Mẫu từ chối chỉ 2,1% là câu MỆNH LỆNH (mẫu gọi hàm: 37,1%) -> model học tắt
         "mệnh lệnh ⟹ gọi hàm", gọi bừa 49 câu ngoài phạm vi. -> sinh câu ra lệnh cho
         tính năng xe thật có mà mình không hỗ trợ, gắn `meta.nhom_phu = "tu_choi_kho"`
         để khâu bốc mẫu luôn giữ hết.

    Mẫu hỏi lại (việc 3) KHÔNG sinh ở đây mà dùng lại `generate_ask(chi_ham=..., tien_to="v2_")`.

    smoke_test=True: 2 hàm × 10 câu + 2 câu mỗi nhóm từ chối — để xem prompt mới có
    thật sự đa dạng từ vựng không TRƯỚC khi chi tiền chạy đủ.

    Luôn HIỆU CHỈNH GIÁM KHẢO trước: chấm 20 câu có sẵn đáp án (BO_HIEU_CHINH_TU_CHOI, cả
    hai chiều). Chạy thử lần đầu cho thấy giám khảo cũ loại oan 6/6 câu từ chối khó sát nách
    hàm thật. Chiều nào khớp dưới `nguong_hieu_chinh` thì DỪNG — mất tiền nạp teacher chứ
    không mất tiền chạy đủ ra một bộ dữ liệu hỏng.
    """
    import math

    from gen.loc import cau_hop_le, tham_so_hop_le
    from gen.ra_soat_tu_vung import la_menh_lenh
    from gen.records import CallRecord, datasheet, negative_record
    from gen.subsample import TU_CHOI_KHO
    from gen.teacher_prompts import (
        BO_HIEU_CHINH_TU_CHOI,
        TINH_NANG_KHONG_HO_TRO,
        TeacherResponseError,
        build_call_gen_prompt,
        build_hard_negative_prompt,
        build_query_gen_prompt,
        build_verify_negative_prompt,
        build_verify_prompt,
        parse_call_gen_response,
        parse_query_gen_response,
        parse_verify_response,
    )
    from schema.loader import CATALOG

    ham = [f for f in CATALOG.functions if f.tu_dong_nghia]
    nhom_tu_choi = list(TINH_NANG_KHONG_HO_TRO)
    if smoke_test:
        ham, cau_moi_ham = ham[:2], 10
        so_tu_choi_kho = 2 * len(nhom_tu_choi)
        print(f"[SMOKE TEST] {len(ham)} hàm × {cau_moi_ham} câu + {so_tu_choi_kho} câu từ chối khó")
    tt = "v2_smoke_" if smoke_test else "v2_"

    sinh, chat = _khoi_tao_teacher()
    CAU_MOI_LO = 50  # xem stage 1 của generate(): xin quá nhiều 1 lô thì JSON bị cắt cụt

    # ── V0: hiệu chỉnh giám khảo từ chối ─────────────────────────────────
    outs = sinh(
        [chat(build_verify_negative_prompt(c)) for c, _ in BO_HIEU_CHINH_TU_CHOI],
        temperature=0.0,
        max_tokens=256,
    )
    khop = {True: [0, 0], False: [0, 0]}  # đáp án -> [khớp, tổng]
    print("\n[v2 0] hiệu chỉnh giám khảo từ chối:")
    for (cau, dung), out in zip(BO_HIEU_CHINH_TU_CHOI, outs, strict=True):
        try:
            kq = parse_verify_response(out)
            ok = kq.valid == dung
            ly_do = kq.reason
        except TeacherResponseError:
            ok, ly_do = False, "không đọc được"
        khop[dung][0] += int(ok)
        khop[dung][1] += 1
        print(f"   {'✓' if ok else '✗'} [{'ngoài' if dung else 'làm được'}] {cau[:40]:42} {ly_do[:60]}")
    ty_le = {k: v[0] / v[1] for k, v in khop.items()}
    print(
        f"[v2 0] giữ đúng câu ngoài phạm vi: {khop[True][0]}/{khop[True][1]} | "
        f"loại đúng câu làm được: {khop[False][0]}/{khop[False][1]}"
    )
    if min(ty_le.values()) < nguong_hieu_chinh:
        raise RuntimeError(
            f"giám khảo lệch quá ngưỡng {nguong_hieu_chinh:.0%} ở ít nhất một chiều — sửa "
            "build_verify_negative_prompt trước, đừng sinh dữ liệu bằng giám khảo này"
        )

    # ── V1: sinh câu ──────────────────────────────────────────────────────
    p1 = _stage_path(lang, f"{tt}stage1_queries")
    queries = None if force_restart else _doc_jsonl(p1)
    if queries:
        print(f"[v2 1] dùng lại {len(queries)} câu")
    else:
        prompts, nhan = [], []
        for fn in ham:
            for _ in range(max(1, math.ceil(cau_moi_ham / CAU_MOI_LO))):
                prompts.append(chat(build_query_gen_prompt(fn, min(cau_moi_ham, CAU_MOI_LO), lang)))
                nhan.append(fn.name)
        # Chia theo TỪNG nhóm tính năng để nhóm nào cũng có mặt — gộp chung một prompt
        # thì teacher dồn vào vài nhóm dễ nghĩ (sưởi ghế, gạt nước) và bỏ quên phần còn lại.
        moi_nhom = max(1, math.ceil(so_tu_choi_kho / len(nhom_tu_choi)))
        for nhom in nhom_tu_choi:
            for _ in range(max(1, math.ceil(moi_nhom / CAU_MOI_LO))):
                prompts.append(chat(build_hard_negative_prompt(min(moi_nhom, CAU_MOI_LO), lang, nhom)))
                nhan.append("irrelevant_function")

        print(f"[v2 1] sinh câu cho {len(prompts)} lô ...")
        outs = sinh(prompts, temperature=0.95, max_tokens=4096)

        queries, da_thay = [], set()
        dem_loai = {"loc_chuong_trinh": 0, "khong_phai_menh_lenh": 0, "trung": 0, "json_hong": 0}
        for fn_name, out in zip(nhan, outs, strict=True):
            try:
                cau = parse_query_gen_response(out)
            except TeacherResponseError:
                dem_loai["json_hong"] += 1
                continue
            for q in cau:
                if not cau_hop_le(q)[0]:
                    dem_loai["loc_chuong_trinh"] += 1
                    continue
                # Mẫu từ chối khó mà không phải câu mệnh lệnh thì vô dụng — mục đích duy nhất
                # của nó là phá luật tắt "mệnh lệnh ⟹ gọi hàm".
                if fn_name == "irrelevant_function" and not la_menh_lenh(q):
                    dem_loai["khong_phai_menh_lenh"] += 1
                    continue
                khoa = (fn_name, " ".join(q.split()).lower())
                if khoa in da_thay:
                    dem_loai["trung"] += 1
                    continue
                da_thay.add(khoa)
                queries.append({"query": q, "fn": fn_name})
        _ghi_jsonl(p1, queries)
        print(f"[v2 1] xong: {len(queries)} câu | đã loại: {dem_loai}")

    # Kiểm NGAY độ phủ từ đồng nghĩa — đây là thứ cả vòng 2 cần đạt. Nếu một cách gọi vẫn
    # gần 0% thì prompt chưa ăn thua, dừng lại sửa prompt chứ đừng train.
    print("\n[v2 1] độ phủ từ đồng nghĩa (mỗi cách gọi xuất hiện ở bao nhiêu % câu):")
    for fn in ham:
        cua_ham = [q["query"].lower() for q in queries if q["fn"] == fn.name]
        if not cua_ham:
            print(f"   {fn.name:20} KHÔNG CÓ CÂU NÀO")
            continue
        phu = "  ".join(
            f"{t}={sum(1 for c in cua_ham if t.lower() in c) * 100 // len(cua_ham)}%"
            for t in fn.tu_dong_nghia
        )
        print(f"   {fn.name:20} ({len(cua_ham):>3} câu)  {phu}")

    # ── V2: trích tham số cho câu dương ──────────────────────────────────
    p2 = _stage_path(lang, f"{tt}stage2_calls")
    calls = None if force_restart else _doc_jsonl(p2)
    if calls:
        print(f"[v2 2] dùng lại {len(calls)} lời gọi")
    else:
        duong = [q for q in queries if q["fn"] != "irrelevant_function"]
        outs = sinh(
            [chat(build_call_gen_prompt(q["query"], CATALOG.by_name[q["fn"]])) for q in duong],
            temperature=0.1,
            max_tokens=512,
        )
        calls, loi = [], 0
        for q, out in zip(duong, outs, strict=True):
            try:
                args = parse_call_gen_response(out, CATALOG.by_name[q["fn"]])
            except TeacherResponseError:
                loi += 1
                continue
            if not tham_so_hop_le(args)[0]:
                loi += 1
                continue
            calls.append({"query": q["query"], "fn": q["fn"], "args": args})
        calls += [
            {"query": q["query"], "fn": "irrelevant_function", "args": {}}
            for q in queries
            if q["fn"] == "irrelevant_function"
        ]
        _ghi_jsonl(p2, calls)
        print(f"[v2 2] xong: {len(calls)} lời gọi ({loi} câu bị loại)")

    # ── V3: teacher chấm ─────────────────────────────────────────────────
    # Câu từ chối khó PHẢI qua bước này: teacher được dặn tránh các tính năng mình hỗ trợ,
    # nhưng vẫn có thể lỡ viết "bật điều hoà lên" — để lọt vào thì dạy model từ chối
    # một lệnh hợp lệ, tức là tạo ra đúng lỗi từ chối nhầm mà mình đang canh.
    prompts = [
        chat(
            build_verify_negative_prompt(c["query"])
            if c["fn"] == "irrelevant_function"
            else build_verify_prompt(c["query"], CATALOG.by_name[c["fn"]], c["args"])
        )
        for c in calls
    ]
    print(f"[v2 3] chấm {len(prompts)} mẫu ...")
    outs = sinh(prompts, temperature=0.0, max_tokens=256)

    rows: list[dict[str, Any]] = []
    bi_loai = {"duong": 0, "tu_choi_kho": 0}
    for c, out in zip(calls, outs, strict=True):
        am = c["fn"] == "irrelevant_function"
        try:
            hop_le = parse_verify_response(out).valid
        except TeacherResponseError:
            hop_le = False  # nhãn mới, chưa có lớp lọc nào phía sau đỡ -> không chắc thì loại
        if not hop_le:
            bi_loai["tu_choi_kho" if am else "duong"] += 1
            continue
        if am:
            row = negative_record(c["query"], lang).to_row()
            row["meta"]["nhom_phu"] = TU_CHOI_KHO
        else:
            row = CallRecord(c["query"], c["fn"], c["args"], lang).to_row()
        rows.append(row)

    _ghi_jsonl(_stage_path(lang, f"{tt}bo_sung_{lang}"), rows)
    bang = datasheet(rows)
    n_kho = sum(1 for r in rows if r["meta"].get("nhom_phu") == TU_CHOI_KHO)
    print(f"[v2 3] xong: giữ {len(rows)} mẫu ({n_kho} từ chối khó) | bị loại: {bi_loai}")
    print(json.dumps(bang, ensure_ascii=False, indent=2))

    print("\n12 câu từ chối khó đầu tiên để soi:")
    for r in [r for r in rows if r["meta"].get("nhom_phu") == TU_CHOI_KHO][:12]:
        print(f"   {r['prompt'].split('<|im_start|>user')[1].split('<|im_end|>')[0].strip()[:70]}")
    return {"so_mau": len(rows), "tu_choi_kho": n_kho, "bi_loai": bi_loai, "datasheet": bang}


# ── giám khảo TỰ CHỌN hàm cho câu than phiền (CẦN A100 80GB) ───────────────
@app.function(
    image=image,
    gpu=TEACHER_GPU,
    volumes={VOL_PATH: volume},
    timeout=TIMEOUT_GENERATE,
)
def chon_de_xuat(
    giam_khao: str = "qwen",
    cac_file: str = "v1_ask_sach,v2_ask_train_vi",
    lang: str = "vi",
    tien_to: str = "",
) -> dict[str, Any]:
    """Một giám khảo tự chọn ≤2 hàm cho MỖI câu, KHÔNG xem nhãn. Ghi `<tien_to>chon_<giam_khao>.jsonl`.

    `tien_to` (vd "v3_") để không ghi đè lựa chọn của đợt chấm trước.

    Quy trình mới chấm mẫu hỏi lại (PLAN.md E0b): code lọc câu ra lệnh + luật -> Qwen và
    Llama TỰ CHỌN độc lập -> code giữ mẫu khi hàm đề xuất chính nằm trong lựa chọn của CẢ
    HAI. Mỗi lần chạy nạp đúng MỘT giám khảo vì hai model 70B không vừa chung một A100 80GB;
    bước quyết định cuối chạy ở máy (`python -m gen.cham_ask`), miễn phí.

    Luôn chấm cả các câu của bộ hiệu chỉnh `BO_HIEU_CHINH_HOI_LAI` để đo kappa ở bước cuối.
    """
    from vllm import LLM, SamplingParams

    from gen.records import query_from_call_prompt
    from gen.teacher_prompts import BO_HIEU_CHINH_HOI_LAI, build_chon_de_xuat_prompt

    if giam_khao not in ("qwen", "llama"):
        raise ValueError("giam_khao phải là 'qwen' hoặc 'llama'")

    cau: list[str] = []
    da_co: set[str] = set()
    for c, _, _ in BO_HIEU_CHINH_HOI_LAI:
        if c not in da_co:
            da_co.add(c)
            cau.append(c)
    for ten in [t.strip() for t in cac_file.split(",") if t.strip()]:
        for r in _doc_jsonl(_stage_path(lang, ten)) or []:
            c = query_from_call_prompt(r["prompt"])
            if c not in da_co:
                da_co.add(c)
                cau.append(c)
    print(f"[chọn/{giam_khao}] {len(cau)} câu khác nhau")

    thong_so = SamplingParams(temperature=0.0, max_tokens=200)
    if giam_khao == "qwen":
        sinh, chat = _khoi_tao_teacher()
        outs = sinh([chat(build_chon_de_xuat_prompt(c)) for c in cau], temperature=0.0, max_tokens=200)
    else:
        # Giám khảo khác họ: để vLLM áp đúng chat template của Llama (xem tham_dinh_cheo).
        llm = LLM(model=JUDGE_DIR, gpu_memory_utilization=0.90, max_model_len=4096, enforce_eager=True)
        res = llm.chat([[{"role": "user", "content": build_chon_de_xuat_prompt(c)}] for c in cau], thong_so)
        outs = [o.outputs[0].text for o in res]

    from gen.cham_ask import doc_lua_chon

    rows, loi = [], 0
    for c, out in zip(cau, outs, strict=True):
        try:
            ham = doc_lua_chon(out)
        except (ValueError, json.JSONDecodeError):
            loi += 1
            ham = []  # không đọc được -> coi như không chọn gì -> mẫu đó sẽ bị LOẠI (thận trọng)
        rows.append({"cau": c, "ham": ham})
    _ghi_jsonl(_stage_path(lang, f"{tien_to}chon_{giam_khao}"), rows)

    dem: dict[str, int] = {}
    for r in rows:
        for h in r["ham"][:1]:
            dem[h] = dem.get(h, 0) + 1
    print(f"[chọn/{giam_khao}] xong: {len(rows)} câu, {loi} câu không đọc được")
    print(f"   hàm chọn đầu tiên: {dict(sorted(dem.items(), key=lambda x: -x[1]))}")
    print("   bộ hiệu chỉnh:")
    for r in rows[: len({c for c, _, _ in BO_HIEU_CHINH_HOI_LAI})]:
        print(f"     {r['cau'][:44]:46} -> {r['ham']}")
    return {"so_cau": len(rows), "loi": loi}


# ── chấm lại mẫu hỏi lại bằng giám khảo đã sửa (CẦN A100 80GB) ─────────────
@app.function(
    image=image,
    gpu=TEACHER_GPU,
    volumes={VOL_PATH: volume},
    timeout=TIMEOUT_GENERATE,
)
def cham_lai_ask(
    cac_file: str = "v1_ask_sach,v2_ask_train_vi",
    lang: str = "vi",
    nguong_hieu_chinh: float = 0.8,
) -> dict[str, Any]:
    """Chấm lại mẫu `<nexa_ask>` bằng `build_verify_ask_prompt` bản sửa.

    Giám khảo cũ giữ 517/518 mẫu vòng 2, kể cả "chỗ đỗ không an toàn" -> đèn khẩn cấp. Mẫu
    vòng 1 cũng đi qua ĐÚNG giám khảo đó nên chấm lại cả hai cho nhất quán.

    `cac_file`: tên file (không đuôi) trong data/<lang>/, cách nhau dấu phẩy. Ghi ra
    `<ten>_cham_lai.jsonl` (giữ) và `<ten>_cham_lai_loai.jsonl` (bị loại, kèm lý do để soi
    tay — theo nguyên tắc Huawei §A.7, giám khảo cũng sai nên phải đọc lại phần bị loại).
    """
    from gen.records import query_from_call_prompt
    from gen.teacher_prompts import (
        BO_HIEU_CHINH_HOI_LAI,
        TeacherResponseError,
        build_verify_ask_prompt,
        parse_verify_response,
    )
    from schema.call_format import parse_calls

    sinh, chat = _khoi_tao_teacher()

    # ── hiệu chỉnh: lệch quá ngưỡng ở chiều nào thì dừng ──────────────────
    outs = sinh(
        [chat(build_verify_ask_prompt(c, de_xuat)) for c, de_xuat, _ in BO_HIEU_CHINH_HOI_LAI],
        temperature=0.0,
        max_tokens=256,
    )
    khop = {True: [0, 0], False: [0, 0]}
    print("\n[ask lại 0] hiệu chỉnh giám khảo hỏi lại:")
    for (cau, de_xuat, dung), out in zip(BO_HIEU_CHINH_HOI_LAI, outs, strict=True):
        try:
            kq = parse_verify_response(out)
            ok, ly_do = kq.valid == dung, kq.reason
        except TeacherResponseError:
            ok, ly_do = False, "không đọc được"
        khop[dung][0] += int(ok)
        khop[dung][1] += 1
        ten = "+".join(n for n, _ in de_xuat)
        print(f"   {'✓' if ok else '✗'} [{'giữ' if dung else 'loại'}] {cau[:34]:36} -> {ten:18} {ly_do[:50]}")
    print(
        f"[ask lại 0] giữ đúng {khop[True][0]}/{khop[True][1]} | loại đúng {khop[False][0]}/{khop[False][1]}"
    )
    if min(v[0] / v[1] for v in khop.values()) < nguong_hieu_chinh:
        raise RuntimeError(f"giám khảo hỏi lại lệch quá ngưỡng {nguong_hieu_chinh:.0%} — sửa prompt trước")

    bao_cao: dict[str, Any] = {"hieu_chinh": {str(k): v for k, v in khop.items()}}
    for ten in [t.strip() for t in cac_file.split(",") if t.strip()]:
        rows = _doc_jsonl(_stage_path(lang, ten)) or []
        if not rows:
            print(f"[ask lại] BỎ QUA {ten}: không có file / rỗng")
            continue
        vao = []
        for r in rows:
            vao.append(
                (
                    r,
                    query_from_call_prompt(r["prompt"]),
                    [(c.name, dict(c.args)) for c in parse_calls(r["completion"].replace("<|im_end|>", ""))],
                )
            )
        outs = sinh([chat(build_verify_ask_prompt(q, c)) for _, q, c in vao], temperature=0.0, max_tokens=256)
        giu, loai = [], []
        for (r, _q, _c), out in zip(vao, outs, strict=True):
            try:
                kq = parse_verify_response(out)
            except TeacherResponseError:
                loai.append({**r, "ly_do_loai": "không đọc được kết quả chấm"})
                continue
            (giu if kq.valid else loai).append(r if kq.valid else {**r, "ly_do_loai": kq.reason})
        _ghi_jsonl(_stage_path(lang, f"{ten}_cham_lai"), giu)
        _ghi_jsonl(_stage_path(lang, f"{ten}_cham_lai_loai"), loai)
        dem_giu: dict[str, int] = {}
        for r in giu:
            dem_giu[r["meta"]["fn"]] = dem_giu.get(r["meta"]["fn"], 0) + 1
        print(f"\n[ask lại] {ten}: giữ {len(giu)}/{len(rows)}")
        print(f"   theo hàm: {dict(sorted(dem_giu.items(), key=lambda x: -x[1]))}")
        bao_cao[ten] = {"vao": len(rows), "giu": len(giu), "theo_ham": dem_giu}
    return bao_cao


# ── thẩm định chéo bằng model KHÁC HỌ (CẦN A100 80GB) ──────────────────────
@app.function(
    image=image,
    gpu=TEACHER_GPU,
    volumes={VOL_PATH: volume},
    timeout=TIMEOUT_GENERATE,
)
def tham_dinh_cheo(so_mau: int = 0, lang: str = "vi") -> dict[str, Any]:
    """Cho model khác họ chấm lại mẫu train do Qwen sinh + tự chấm.

    Đọc mẫu ở `data/{lang}/tham_dinh_mau.jsonl` (đẩy lên bằng `modal volume put`),
    dùng ĐÚNG bộ prompt chấm đã dùng cho Qwen, rồi ghi kết quả từng mẫu ra
    `data/{lang}/tham_dinh_ket_qua.jsonl`.

    `so_mau=0` nghĩa là chấm hết; đặt số nhỏ (vd 30) để chạy thử trước cho rẻ.

    KHÔNG tự động xoá mẫu nào — chỉ báo cáo. Quyết định sửa data là việc của người,
    sau khi đọc tay các mẫu bị chê (xem lý do trong PLAN.md / paper Huawei §A.7).
    """
    from vllm import LLM, SamplingParams

    from gen.records import query_from_call_prompt, result_from_reply_prompt
    from gen.teacher_prompts import (
        TeacherResponseError,
        build_verify_ask_prompt,
        build_verify_negative_prompt,
        build_verify_prompt,
        parse_verify_response,
    )
    from schema.call_format import parse_calls
    from schema.loader import CATALOG

    mau = _doc_jsonl(_stage_path(lang, "tham_dinh_mau")) or []
    if not mau:
        raise RuntimeError(
            "chưa có mẫu để chấm — chạy trước:\n"
            "  python -m eval.boc_mau_tham_dinh --input data/processed/final/train_vi.jsonl "
            "--n 500 --output data/processed/tham_dinh/mau.jsonl\n"
            "  modal volume put nexa-drive data/processed/tham_dinh/mau.jsonl "
            f"data/{lang}/tham_dinh_mau.jsonl"
        )
    if so_mau:
        mau = mau[:so_mau]
    print(f"[thẩm định] {len(mau)} mẫu, giám khảo = {JUDGE_MODEL}")

    llm = LLM(model=JUDGE_DIR, gpu_memory_utilization=0.90, max_model_len=4096, enforce_eager=True)

    # Giám khảo khác họ -> KHÔNG bọc ChatML thủ công như teacher Qwen, phải để vLLM
    # áp đúng chat template của chính model đó, không thì output lệch định dạng.
    hoi_thoai, kem = [], []
    for r in mau:
        query = query_from_call_prompt(r["prompt"])
        loai, fn_name = r["meta"]["type"], r["meta"]["fn"]
        if fn_name == "irrelevant_function":
            noi_dung = build_verify_negative_prompt(query)
        elif loai == "ask":
            calls = [(c.name, c.args) for c in parse_calls(r["completion"])]
            noi_dung = build_verify_ask_prompt(query, calls)
        elif loai == "reply":
            call = parse_calls(r["prompt"].split("<|im_start|>assistant\n")[1])[0]
            noi_dung = build_verify_prompt(
                query,
                CATALOG.by_name[fn_name],
                call.args,
                reply=r["completion"].split("<|im_end|>")[0].strip(),
                result=result_from_reply_prompt(r["prompt"]),
            )
        else:
            call = parse_calls(r["completion"])[0]
            noi_dung = build_verify_prompt(query, CATALOG.by_name[fn_name], call.args)
        hoi_thoai.append([{"role": "user", "content": noi_dung}])
        kem.append({"query": query, "type": loai, "fn": fn_name})

    outs = llm.chat(hoi_thoai, SamplingParams(temperature=0.0, max_tokens=256))

    rows, loi_doc = [], 0
    for x, o in zip(kem, outs, strict=True):
        text = o.outputs[0].text
        try:
            kq = parse_verify_response(text)
            rows.append({**x, "giam_khao_dong_y": kq.valid, "ly_do": kq.reason})
        except TeacherResponseError:
            loi_doc += 1
            rows.append({**x, "giam_khao_dong_y": None, "ly_do": text[:200]})
    _ghi_jsonl(_stage_path(lang, "tham_dinh_ket_qua"), rows)

    def ty_le(ds: list[dict[str, Any]]) -> float | None:
        doc_duoc = [r for r in ds if r["giam_khao_dong_y"] is not None]
        if not doc_duoc:
            return None
        return round(sum(r["giam_khao_dong_y"] for r in doc_duoc) / len(doc_duoc) * 100, 1)

    theo_loai: dict[str, Any] = {}
    for loai in ("call", "ask", "reply"):
        ds = [r for r in rows if r["type"] == loai and r["fn"] != "irrelevant_function"]
        if ds:
            theo_loai[loai] = {"so_mau": len(ds), "dong_y_%": ty_le(ds)}
    neg = [r for r in rows if r["fn"] == "irrelevant_function"]
    if neg:
        theo_loai["ngoai_pham_vi"] = {"so_mau": len(neg), "dong_y_%": ty_le(neg)}

    bao_cao = {
        "giam_khao": JUDGE_MODEL,
        "so_mau": len(rows),
        "dong_thuan_chung_%": ty_le(rows),
        "theo_loai": theo_loai,
        "khong_doc_duoc_ket_qua_cham": loi_doc,
    }
    print(json.dumps(bao_cao, ensure_ascii=False, indent=2))
    print("\n[thẩm định] 15 mẫu bị giám khảo CHÊ (đọc tay để quyết định, KHÔNG tự xoá):")
    for r in [x for x in rows if x["giam_khao_dong_y"] is False][:15]:
        print(f"  [{r['type']:5}] {r['fn']:20} {r['query'][:60]}\n        -> {r['ly_do'][:110]}")
    return bao_cao


@app.function(image=image, gpu=TEACHER_GPU, volumes={VOL_PATH: volume}, timeout=60 * 60)
def tham_dinh_v6(
    so_moi_khoi: int = 125, tien_to: str = "v6_", lang: str = "vi", seed: int = 2071
) -> dict[str, Any]:
    """Llama-3.3-70B (khác họ) chấm lại mẫu v6 ĐÃ GIỮ — bốc ngẫu nhiên `so_moi_khoi` mẫu / khối, cùng prompt
    giám khảo
    Qwen đã dùng. CHỈ báo cáo (như `tham_dinh_cheo`): không tự xoá mẫu; mẫu bị chê đọc tay rồi người quyết."""
    import random

    from vllm import LLM, SamplingParams

    from gen.teacher_prompts import TeacherResponseError, build_verify_prompt, parse_verify_response
    from gen.v6 import build_verify_noi_tiep_prompt
    from schema.call_format import render_readable
    from schema.loader import CATALOG

    rng = random.Random(seed)
    mau: list[dict[str, Any]] = []
    for khoi in ("k1_da_dang", "k2_moi", "k3_doi_xung", "k4_hai_luot"):
        giu = [m for m in (_doc_jsonl(_stage_path(lang, f"{tien_to}{khoi}")) or []) if m.get("giu")]
        mau += [{**m, "khoi_ten": khoi} for m in rng.sample(giu, min(so_moi_khoi, len(giu)))]
    hoi = []
    for m in mau:
        if m["khoi"] == "K4":
            goi_1, goi_2 = render_readable(m["fn_1"], m["args_1"]), render_readable(m["fn"], m["args"])
            noi_dung = build_verify_noi_tiep_prompt(m["cau_1"], goi_1, m["query"], goi_2)
        else:
            noi_dung = build_verify_prompt(m["query"], CATALOG.by_name[m["fn"]], m["args"])
        hoi.append([{"role": "user", "content": noi_dung}])
    llm = LLM(model=JUDGE_DIR, gpu_memory_utilization=0.90, max_model_len=4096, enforce_eager=True)
    outs = llm.chat(hoi, SamplingParams(temperature=0.0, max_tokens=256))
    for m, o in zip(mau, outs, strict=True):
        try:
            kq = parse_verify_response(o.outputs[0].text)
            m.update(llama_dong_y=kq.valid, llama_ly_do=kq.reason)
        except TeacherResponseError:
            m.update(llama_dong_y=None, llama_ly_do=o.outputs[0].text[:200])
    _ghi_jsonl(_stage_path(lang, f"{tien_to}tham_dinh"), mau)
    bao_cao: dict[str, Any] = {}
    for khoi in sorted({m["khoi_ten"] for m in mau}):
        ds = [m for m in mau if m["khoi_ten"] == khoi and m["llama_dong_y"] is not None]
        dong_y = round(100 * sum(m["llama_dong_y"] for m in ds) / max(1, len(ds)), 1)
        bao_cao[khoi] = {"so_mau": len(ds), "dong_y_%": dong_y}
    print(json.dumps(bao_cao, ensure_ascii=False, indent=2))
    return bao_cao


# ── tiện ích: xem tiến độ đã có gì trên Volume ─────────────────────────────
@app.function(
    image=image,
    gpu=TEACHER_GPU,
    volumes={VOL_PATH: volume},
    timeout=60 * 60,
)
def thu_tu_choi_hoan_canh(
    so_lo_xe: int = 16,
    so_lo_ngoai_xe: int = 5,
    cau_moi_lo: int = 3,
    seed: int = 2031,
    lang: str = "vi",
    tien_to: str = "v4_thu2_",
    nguong_hieu_chinh: float = 0.8,
) -> dict[str, Any]:
    """CHẠY THỬ prompt câu từ chối kể hoàn cảnh (vòng 4) — vài chục câu để người dùng xem trước.

    Mỗi lô bốc ngẫu nhiên một bộ (việc, vai người nói, bẫy từ khoá) khác nhau. Lọc code (câu hợp
    lệ, đủ dài để có hoàn cảnh, không trùng, không cùng khuôn) rồi giám khảo từ chối chấm — GIỮ
    cả câu bị loại kèm kết quả chấm, để đọc tay xem giám khảo loại đúng hay oan.

    Lần thử 1 (8 câu/lô): trong lô đóng khuôn nặng -> lần 2: 3 câu/lô, nhiều lô hơn, lọc cùng khuôn.
    """
    import random

    from gen.loc import bo_cung_khuon, cau_hop_le, co_lenh_ham_that, co_viec_lam_duoc_an, lai_tieng_anh
    from gen.teacher_prompts import (
        BAY_TU_KHOA,
        BO_HIEU_CHINH_TU_CHOI,
        TINH_NANG_KHONG_HO_TRO,
        VAI_NGUOI_NOI,
        VIEC_NGOAI_XE,
        TeacherResponseError,
        build_contextual_negative_prompt,
        build_verify_negative_prompt,
        parse_query_gen_response,
        parse_verify_response,
    )

    rng = random.Random(seed)

    def boc_deu(ds: list[str], k: int) -> list[str]:
        """Bốc k việc, phủ đều danh sách trước khi lặp lại (tránh dồn vào vài việc)."""
        ra: list[str] = []
        while len(ra) < k:
            ra += rng.sample(ds, min(len(ds), k - len(ra)))
        return ra

    bo = [("xe", v) for v in boc_deu(TINH_NANG_KHONG_HO_TRO, so_lo_xe)]
    bo += [("ngoai_xe", v) for v in boc_deu(VIEC_NGOAI_XE, so_lo_ngoai_xe)]
    bo = [(loai, viec, rng.choice(VAI_NGUOI_NOI), rng.choice(BAY_TU_KHOA)) for loai, viec in bo]

    sinh, chat = _khoi_tao_teacher()

    # hiệu chỉnh giám khảo — như sinh_vong_2
    outs = sinh([chat(build_verify_negative_prompt(c)) for c, _ in BO_HIEU_CHINH_TU_CHOI], 0.0, 256)
    chieu = {True: [0, 0], False: [0, 0]}  # đáp án -> [khớp, tổng]
    for (_, dap_an), o in zip(BO_HIEU_CHINH_TU_CHOI, outs, strict=True):
        try:
            chieu[dap_an][0] += int(parse_verify_response(o).valid == dap_an)
        except TeacherResponseError:
            pass
        chieu[dap_an][1] += 1
    khop = chieu[True][0] + chieu[False][0]
    print(
        f"[hiệu chỉnh] giám khảo khớp {khop}/{len(BO_HIEU_CHINH_TU_CHOI)} "
        f"(giữ đúng {chieu[True][0]}/{chieu[True][1]}, loại đúng {chieu[False][0]}/{chieu[False][1]})"
    )
    # Cổng như sinh_vong_2: chiều nào lệch quá ngưỡng thì DỪNG trước khi sinh
    # (mất tiền nạp model, không mất cả lô)
    if min(v[0] / v[1] for v in chieu.values()) < nguong_hieu_chinh:
        raise RuntimeError(f"giám khảo lệch quá ngưỡng {nguong_hieu_chinh:.0%} — dừng trước khi sinh")

    prompts = [
        chat(build_contextual_negative_prompt(cau_moi_lo, viec, vai, bay, lang)) for _, viec, vai, bay in bo
    ]
    outs = sinh(prompts, temperature=0.95, max_tokens=2048)
    cau: list[dict[str, Any]] = []
    da_thay: set[str] = set()
    dem = dict.fromkeys(
        ("json_hong", "loc_code", "qua_ngan", "trung", "ghep_lenh_that", "viec_lam_duoc_an", "cung_khuon"), 0
    )
    for (loai, viec, vai, bay), out in zip(bo, outs, strict=True):
        try:
            ds = parse_query_gen_response(out)
        except TeacherResponseError:
            dem["json_hong"] += 1
            continue
        for q in ds:
            if not cau_hop_le(q)[0] or lai_tieng_anh(q):
                dem["loc_code"] += 1
                continue
            if len(q.split()) < 8:
                dem["qua_ngan"] += 1
                continue
            if co_lenh_ham_that(q):  # câu ghép có lệnh việc làm được -> nhãn từ chối sai một nửa
                dem["ghep_lenh_that"] += 1
                continue
            if co_viec_lam_duoc_an(q):  # xem thời tiết / tìm chỗ gần đây: tác tử LÀM ĐƯỢC
                dem["viec_lam_duoc_an"] += 1
                continue
            k = " ".join(q.split()).lower()
            if k in da_thay:
                dem["trung"] += 1
                continue
            da_thay.add(k)
            cau.append({"query": q, "loai": loai, "viec": viec, "vai": vai, "bay": bay})
    giu_khuon = bo_cung_khuon([c["query"] for c in cau])
    dem["cung_khuon"] = len(cau) - len(giu_khuon)
    cau = [cau[i] for i in giu_khuon]

    outs = sinh([chat(build_verify_negative_prompt(c["query"])) for c in cau], 0.0, 256)
    for c, o in zip(cau, outs, strict=True):
        try:
            kq = parse_verify_response(o)
            c["giu"], c["ly_do_giam_khao"] = kq.valid, kq.reason
        except TeacherResponseError:
            c["giu"], c["ly_do_giam_khao"] = False, "không đọc được"
    _ghi_jsonl(_stage_path(lang, f"{tien_to}tu_choi_hc"), cau)
    giu = sum(c["giu"] for c in cau)
    print(f"[thử] {len(cau)} câu sau lọc code | giám khảo giữ {giu} | đã loại: {dem}")
    return {"so_cau": len(cau), "giu": giu, "loai_code": dem, "hieu_chinh": khop}


@app.function(
    image=image,
    gpu=TEACHER_GPU,
    volumes={VOL_PATH: volume},
    timeout=2 * 60 * 60,
)
def sinh_goi_ham_hoan_canh(
    so_lo_moi_ham: int = 50,
    cau_moi_lo: int = 3,
    seed: int = 2050,
    lang: str = "vi",
    tien_to: str = "v4_",
) -> dict[str, Any]:
    """Câu GỌI HÀM kể hoàn cảnh — cặp đối chứng của câu từ chối kể hoàn cảnh (PLAN.md mục P, kết quả vòng 4).

    Cùng khung (vai, bẫy, hoàn cảnh hợp lý), lời nhờ là việc LÀM ĐƯỢC, chia đều 24 hàm. Đi qua đúng quy
    trình câu dương: lọc code (câu hợp lệ, không lai tiếng Anh, đủ dài, CÓ LỜI NHỜ, không cùng khuôn) ->
    teacher trích tham số -> giám khảo kiểm. Giữ cả câu bị loại kèm lý do để đọc tay.
    """
    import random

    from gen.loc import bo_cung_khuon, cau_hop_le, co_loi_nho, lai_tieng_anh, tham_so_hop_le
    from gen.teacher_prompts import (
        BAY_TU_KHOA,
        VAI_NGUOI_NOI,
        TeacherResponseError,
        build_call_gen_prompt,
        build_contextual_positive_prompt,
        build_verify_prompt,
        parse_call_gen_response,
        parse_query_gen_response,
        parse_verify_response,
    )
    from schema.loader import CATALOG

    rng = random.Random(seed)
    ham = [f for f in CATALOG.functions if f.name != "irrelevant_function"]
    bo = [(f, rng.choice(VAI_NGUOI_NOI), rng.choice(BAY_TU_KHOA)) for f in ham for _ in range(so_lo_moi_ham)]

    sinh, chat = _khoi_tao_teacher()
    prompts = [chat(build_contextual_positive_prompt(cau_moi_lo, f, vai, bay, lang)) for f, vai, bay in bo]
    outs = sinh(prompts, temperature=0.95, max_tokens=2048)

    ly_do_loai = ("json_hong", "loc_code", "qua_ngan", "khong_loi_nho", "trung", "cung_khuon")
    dem = dict.fromkeys((*ly_do_loai, "tham_so_hong", "giam_khao_loai"), 0)
    cau: list[dict[str, Any]] = []
    da_thay: set[str] = set()
    for (f, vai, bay), out in zip(bo, outs, strict=True):
        try:
            ds = parse_query_gen_response(out)
        except TeacherResponseError:
            dem["json_hong"] += 1
            continue
        for q in ds:
            if not cau_hop_le(q)[0] or lai_tieng_anh(q):
                dem["loc_code"] += 1
                continue
            if len(q.split()) < 8:
                dem["qua_ngan"] += 1
                continue
            if not co_loi_nho(q):  # chỉ than thở -> là câu HỎI LẠI, không phải câu gọi hàm
                dem["khong_loi_nho"] += 1
                continue
            k = " ".join(q.split()).lower()
            if k in da_thay:
                dem["trung"] += 1
                continue
            da_thay.add(k)
            cau.append({"query": q, "fn": f.name, "vai": vai, "bay": bay})
    giu_khuon = bo_cung_khuon([c["query"] for c in cau])
    dem["cung_khuon"] = len(cau) - len(giu_khuon)
    cau = [cau[i] for i in giu_khuon]

    # trích tham số (biết trước hàm) — như câu dương của mọi vòng
    outs = sinh([chat(build_call_gen_prompt(c["query"], CATALOG.by_name[c["fn"]])) for c in cau], 0.1, 512)
    co_tham_so: list[dict[str, Any]] = []
    for c, out in zip(cau, outs, strict=True):
        try:
            args = parse_call_gen_response(out, CATALOG.by_name[c["fn"]])
        except TeacherResponseError:
            dem["tham_so_hong"] += 1
            continue
        if not tham_so_hop_le(args)[0]:
            dem["tham_so_hong"] += 1
            continue
        co_tham_so.append({**c, "args": args})

    prompts = [chat(build_verify_prompt(c["query"], CATALOG.by_name[c["fn"]], c["args"])) for c in co_tham_so]
    outs = sinh(prompts, 0.0, 256)
    for c, out in zip(co_tham_so, outs, strict=True):
        try:
            kq = parse_verify_response(out)
            c["giu"], c["ly_do_giam_khao"] = kq.valid, kq.reason
        except TeacherResponseError:
            c["giu"], c["ly_do_giam_khao"] = False, "không đọc được"
        dem["giam_khao_loai"] += int(not c["giu"])
    _ghi_jsonl(_stage_path(lang, f"{tien_to}goi_ham_hc"), co_tham_so)
    giu = sum(c["giu"] for c in co_tham_so)
    print(f"[gọi hàm hoàn cảnh] {len(bo)} lô -> {len(co_tham_so)} câu có tham số -> giữ {giu}")
    print(f"[gọi hàm hoàn cảnh] đã loại: {dem}")
    return {"so_lo": len(bo), "co_tham_so": len(co_tham_so), "giu": giu, "loai": dem}


def _gan_nhan_hoi_lai(
    sinh: Any, chat: Any, cau: list[dict[str, Any]], lang: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Teacher gán nhãn 3 loại KHÔNG biết hàm đích; giữ câu "ngam_hieu" có chứa hàm đích (`fn_dich`).

    Dùng chung cho câu mới sinh và câu cứu lại sau lọc cùng khuôn — để hai nguồn đi ĐÚNG một quy trình.
    Trả về (mẫu <nexa_ask> giữ lại, chi tiết mọi câu kèm lý do, đếm theo loại)."""
    from gen.records import SuggestRecord
    from gen.teacher_prompts import TeacherResponseError, build_ask_label_prompt, parse_ask_label_response

    outs = sinh([chat(build_ask_label_prompt(c["query"])) for c in cau], 0.0, 512)
    rows: list[dict[str, Any]] = []
    chi_tiet: list[dict[str, Any]] = []
    loai_dem: dict[str, int] = {}
    for c, out in zip(cau, outs, strict=True):
        try:
            lab = parse_ask_label_response(out, c["query"])
        except TeacherResponseError:
            chi_tiet.append({**c, "loai_vi": "nhan_hong"})
            loai_dem["nhan_hong"] = loai_dem.get("nhan_hong", 0) + 1
            continue
        loai_dem[lab.loai] = loai_dem.get(lab.loai, 0) + 1
        ten = [n for n, _ in lab.calls]
        if lab.loai != "ngam_hieu":
            chi_tiet.append({**c, "loai_vi": f"teacher_xep_{lab.loai}", "calls": ten, "ly_do": lab.ly_do})
            continue
        if c["fn_dich"] not in ten:
            loai_dem["lech_ham_dich"] = loai_dem.get("lech_ham_dich", 0) + 1
            chi_tiet.append({**c, "loai_vi": "lech_ham_dich", "calls": ten, "ly_do": lab.ly_do})
            continue
        rec = SuggestRecord(c["query"], tuple(lab.calls), lang)
        rows.append({**rec.to_row(), "fn_dich": c["fn_dich"]})
        chi_tiet.append({**c, "loai_vi": "", "calls": ten, "ly_do": lab.ly_do})
    return rows, chi_tiet, loai_dem


@app.function(image=image, gpu=TEACHER_GPU, volumes={VOL_PATH: volume}, timeout=2 * 60 * 60)
def gan_nhan_hoi_lai(
    ten_vao: str = "v5_cuu_cung_khuon", tien_to: str = "v5cuu_", lang: str = "vi"
) -> dict[str, Any]:
    """Gán nhãn câu cứu lại sau lọc cùng khuôn (`python -m gen.dung_vong_5 --cuu ...`, upload lên
    data/<lang>/<ten_vao>.jsonl) — đúng bước gán nhãn của `sinh_hoi_lai_hoan_canh`."""
    cau = _doc_jsonl(_stage_path(lang, ten_vao)) or []
    sinh, chat = _khoi_tao_teacher()
    rows, chi_tiet, loai_dem = _gan_nhan_hoi_lai(sinh, chat, cau, lang)
    _ghi_jsonl(_stage_path(lang, f"{tien_to}hoi_lai_hc"), rows)
    _ghi_jsonl(_stage_path(lang, f"{tien_to}hoi_lai_hc_chi_tiet"), chi_tiet)
    theo_ham: dict[str, int] = {}
    for r in rows:
        theo_ham[r["fn_dich"]] = theo_ham.get(r["fn_dich"], 0) + 1
    bao_cao = {"vao": len(cau), "teacher_xep": loai_dem, "giu_khop_dich": len(rows), "theo_ham": theo_ham}
    print(json.dumps(bao_cao, ensure_ascii=False, indent=2))
    return bao_cao


@app.function(
    image=image,
    gpu=TEACHER_GPU,
    volumes={VOL_PATH: volume},
    timeout=2 * 60 * 60,
)
def sinh_hoi_lai_hoan_canh(
    so_lo_moi_ham: int = 4,
    cau_moi_lo: int = 3,
    seed: int = 2060,
    lang: str = "vi",
    tien_to: str = "v5_",
    nhiet_do: float = 0.8,
) -> dict[str, Any]:
    """Câu HỎI LẠI kể hoàn cảnh — cặp đối chứng THỨ BA (PLAN.md mục U).

    Sửa sau chạy thử 1 (64% khớp hàm đích, lẫn tiếng Anh/Trung): bẫy chọn theo hàm (`BAY_THEO_HAM`),
    nhiệt độ 0,95 -> 0,8, lọc từ tiếng Anh (`tu_tieng_anh`), và CHỈ giữ câu mà người gán nhãn (không biết
    hàm đích) cũng hiểu ra đúng hàm đích — câu viết cho X mà đọc ra Y là câu không đủ manh mối.

    Cùng khung (vai, bẫy) với câu từ chối / gọi hàm kể hoàn cảnh của vòng 4, chia đều các hàm có
    hỏi lại (`implicit_examples`). Quy trình:
      1. teacher sinh câu (`build_contextual_ask_prompt`)
      2. CODE lọc: câu hợp lệ, không lai tiếng Anh, ≥ 10 chữ, KHÔNG ra lệnh (`co_ra_lenh`,
         `co_tieu_tu_sai_bao` — cùng lớp code của quy trình chấm hỏi lại), không trùng, không cùng khuôn
      3. teacher gán nhãn 3 loại (`build_ask_label_prompt`, không biết hàm đích) — chỉ giữ "ngam_hieu"
    Ghi `<tien_to>hoi_lai_hc.jsonl` (mẫu `<nexa_ask>`) + `<tien_to>hoi_lai_hc_chi_tiet.jsonl` (mọi câu kèm
    lý do bị loại). Bước giám khảo chạy SAU bằng `chon_de_xuat` (Qwen rồi Llama, cac_file=<tien_to>hoi_lai_hc)
    + `python -m gen.cham_ask` ở máy — đúng quy trình chấm hỏi lại đã có (PLAN.md E0b).
    """
    import random

    from gen.cham_ask import co_ra_lenh, co_tieu_tu_sai_bao
    from gen.loc import bo_cung_khuon, cau_hop_le, lai_tieng_anh, tu_tieng_anh
    from gen.teacher_prompts import (
        BAY_THEO_HAM,
        VAI_NGUOI_NOI,
        TeacherResponseError,
        build_contextual_ask_prompt,
        parse_query_gen_response,
    )
    from schema.loader import CATALOG

    rng = random.Random(seed)
    ham = [f for f in CATALOG.functions if f.implicit_examples]
    bo = [
        (f, rng.choice(VAI_NGUOI_NOI), rng.choice(BAY_THEO_HAM[f.name]))
        for f in ham
        for _ in range(so_lo_moi_ham)
    ]

    sinh, chat = _khoi_tao_teacher()
    outs = sinh(
        [chat(build_contextual_ask_prompt(cau_moi_lo, f, v, b, lang)) for f, v, b in bo], nhiet_do, 2048
    )

    ly_do_loai = ("json_hong", "loc_code", "tieng_anh", "qua_ngan", "ra_lenh", "trung", "cung_khuon")
    dem = dict.fromkeys(ly_do_loai, 0)
    chi_tiet: list[dict[str, Any]] = []
    cau: list[dict[str, Any]] = []
    da_thay: set[str] = set()
    for (f, vai, bay), out in zip(bo, outs, strict=True):
        try:
            ds = parse_query_gen_response(out)
        except TeacherResponseError:
            dem["json_hong"] += 1
            continue
        for q in ds:
            ly_do = ""
            if not cau_hop_le(q)[0] or lai_tieng_anh(q):
                ly_do = "loc_code"
            elif tu_tieng_anh(q):
                ly_do = "tieng_anh"
            elif len(q.split()) < 10:
                ly_do = "qua_ngan"
            elif co_ra_lenh(q) or co_tieu_tu_sai_bao(q):
                ly_do = "ra_lenh"
            elif " ".join(q.split()).lower() in da_thay:
                ly_do = "trung"
            if ly_do:
                dem[ly_do] += 1
                chi_tiet.append({"query": q, "fn_dich": f.name, "vai": vai, "bay": bay, "loai_vi": ly_do})
                continue
            da_thay.add(" ".join(q.split()).lower())
            cau.append({"query": q, "fn_dich": f.name, "vai": vai, "bay": bay})
    giu_khuon = set(bo_cung_khuon([c["query"] for c in cau]))
    for i, c in enumerate(cau):
        if i not in giu_khuon:
            dem["cung_khuon"] += 1
            chi_tiet.append({**c, "loai_vi": "cung_khuon"})
    cau = [c for i, c in enumerate(cau) if i in giu_khuon]

    rows, ct, loai_dem = _gan_nhan_hoi_lai(sinh, chat, cau, lang)
    chi_tiet += ct
    _ghi_jsonl(_stage_path(lang, f"{tien_to}hoi_lai_hc"), rows)
    _ghi_jsonl(_stage_path(lang, f"{tien_to}hoi_lai_hc_chi_tiet"), chi_tiet)
    theo_ham: dict[str, int] = {}
    for r in rows:
        theo_ham[r["fn_dich"]] = theo_ham.get(r["fn_dich"], 0) + 1
    bao_cao = {
        "so_lo": len(bo),
        "qua_loc_code": len(cau),
        "loai_o_buoc_code": dem,
        "teacher_xep": loai_dem,
        "giu_hoi_lai_khop_dich": len(rows),
        "theo_ham_dich": dict(sorted(theo_ham.items())),
    }
    print(json.dumps(bao_cao, ensure_ascii=False, indent=2))
    return bao_cao


# ── VÒNG v6 (PLAN.md Phase 7 "KẾ HOẠCH v6") ─────────────────────────────────
# Từ mượn / viết tắt hợp lệ trong câu lệnh xe (bổ sung cho gen.loc.TU_MUON_QUEN khi lọc câu K1–K4)
_V6_MIEN_ANH = {"a", "c", "fm", "odo", "am"}
# hàm lượt 2 "cùng việc" với hàm lượt 1 — kiểu nối tiếp doi_chieu / them_nua / bo_phan_khac phải rơi vào đây
_V6_CUNG_VIEC = {
    "open_trunk": {"open_trunk", "close_trunk"}, "close_trunk": {"open_trunk", "close_trunk"},
    "lock_doors": {"lock_doors", "unlock_doors"}, "unlock_doors": {"lock_doors", "unlock_doors"},
}
# K2: tham số BẮT BUỘC có trong nhãn cho từng loại tính năng mới (lệch -> loại, ghi lý do)
_V6_K2_KIEM = {
    "delta_so": lambda a: "delta" in a,
    "delta_khong_so": lambda a: a.get("delta") in (-2, 2),
    "ve_so": lambda a: "temperature" in a and "delta" not in a,
    "suoi": lambda a: a.get("mode") == "heat",
    "lam_lanh": lambda a: a.get("mode") == "cool",
    "tu_dong": lambda a: a.get("mode") == "auto",
    "hai_kinh_sau": lambda a: a.get("window") == "rear",
    "hai_kinh_truoc": lambda a: a.get("window") == "front",
}


@app.function(image=image, gpu=TEACHER_GPU, volumes={VOL_PATH: volume}, timeout=3 * 60 * 60)
def sinh_v6(
    khoi: str = "K1,K2,K3,K4",
    thu: bool = False,
    seed: int = 2070,
    lang: str = "vi",
    tien_to: str = "v6_",
    k1_lo_moi_ham: int = 24,
    k3_lo_moi_cap: int = 8,
    k4_so_luot_1: int = 750,
) -> dict[str, Any]:
    """Sinh dữ liệu v6 (gen/v6.py). `thu=True`: chạy thử nhỏ (~1 lô / nhóm) để đọc mẫu trước khi chạy thật.

    K1 đa dạng cách nói 24 hàm · K2 tính năng mới (delta, mode, hai kính) · K3 cặp đối xứng với câu từ chối
    sát nách ·
    K4 hai lượt (cần `data/<lang>/v6_luot_1.jsonl` — `python -m gen.dung_vong_6 --luot-1` rồi upload). Mỗi
    khối:
    teacher sinh câu -> CODE lọc -> teacher trích tham số (biết hàm) / gán nhãn theo ngữ cảnh -> luật code
    (quy ước
    nhiệt độ, kiểm loại) -> giám khảo Qwen. Ghi `<tien_to><khối>.jsonl` (mọi câu kèm `giu` + lý do loại).
    """
    import random

    from gen.loc import cau_hop_le, lai_tieng_anh, tham_so_hop_le, tu_tieng_anh
    from gen.teacher_prompts import (
        TeacherResponseError,
        build_call_gen_prompt,
        build_verify_prompt,
        parse_query_gen_response,
        parse_verify_response,
    )
    from gen.v6 import (
        CAP_DOI_XUNG,
        HAM_LUOT_1,
        TINH_NANG_MOI,
        ap_quy_uoc,
        build_da_dang_prompt,
        build_doi_xung_prompt,
        build_nhan_noi_tiep_prompt,
        build_noi_tiep_prompt,
        build_tinh_nang_moi_prompt,
        build_verify_noi_tiep_prompt,
        chon_giong,
        lap_lo_k1,
        parse_nhan_noi_tiep,
        parse_tham_so_v6,
        tham_so_v6,
    )
    from schema.call_format import render_call, render_readable
    from schema.loader import CATALOG

    rng = random.Random(seed)
    cac_khoi = {k.strip() for k in khoi.split(",") if k.strip()}
    sinh, chat = _khoi_tao_teacher()
    bao_cao: dict[str, Any] = {}

    def loc_cau(q: str, ham: str) -> str:
        """'' = giữ; khác = lý do loại."""
        if not cau_hop_le(q)[0] or lai_tieng_anh(q):
            return "loc_code"
        if ham not in ("play_media", "navigate_to", "find_nearby") and [
            w for w in tu_tieng_anh(q) if w.lower() not in _V6_MIEN_ANH
        ]:
            return "tieng_anh"
        return ""

    def sinh_cau(ds: list[tuple[str, dict[str, Any]]], nhiet: float) -> list[tuple[dict[str, Any], str]]:
        """ds = [(prompt, thông tin lô)] -> [(thông tin lô, câu)] — bỏ trùng chữ y hệt trong cùng khối."""
        outs = sinh([chat(p) for p, _ in ds], nhiet, 1024)
        ra: list[tuple[dict[str, Any], str]] = []
        da_thay: set[str] = set()
        for (_, lo), out in zip(ds, outs, strict=True):
            try:
                cau = parse_query_gen_response(out)
            except TeacherResponseError:
                continue
            for q in cau:
                k = " ".join(q.lower().split())
                if k not in da_thay:
                    da_thay.add(k)
                    ra.append((lo, q))
        return ra

    def gan_va_cham(mau: list[dict[str, Any]]) -> None:
        """Mẫu biết trước hàm (`fn`): teacher trích tham số -> luật -> giám khảo. Ghi `args`, `giu`,
        `ly_do`."""
        prompts = [chat(build_call_gen_prompt(m["query"], CATALOG.by_name[m["fn"]])) for m in mau]
        outs = sinh(prompts, 0.1, 512)
        can_cham: list[dict[str, Any]] = []
        for m, out in zip(mau, outs, strict=True):
            try:
                args = parse_tham_so_v6(out, m["fn"])
            except TeacherResponseError as e:
                m.update(giu=False, ly_do=f"tham_so_hong: {str(e)[:80]}")
                continue
            if not tham_so_hop_le(args)[0]:
                m.update(giu=False, ly_do="tham_so_hong")
                continue
            args = ap_quy_uoc(m["fn"], m["query"], args)
            if args is None:
                m.update(giu=False, ly_do="quy_uoc_nhiet")
                continue
            m["args"] = args
            if (kiem := _V6_K2_KIEM.get(m.get("loai", ""))) is not None and not kiem(args):
                m.update(giu=False, ly_do="lech_loai_k2")
                continue
            can_cham.append(m)
        prompts = [
            chat(build_verify_prompt(m["query"], CATALOG.by_name[m["fn"]], m["args"])) for m in can_cham
        ]
        outs = sinh(prompts, 0.0, 256)
        for m, out in zip(can_cham, outs, strict=True):
            try:
                kq = parse_verify_response(out)
                m.update(giu=kq.valid, ly_do="" if kq.valid else f"giam_khao: {kq.reason}")
            except TeacherResponseError:
                m.update(giu=False, ly_do="giam_khao: không đọc được")

    def ghi(ten: str, mau: list[dict[str, Any]]) -> None:
        _ghi_jsonl(_stage_path(lang, f"{tien_to}{ten}"), mau)
        ly_do: dict[str, int] = {}
        for m in mau:
            if not m.get("giu"):
                k = (m.get("ly_do") or "?").split(":")[0]
                ly_do[k] = ly_do.get(k, 0) + 1
        bao_cao[ten] = {"cau": len(mau), "giu": sum(bool(m.get("giu")) for m in mau), "loai": ly_do}
        print(f"[{ten}] {bao_cao[ten]}", flush=True)

    # ── K1 ──
    if "K1" in cac_khoi:
        lo = lap_lo_k1(1 if thu else k1_lo_moi_ham, 5, seed)
        cau = sinh_cau([(build_da_dang_prompt(x), {"lo": x}) for x in lo], 0.9)
        mau: list[dict[str, Any]] = []
        for info, q in cau:
            x = info["lo"]
            # KHÔNG lọc theo độ dài (chạy thử: lô "1–3 chữ" ra 27/120 câu ≤ 3 chữ, 73 câu 4–6 chữ — vẫn là câu
            # ngắn tự nhiên, đáng giữ); chỉ ghi lại số chữ thật để báo cáo phân bố
            m = {"khoi": "K1", "query": q, "fn": x.ham, "goi": x.goi, "do_dai": x.do_dai, "giong": x.giong,
                 "so_chu": len(q.split())}
            ly_do = loc_cau(q, x.ham)
            if not ly_do and x.goi.lower() not in q.lower():
                ly_do = "thieu_cach_goi"
            if ly_do:
                m.update(giu=False, ly_do=ly_do)
            mau.append(m)
        gan_va_cham([m for m in mau if "giu" not in m])
        ghi("k1_da_dang", mau)

    # ── K2 ──
    if "K2" in cac_khoi:
        ds = [
            (build_tinh_nang_moi_prompt(mo_ta, chon_giong(rng), 5), {"loai": loai, "fn": ham})
            for loai, ham, mo_ta, so_lo in TINH_NANG_MOI
            for _ in range(1 if thu else so_lo)
        ]
        mau = []
        for info, q in sinh_cau(ds, 0.9):
            m = {"khoi": "K2", "query": q, **info}
            if ly_do := loc_cau(q, info["fn"]):
                m.update(giu=False, ly_do=ly_do)
            mau.append(m)
        gan_va_cham([m for m in mau if "giu" not in m])
        ghi("k2_moi", mau)

    # ── K3 ──
    if "K3" in cac_khoi:
        ds = [
            (build_doi_xung_prompt(ham, cum, de_nham, chon_giong(rng), 5), {"fn": ham, "cum": cum})
            for ham, cum, de_nham in CAP_DOI_XUNG
            for _ in range(1 if thu else k3_lo_moi_cap)
        ]
        mau = []
        for info, q in sinh_cau(ds, 0.9):
            m = {"khoi": "K3", "query": q, **info}
            if ly_do := loc_cau(q, info["fn"]):
                m.update(giu=False, ly_do=ly_do)
            mau.append(m)
        gan_va_cham([m for m in mau if "giu" not in m])
        ghi("k3_doi_xung", mau)

    # ── K4 ──
    if "K4" in cac_khoi:
        luot_1 = _doc_jsonl(_stage_path(lang, "v6_luot_1")) or []
        if not luot_1:
            raise RuntimeError(
                "thiếu data/<lang>/v6_luot_1.jsonl — chạy `python -m gen.dung_vong_6 --luot-1` rồi upload"
            )
        luot_1 = luot_1[: (12 if thu else k4_so_luot_1)]
        ds = []
        for r in luot_1:
            goi_1 = render_readable(r["fn"], r["args"])
            for kieu in HAM_LUOT_1[r["fn"]]:
                p = build_noi_tiep_prompt(r["query"], goi_1, kieu, chon_giong(rng), 2)
                ds.append((p, {**r, "kieu": kieu}))
        rng.shuffle(ds)  # chạy thử giữ được 36% -> dùng ĐỦ mọi kiểu hợp lệ cho mỗi lượt 1 (~2.700 lô × 2 câu)
        mau = []
        for info, q in sinh_cau(ds, 0.9):
            m = {"khoi": "K4", "cau_1": info["query"], "fn_1": info["fn"], "args_1": info["args"],
                 "goi_1": render_call(info["fn"], info["args"]), "kieu": info["kieu"], "query": q}
            if (ly_do := loc_cau(q, "")) or (ly_do := "" if len(q.split()) <= 14 else "qua_dai"):
                m.update(giu=False, ly_do=ly_do)
            mau.append(m)
        can = [m for m in mau if "giu" not in m]
        outs = sinh([chat(build_nhan_noi_tiep_prompt(m["cau_1"], render_readable(m["fn_1"], m["args_1"]),
                                                     m["query"])) for m in can], 0.0, 512)
        can_cham = []
        for m, out in zip(can, outs, strict=True):
            try:
                ten, raw = parse_nhan_noi_tiep(out)
                args = {} if ten == "irrelevant_function" else tham_so_v6(raw, ten)
            except TeacherResponseError as e:
                m.update(giu=False, ly_do=f"nhan_hong: {str(e)[:80]}")
                continue
            if (args := ap_quy_uoc(ten, m["query"], args)) is None:
                m.update(giu=False, ly_do="quy_uoc_nhiet")
                continue
            cung_viec = _V6_CUNG_VIEC.get(m["fn_1"], {m["fn_1"]})
            sai_kieu = (
                (m["kieu"] in ("doi_chieu", "them_nua", "bo_phan_khac") and ten not in cung_viec)
                or (m["kieu"] == "ngoai_pham_vi" and ten != "irrelevant_function")
                or (m["kieu"] == "lenh_moi" and ten in ("irrelevant_function", *cung_viec))
            )
            m.update(fn=ten, args=args)
            if sai_kieu:
                m.update(giu=False, ly_do="lech_kieu")
                continue
            can_cham.append(m)
        outs = sinh([chat(build_verify_noi_tiep_prompt(m["cau_1"], render_readable(m["fn_1"], m["args_1"]),
                                                       m["query"], render_readable(m["fn"], m["args"])))
                     for m in can_cham], 0.0, 256)
        for m, out in zip(can_cham, outs, strict=True):
            try:
                kq = parse_verify_response(out)
                m.update(giu=kq.valid, ly_do="" if kq.valid else f"giam_khao: {kq.reason}")
            except TeacherResponseError:
                m.update(giu=False, ly_do="giam_khao: không đọc được")
        ghi("k4_hai_luot", mau)

    print(json.dumps(bao_cao, ensure_ascii=False, indent=2))
    return bao_cao


BASELINE_MODEL = {
    # tên: (thư mục model, thư mục tokenizer, kiểu prompt)
    "sip_72b": (MODEL_DIR, MODEL_DIR, "sip"),  # teacher Qwen2.5-72B, KHÔNG train
    "sip_1_5b": (
        f"{VOL_PATH}/models/student-base",
        f"{VOL_PATH}/models/student-base",
        "sip",
    ),  # Qwen 1,5B gốc
    "student": (
        f"{VOL_PATH}/models/student-v4dc-s42/checkpoint-2206",
        f"{VOL_PATH}/models/student-v4dc-s42",
        "ft",
    ),
}


@app.function(image=image, gpu=TEACHER_GPU, volumes={VOL_PATH: volume}, timeout=60 * 60)
def baseline(
    ten: str = "sip_1_5b",
    thu_muc: str = "final_cuoi",
    cac_tap: str = "test_vi_sach_v2,test_vi_bo_sung,dev_vi_2",
    so_cau_do_tre: int = 40,
    lang: str = "vi",
) -> dict[str, Any]:
    """Chạy baseline (PLAN.md Phase 3) trên các tập — CÙNG bộ máy vLLM cho mọi model để so độ trễ công bằng.

    - sip_*: prompt "mô tả hàm trong prompt" (`eval/baseline.py`), JSON -> chuyển sang `<nexa_i>(...)` để
      chấm bằng bộ chấm cũ ở máy. student: prompt ngắn của bài (functional token), dừng ở `<nexa_end>`.
    - Độ chính xác: sinh theo lô (nhanh). Độ trễ: `so_cau_do_tre` câu đầu, MỖI LẦN MỘT CÂU (như dùng thật trên
      xe), đo thời gian + số token prompt/đầu ra. Số đo trên A100, không phải trên thiết bị.
    """
    import statistics

    from vllm import LLM, SamplingParams

    from eval.baseline import build_sip_prompt, sang_nexa
    from gen.records import query_from_call_prompt

    model_dir, tok_dir, kieu = BASELINE_MODEL[ten]
    llm = LLM(
        model=model_dir,
        tokenizer=tok_dir,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        enforce_eager=True,
    )
    tok = llm.get_tokenizer()
    if kieu == "sip":
        params = SamplingParams(temperature=0.0, max_tokens=256)
    else:
        params = SamplingParams(
            temperature=0.0,
            max_tokens=64,
            stop=["<nexa_end>"],
            include_stop_str_in_output=True,
            skip_special_tokens=False,
        )

    def lam_prompt(row: dict[str, Any]) -> str:
        if kieu != "sip":
            return row["prompt"]
        m = row.get("meta") or {}
        if "fn_1" in m and "\nLượt trước:" in row["prompt"]:  # tập hai lượt: lời gọi lượt trước như student
            return build_sip_prompt(m["cau"], (m["fn_1"], m["args_1"]))
        return build_sip_prompt(query_from_call_prompt(row["prompt"]))

    def ra_nexa(text: str) -> str:
        return sang_nexa(text) if kieu == "sip" else text.strip()

    bao_cao: dict[str, Any] = {"ten": ten}
    tat_ca: list[dict[str, Any]] = []
    for tap in cac_tap.split(","):
        rows = _doc_jsonl(f"{DATA_DIR}/{lang}/{thu_muc}/{tap}.jsonl") or []
        outs = llm.generate([lam_prompt(r) for r in rows], params)
        ket = [
            {"prompt": r["prompt"], "output": ra_nexa(o.outputs[0].text), "raw": o.outputs[0].text}
            for r, o in zip(rows, outs, strict=True)
        ]
        _ghi_jsonl(f"{DATA_DIR}/{lang}/{thu_muc}/du_doan_bl_{ten}_{tap}.jsonl", ket)
        bao_cao[tap] = {"so_cau": len(rows), "doc_khong_duoc": sum(1 for k in ket if not k["output"])}
        tat_ca += rows

    # độ trễ: từng câu một
    thoi_gian, tok_vao, tok_ra = [], [], []
    for r in tat_ca[:so_cau_do_tre]:
        p = lam_prompt(r)
        t0 = time.perf_counter()
        o = llm.generate([p], params, use_tqdm=False)[0]
        thoi_gian.append((time.perf_counter() - t0) * 1000)
        tok_vao.append(len(tok.encode(p)))
        tok_ra.append(len(o.outputs[0].token_ids))
    s = sorted(thoi_gian)
    bao_cao["do_tre"] = {
        "so_cau": len(s),
        "p50_ms": round(s[len(s) // 2], 1),
        "p95_ms": round(s[min(len(s) - 1, int(len(s) * 0.95))], 1),
        "token_prompt_tb": round(statistics.mean(tok_vao), 1),
        "token_ra_tb": round(statistics.mean(tok_ra), 1),
    }
    print(json.dumps(bao_cao, ensure_ascii=False, indent=2))
    _ghi_jsonl(f"{DATA_DIR}/{lang}/{thu_muc}/bao_cao_bl_{ten}.jsonl", [bao_cao])
    return bao_cao


@app.function(image=image, volumes={VOL_PATH: volume}, timeout=300)
def show_progress(lang: str = "vi") -> None:
    for ten in (
        "stage1_queries",
        "stage2_calls",
        "stage3_replies",
        f"train_{lang}",
        "ask_stage1_queries",
        "ask_stage2_labels",
        f"ask_train_{lang}",
    ):
        rows = _doc_jsonl(_stage_path(lang, ten))
        print(f"  {ten:18} {'chưa có' if rows is None else f'{len(rows)} dòng'}")
