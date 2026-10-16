"""ĐO TTS — bước 3: chấm độ rõ. Chữ trợ lý định nói vs chữ STT nghe lại từ tiếng VieNeu.

Chuẩn hoá (`chuan`): NFC, thường hoá, số -> chữ đọc, "%" / "km" -> chữ, một kiểu đặt dấu thanh, bỏ mọi dấu câu
(câu trả lời nhiều dấu phẩy; STT có ghi hay không không phải lỗi đọc).
Báo: WER gộp (tổng lỗi / tổng từ), CER, % câu nghe đúng trọn, tách câu có số / không số; câu CẢ HAI giám
khảo cùng sai (khả năng cao là TTS đọc chưa rõ, vì hai STT khác kiến trúc khó sai cùng chỗ); tốc độ tạo tiếng.
Chạy:

    conda run -n giong_noi python -m eval.tts.cham
"""

from __future__ import annotations

import json
import re
import statistics
import sys
import unicodedata
from pathlib import Path

from eval.stt.ft.tao_tts import so_sang_chu
from serve.tra_loi import KHA_NANG

DONG_TU = {"bật", "tắt", "mở", "đóng", "hạ", "nâng", "tăng", "giảm", "chỉnh", "mức", "khoá"}

VAO = Path("runs/tts/nghe_lai.jsonl")
RA = Path("runs/tts/ket_qua.txt")
GIAM_KHAO = ["phowhisper-small", "zipformer"]


# hai kiểu đặt dấu thanh đều đúng chính tả ("hoà" / "hòa", "khoá" / "khóa", "thuỷ" / "thủy")
# -> đưa về một kiểu
DAU_CU = {
    "òa": "oà",
    "óa": "oá",
    "ỏa": "oả",
    "õa": "oã",
    "ọa": "oạ",
    "òe": "oè",
    "óe": "oé",
    "ỏe": "oẻ",
    "õe": "oẽ",
    "ọe": "oẹ",
    "ùy": "uỳ",
    "úy": "uý",
    "ủy": "uỷ",
    "ũy": "uỹ",
    "ụy": "uỵ",
}


# âm tiết tiếng Việt = [phụ âm đầu] + 1–3 nguyên âm + [phụ âm cuối]; từ không khớp (NexaAI, Playlist, PSI,
# ATM, charging_station_1…) là từ ngoại / tên riêng / mã máy — chia nhóm bằng QUY TẮC, không chọn tay từng câu
_NGUYEN_AM = "aàáảãạăằắẳẵặâầấẩẫậeèéẻẽẹêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹỵ"
_DAU = "ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|[bcdđghklmnpqrstvx]"
_AM_TIET = re.compile(rf"({_DAU})?[{_NGUYEN_AM}]{{1,3}}(ch|ng|nh|[cmnpt])?")
_DON_VI = {"km"}


def co_tu_ngoai(cau: str) -> bool:
    for tu in re.findall(r"[^\W\d]+(?:_\w+)?", unicodedata.normalize("NFC", cau).lower()):
        if tu not in _DON_VI and ("_" in tu or not _AM_TIET.fullmatch(tu)):
            return True
    return False


# cách đọc số khác nhau mà cùng nghĩa -> một dạng
TU_TUONG_DUONG = {
    "lẻ": "linh",
    "ngàn": "nghìn",
    "tư": "bốn",
    "mốt": "một",
    "lăm": "năm",
    "nhăm": "năm",
    "bẩy": "bảy",
}


def chuan(s: str) -> list[str]:
    """So ở dạng CHỮ ĐỌC: số -> chữ (`so_sang_chu`, hàm đã dùng làm dữ liệu fine-tune STT) cho cả câu gốc
    lẫn chữ STT — tránh "3" / "ba", "5" / "năm" bị tính sai ("ba", "năm" đứng một mình `doi_so` không đổi)."""
    # 23.450 -> 23450 (dấu hàng nghìn), trước khi đổi số sang chữ
    s = re.sub(r"(\d)\.(\d{3})\b", r"\1\2", unicodedata.normalize("NFC", s))
    s = so_sang_chu(s).lower().replace("%", " phần trăm ")
    s = re.sub(r"\bkm\b", "ki lô mét", s)
    for cu, moi in DAU_CU.items():
        s = s.replace(cu, moi)
    return [TU_TUONG_DUONG.get(t, t) for t in re.sub(r"[^\w\s]", " ", s).split()]


def sua(a: list, b: list) -> int:
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[len(b)]


def main() -> None:
    rows = [json.loads(x) for x in VAO.open(encoding="utf-8")]
    ra: list[str] = [f"Đo độ rõ TTS (VieNeu, giọng Trúc Ly): {len(rows)} câu trả lời khác nhau của trợ lý"]
    for r in rows:
        r["goc"] = chuan(r["text"])
        r["co_so"] = bool(re.search(r"\d", r["text"]))
        r["ngoai"] = co_tu_ngoai(r["text"])
        for g in GIAM_KHAO:
            r[f"loi_{g}"] = sua(r["goc"], chuan(r[f"chu_{g}"]))
            goc_c, nghe_c = " ".join(r["goc"]), " ".join(chuan(r[f"chu_{g}"]))
            r[f"loi_c_{g}"] = sua(list(goc_c), list(nghe_c))
    tu = sum(len(r["goc"]) for r in rows)
    ky_tu = sum(len(" ".join(r["goc"])) for r in rows)
    ra.append(f"Tổng {tu} từ · {ky_tu} ký tự · {sum(r['co_so'] for r in rows)} câu có số")
    for g in GIAM_KHAO:
        ra.append(f"\n== Giám khảo {g}")
        for ten, nhom in (
            ("tất cả", rows),
            ("câu có số", [r for r in rows if r["co_so"]]),
            ("câu không số", [r for r in rows if not r["co_so"]]),
            ("thuần Việt", [r for r in rows if not r["ngoai"]]),
            ("có từ ngoại", [r for r in rows if r["ngoai"]]),
        ):
            n_tu = sum(len(r["goc"]) for r in nhom)
            wer = 100 * sum(r[f"loi_{g}"] for r in nhom) / n_tu
            dung = sum(r[f"loi_{g}"] == 0 for r in nhom)
            ra.append(
                f"  {ten:13} {len(nhom):3d} câu · WER {wer:5.1f}% · nghe đúng trọn {dung}/{len(nhom)} "
                f"({100 * dung / len(nhom):.0f}%)"
            )
        cer = 100 * sum(r[f"loi_c_{g}"] for r in rows) / ky_tu
        ra.append(f"  CER (ký tự) {cer:.1f}%")
    ca_hai = [r for r in rows if all(r[f"loi_{g}"] for g in GIAM_KHAO)]
    ra.append(f"\n== Cả hai giám khảo cùng nghe sai: {len(ca_hai)}/{len(rows)} câu (nghi TTS đọc chưa rõ)")
    for r in sorted(ca_hai, key=lambda r: -sum(r[f"loi_{g}"] for g in GIAM_KHAO)):
        ra.append(f"  [{r['id']}] định nói: {r['text']}")
        for g in GIAM_KHAO:
            ra.append(f"         {g[:9]:9}: {r[f'chu_{g}']}")
    # TỪ KHOÁ XE (âm tiết trong danh sách khả năng `serve.tra_loi.KHA_NANG` + động từ điều khiển — có từ
    # trước, không chọn theo lỗi đã thấy): câu có từ đó mà CẢ HAI giám khảo đều không nghe ra từ đó
    tu_khoa = sorted(set(chuan(KHA_NANG)) | DONG_TU)
    ra.append("\n== Từ khoá xe bị CẢ HAI giám khảo nghe mất (số câu mất / số câu có từ)")
    for tu in tu_khoa:
        co = [r for r in rows if tu in r["goc"]]
        mat = [r["id"] for r in co if all(tu not in chuan(r[f"chu_{g}"]) for g in GIAM_KHAO)]
        if mat:
            ra.append(f"  {tu:6} {len(mat):2d}/{len(co):3d}  {' '.join(mat[:8])}")
    rtf = [r["tao_ms"] / 1000 / r["giay"] for r in rows]
    tao = statistics.median(r["tao_ms"] for r in rows)
    giay = statistics.median(r["giay"] for r in rows)
    ra.append(
        f"\n== Tốc độ tạo tiếng (cả câu, CPU): tạo p50 {tao:.0f} ms cho p50 {giay:.1f} s tiếng · "
        f"RTF p50 {statistics.median(rtf):.2f} · lớn nhất {max(rtf):.2f}"
    )
    RA.write_text("\n".join(ra) + "\n", encoding="utf-8")
    print("\n".join(ra))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
