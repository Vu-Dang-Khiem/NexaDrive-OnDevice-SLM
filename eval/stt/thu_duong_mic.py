"""KIỂM ĐƯỜNG MIC CỦA DEMO bằng bộ ghi âm CŨ (bộ C + D, giọng người dùng) — không cần ghi âm mới.

So cùng một dịch vụ giọng nói đang chạy (cổng 8001):
  A. gửi thẳng file WAV lên /stt                                    (mốc — như lúc đo bộ C / D)
  B. nút micro của giao diện: Chromium dùng file làm MIC GIẢ (--use-file-for-fake-audio-capture),
     trang ghi bằng MediaRecorder (webm/opus) rồi gửi /stt — đúng đường người dùng bấm mic
Nếu B kém A rõ -> lỗi nằm ở đường demo (sửa được bằng code); nếu ngang -> lỗi ở điều kiện nói trực tiếp.

    python eval/stt/thu_duong_mic.py   (Python hệ thống — có Playwright; cần 8000 / 8001 / 5173 đang chạy)
Ra: runs/stt/duong_mic/ket_qua.jsonl (+ âm thanh webm trình duyệt gửi đi, để nghe lại).
"""
import json
import sys
import time
import urllib.request
import wave
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding="utf-8")
STT = "http://127.0.0.1:8001/stt"
RA = Path("runs/stt/duong_mic")
RA.mkdir(parents=True, exist_ok=True)


def danh_sach():
    for bo in ("bo_c", "bo_d"):
        for x in open(f"data/stt/{bo}/danh_sach.jsonl", encoding="utf-8"):
            r = json.loads(x)
            yield r["id"], Path(f"data/stt/{bo}/khiem/{r['tep']}").resolve(), r["cau"]


def gui_thang(tep: Path) -> dict:
    req = urllib.request.Request(STT, data=tep.read_bytes(), headers={"Content-Type": "audio/wav"},
                                 method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def qua_mic_demo(p, tep: Path, giay: float) -> dict:
    b = p.chromium.launch(args=[
        "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
        f"--use-file-for-fake-audio-capture={tep}%noloop", "--autoplay-policy=no-user-gesture-required",
        "--mute-audio", "--enable-gpu", "--ignore-gpu-blocklist", "--use-angle=d3d11"])
    try:
        pg = b.new_context(permissions=["microphone"]).new_page()
        # đo trễ: bấm mic -> MediaRecorder thật sự bắt đầu ghi (người nói ngay sau khi bấm sẽ mất đoạn này)
        pg.add_init_script("""window.__t = {};
          document.addEventListener('click', e => {
            if (e.target.closest && e.target.closest('#mic') && !window.__t.bam)
              window.__t.bam = performance.now();
          }, true);
          const goc = MediaRecorder.prototype.start;
          MediaRecorder.prototype.start = function (...a) {
            window.__t.ghi = performance.now(); return goc.apply(this, a);
          };""")
        kq: dict = {}
        pg.on("response", lambda r: r.url.endswith("/stt") and kq.update(r.json()))
        pg.on("request", lambda r: r.url.endswith("/stt") and kq.update(_webm=r.post_data_buffer))
        pg.goto("http://localhost:5173", wait_until="domcontentloaded")
        pg.wait_for_timeout(3000)
        pg.mouse.click(5, 5)  # lần tương tác đầu: mở khoá âm thanh
        pg.wait_for_timeout(300)
        if pg.get_attribute("#heyNexa", "aria-pressed") == "true":
            pg.click("#heyNexa")  # tắt «hey nexa» để mic giả không bị luồng nghe nền giữ trước
        pg.wait_for_timeout(500)
        pg.click("#mic")  # bắt đầu ghi — trình duyệt mở mic giả, file phát từ đầu
        pg.wait_for_timeout(int(giay * 1000) + 1500)  # dừng muộn: file phát từ lúc mic mở, không cắt oan đuôi
        t = pg.evaluate("window.__t")
        pg.click("#mic")  # dừng + gửi
        for _ in range(100):
            if "chu_stt" in kq:
                break
            pg.wait_for_timeout(100)
        kq["_tre_ms"] = round(t["ghi"] - t["bam"]) if "ghi" in t and "bam" in t else None
        return kq
    finally:
        b.close()


def main():
    ra = []
    gioi_han = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else None  # chạy thử vài câu
    with sync_playwright() as p:
        for i, tep, cau in list(danh_sach())[:gioi_han]:
            giay = (lambda w: w.getnframes() / w.getframerate())(wave.open(str(tep)))
            a = gui_thang(tep)
            bb = qua_mic_demo(p, tep, giay)
            webm = bb.pop("_webm", None)
            if webm:
                (RA / f"{i}.webm").write_bytes(webm)
            d = {"id": i, "cau": cau, "giay": round(giay, 2), "thang": a.get("chu_stt", ""),
                 "mic_demo": bb.get("chu_stt"), "webm_kb": round(len(webm) / 1024, 1) if webm else None,
                 "tre_bat_mic_ms": bb.get("_tre_ms")}
            ra.append(d)
            dau = "=" if d["thang"] == d["mic_demo"] else "≠"
            print(f"{i}  {dau}  thẳng: {d['thang']}\n      mic : {d['mic_demo']}", flush=True)
            time.sleep(0.2)
    with open(RA / "ket_qua.jsonl", "w", encoding="utf-8") as f:
        for d in ra:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    giong = sum(d["thang"] == d["mic_demo"] for d in ra)
    print(f"\nGiống hệt: {giong}/{len(ra)} · mic demo trả rỗng / lỗi: {sum(not d['mic_demo'] for d in ra)}")


if __name__ == "__main__":
    main()
