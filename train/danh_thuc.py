"""Wake-word "hey nexa" — train bộ phân loại + chọn ngưỡng theo luật KHOÁ (PLAN.md Phase 6b,
"Wake-word — KẾ HOẠCH"). Chạy trong env `giong_noi`, CPU, $0:

    python -m train.danh_thuc

Dữ liệu: `data/wake/tts/` (VieNeu, chia theo giọng) + FLEURS tiếng Việt phần DEV
(`data/wake/fleurs/dev.tar.gz`, giọng người thật, chia 70% train / 30% val theo seed). FLEURS phần TEST và
bản ghi giọng người dùng KHÔNG đụng tới ở đây.

Đặc trưng lấy bằng ĐÚNG đường stream lúc chạy thật (`serve.danh_thuc.DacTrung`).
- Dương: câu "hey nexa" đã biến đổi (tốc độ, âm lượng, vang, ồn trắng / hồng / trầm / tiếng người) đặt
  trong khung 2,4 s, kết thúc ở 0–0,5 s cuối -> lấy đặc trưng khung CUỐI.
- Âm: mọi khung (bỏ 10 khung đầu còn dính bộ đệm khởi tạo) của câu gần giống / câu lệnh / FLEURS dev / ồn
  + đặc trưng ACAV100M (openWakeWord) + FLEURS phần TRAIN (thêm sau lần chạy đầu — xem PLAN).
Chọn (k khung liền, ngưỡng) trên VAL: xem `chon_nguong`. Bắt VAL = câu "hey nexa" của 5 giọng giữ riêng, đặt
giữa 1,2 s ồn trước + 1,0 s ồn sau, báo ở bất kỳ đâu trong đoạn.
"""

from __future__ import annotations

import io
import json
import random
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

from serve.danh_thuc import MO_HINH, DacTrung, dem_bao, diem

TTS = Path("data/wake/tts")
FLEURS_DEV = Path("data/wake/fleurs/dev.tar.gz")
FLEURS_TRAIN = Path("data/wake/fleurs/train.tar.gz")
CACHE = Path("data/wake/dac_trung_v2.npz")  # bản 1: dac_trung.npz (biến đổi âm lượng hẹp)
# Giọng THẬT người dùng (bản 2): 60 lần "hey nexa" để train (cột `tach` train / val, khoá trong danh sách)
# + bộ B (40 câu lệnh xe, giọng người dùng) làm ÂM. Bộ C và bộ đo bo_wake_2 KHÔNG đụng ở đây.
GHI_TRAIN = Path("data/stt/bo_wake_train/khiem")
BO_B = Path("data/stt/bo_b/khiem")
KHO_FLEURS = Path("data/wake/fleurs_train_dac_trung.npy")
# Âm bổ sung: 500.000 cửa sổ đầu của đặc trưng ACAV100M (~2.000 giờ tiếng nói nhiều thứ tiếng, nhạc, ồn —
# openWakeWord công bố, CÙNG bộ trích đặc trưng; giấy phép CC BY-NC-SA 4.0 -> model wake-word chỉ dùng phi
# thương mại).
ACAV = Path("data/wake/owf/acav_500k.npy")
SR = 16000
SEED = 2026_10_08
DAI = int(2.4 * SR)  # khung dương 2,4 s = 30 khung 80 ms
BO_DAU = 10  # khung đầu còn dính bộ đệm khởi tạo của openWakeWord


# ── âm thanh ───────────────────────────────────────────────────────────────
def doc_wav(tep: Path | io.BytesIO) -> np.ndarray:
    import soundfile as sf
    import soxr

    a, sr = sf.read(tep, dtype="float32", always_2d=True)
    a = a.mean(axis=1)
    return soxr.resample(a, sr, SR).astype(np.float32) if sr != SR else a


def doc_fleurs(tep: Path) -> dict[str, np.ndarray]:
    """Đọc wav trong tar.gz NGAY TRONG BỘ NHỚ (không giải nén ra đĩa)."""
    ra: dict[str, np.ndarray] = {}
    with tarfile.open(tep, "r:gz") as tar:
        for m in tar:
            if m.isfile() and m.name.endswith(".wav"):
                f = tar.extractfile(m)
                if f is not None:
                    ra[Path(m.name).name] = doc_wav(io.BytesIO(f.read()))
    return ra


def cat_lang(a: np.ndarray) -> np.ndarray:
    import librosa

    t, _ = librosa.effects.trim(a, top_db=35)
    return t if len(t) > SR // 5 else a


def sang_int16(a: np.ndarray) -> np.ndarray:
    return (np.clip(a, -1, 1) * 32767).astype(np.int16)


def on(loai: str, n: int, rng: np.random.Generator, tieng_nguoi: list[np.ndarray]) -> np.ndarray:
    if loai == "nguoi" and tieng_nguoi:
        src = tieng_nguoi[rng.integers(len(tieng_nguoi))]
        if len(src) < n:
            src = np.tile(src, n // len(src) + 1)
        s = rng.integers(0, len(src) - n + 1)
        x = src[s : s + n].copy()
    else:
        w = rng.standard_normal(n).astype(np.float32)
        if loai in ("hong", "tram"):
            f = np.fft.rfft(w)
            k = np.arange(len(f)) + 1.0
            f /= np.sqrt(k) if loai == "hong" else k  # 1/f (hồng) · 1/f² (trầm, giống ồn xe / gió)
            w = np.fft.irfft(f, n).astype(np.float32)
        x = w
    return x / (np.sqrt(np.mean(x**2)) + 1e-9)


def tron_on(
    a: np.ndarray, rng: np.random.Generator, tieng_nguoi: list[np.ndarray], p: float = 0.85
) -> np.ndarray:
    if rng.random() > p:
        return a
    loai = rng.choice(["trang", "hong", "tram", "nguoi"], p=[0.2, 0.25, 0.3, 0.25])
    snr = rng.uniform(3, 30)
    p_tin = np.sqrt(np.mean(a**2)) + 1e-9
    return a + on(loai, len(a), rng, tieng_nguoi) * p_tin / (10 ** (snr / 20))


def bien_doi(a: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Tốc độ ±12%, vang (xác suất 0,25), đỉnh câu −46..−3 dB (bản 2 — bản 1 −18..−3 dB, hụt mức thu nhỏ của
    micro laptop: bản ghi người dùng đỉnh −23..−38 dB)."""
    import soxr
    from scipy.signal import fftconvolve

    a = soxr.resample(a, SR, SR / rng.uniform(0.88, 1.12)).astype(np.float32)
    if rng.random() < 0.25:
        dai = int(rng.uniform(0.15, 0.5) * SR)
        rir = rng.standard_normal(dai) * np.exp(-6.9 * np.arange(dai) / dai)
        rir[0] = 1.0
        a = fftconvolve(a, rir)[: len(a)].astype(np.float32)
    a = a / (np.abs(a).max() + 1e-9) * 0.5
    return a * 10 ** (rng.uniform(-40, 3) / 20)


# ── dựng mẫu ───────────────────────────────────────────────────────────────
def mau_duong(cau: np.ndarray, rng: np.random.Generator, nguoi: list[np.ndarray]) -> np.ndarray:
    """Khung 2,4 s, câu "hey nexa" kết thúc trong 0–0,5 s cuối; trước đó là im / ồn (đôi khi tiếng người
    nhỏ)."""
    c = bien_doi(cat_lang(cau), rng)[: DAI - SR // 4]
    nen = np.zeros(DAI, np.float32)
    if rng.random() < 0.2 and nguoi:
        nen += on("nguoi", DAI, rng, nguoi) * 0.03
    het = max(DAI - int(rng.uniform(0, 0.5) * SR), len(c))  # giọng đọc chậm (~1,8 s) vẫn nằm trọn trong khung
    nen[het - len(c) : het] += c
    return sang_int16(tron_on(nen, rng, nguoi))


def dac_trung_duong(dt: DacTrung, cau: list[np.ndarray], so_lan: int, rng, nguoi) -> np.ndarray:
    return np.stack([dt.chuoi(mau_duong(c, rng, nguoi))[-1] for c in cau for _ in range(so_lan)])


def dac_trung_am(dt: DacTrung, doan: list[np.ndarray], so_lan: int, rng, nguoi, buoc: int = 1) -> np.ndarray:
    """Mọi khung (bỏ khung đầu) của từng đoạn, có đệm ồn 0,9 s trước / 0,4 s sau và biến đổi."""
    ra = []
    truoc, sau = np.zeros(int(0.9 * SR), np.float32), np.zeros(int(0.4 * SR), np.float32)
    for a in doan:
        for _ in range(so_lan):
            x = np.concatenate([truoc, bien_doi(a, rng), sau])
            v = dt.chuoi(sang_int16(tron_on(x, rng, nguoi, p=0.6)))
            ra.append(v[BO_DAU::buoc])
    return np.concatenate(ra) if ra else np.zeros((0, 1536))


# ── đo trên VAL ────────────────────────────────────────────────────────────
def dong_am(doan: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Nối các đoạn âm thành MỘT dòng liên tục (cách nhau 0,3–1,0 s im) — đo báo nhầm / giờ như lúc chạy
    thật."""
    phan = []
    for a in doan:
        phan += [a, np.zeros(int(rng.uniform(0.3, 1.0) * SR), np.float32)]
    return sang_int16(np.concatenate(phan)) if phan else np.zeros(0, np.int16)


def dac_trung_bat(dt: DacTrung, cau: list[np.ndarray], rng, nguoi, so_lan: int = 3) -> list[np.ndarray]:
    """Đặc trưng cả đoạn (1,2 s ồn + câu + 1,0 s ồn) cho mỗi lần thử bắt trên VAL (như `diem_bat`)."""
    ra = []
    for c in cau:
        for _ in range(so_lan):
            x = np.concatenate(
                [np.zeros(int(1.2 * SR), np.float32), bien_doi(cat_lang(c), rng), np.zeros(SR, np.float32)]
            )
            ra.append(dt.chuoi(sang_int16(tron_on(x, rng, nguoi)))[BO_DAU:])
    return ra


def kho_fleurs_train(dt: DacTrung) -> np.ndarray:
    """Đặc trưng FLEURS tiếng Việt phần TRAIN (~10 giờ giọng người thật, CC BY 4.0) làm ÂM — tính một lần,
    lấy 1 khung / 4 (cách 320 ms), lưu float16 rồi đọc thẳng từ đĩa (memmap)."""
    if not KHO_FLEURS.exists():
        ra = []
        with tarfile.open(FLEURS_TRAIN, "r:gz") as tar:
            for k, m in enumerate(tar, 1):
                f = tar.extractfile(m) if m.isfile() and m.name.endswith(".wav") else None
                if f is not None:
                    ra.append(
                        dt.chuoi(sang_int16(doc_wav(io.BytesIO(f.read()))))[BO_DAU::4].astype(np.float16)
                    )
                if k % 500 == 0:
                    print(f"   FLEURS train: {k} file", flush=True)
        np.save(KHO_FLEURS, np.concatenate(ra))
    return np.load(KHO_FLEURS, mmap_mode="r")


def train_lo(
    X_duong: np.ndarray, kho_am: list[tuple[np.ndarray, float]], rng: np.random.Generator, so_vong: int = 6
):
    """MLP train theo LÔ (partial_fit) để không nạp cả kho âm vào RAM: mỗi lô 4096 mẫu = 1/5 dương + 4/5 âm,
    phần âm chia cho các kho theo tỉ lệ cho trước (kho lớn đọc từng lô từ memmap float16)."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def boc(kho: np.ndarray, n: int) -> np.ndarray:
        return np.asarray(kho[np.sort(rng.integers(len(kho), size=n))], np.float32).reshape(n, -1)

    chuan = StandardScaler().fit(np.concatenate([X_duong] + [boc(k, 20000) for k, _ in kho_am]))
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, learning_rate_init=1e-3, random_state=0)
    lo, n_duong = 4096, 4096 // 5
    n_kho = [int((lo - n_duong) * ty) for _, ty in kho_am]
    n_kho[0] += lo - n_duong - sum(n_kho)
    so_buoc = sum(len(k) for k, _ in kho_am) // (lo - n_duong) + 1
    for v in range(so_vong):
        for _ in range(so_buoc):
            Xb = np.concatenate(
                [X_duong[rng.integers(len(X_duong), size=n_duong)]]
                + [boc(k, n) for (k, _), n in zip(kho_am, n_kho, strict=True)]
            )
            yb = np.concatenate([np.ones(n_duong), np.zeros(lo - n_duong)])
            mlp.partial_fit(chuan.transform(Xb), yb, classes=[0, 1])
        print(f"   vòng {v + 1}/{so_vong} xong, loss {mlp.loss_:.4f}", flush=True)
    return make_pipeline(chuan, mlp)


def chon_nguong(
    diem_may: list[np.ndarray], diem_that: list[np.ndarray], diem_am: np.ndarray, gio_am: float
) -> tuple[dict, list[dict]]:
    """Luật chọn (k, ngưỡng) trên VAL — bản 2 (khoá trong PLAN trước khi train): k ∈ {1..4} khung liền ≥
    ngưỡng, lưới 0,05–0,99; giữ cặp báo nhầm ≤ 1 lần / giờ; lấy bắt cao nhất trên GIỌNG THẬT val; hoà -> bắt
    giọng VieNeu giữ riêng cao hơn -> k nhỏ hơn -> ngưỡng thấp hơn. (Bản 1 chỉ có giọng máy.)"""

    def ty_le(ds: list[np.ndarray], t: float, k: int) -> float:
        return float(np.mean([bool(dem_bao(s, t, k)) for s in ds])) if ds else 0.0

    bang = []
    for k in (1, 2, 3, 4):
        for t in np.round(np.arange(0.05, 0.995, 0.01), 2):
            bao = len(dem_bao(diem_am, t, k))
            bang.append(
                {
                    "lien": k,
                    "nguong": float(t),
                    "bao_nham": bao,
                    "bao_nham_gio": bao / gio_am,
                    "bat_that": ty_le(diem_that, t, k),
                    "bat": ty_le(diem_may, t, k),
                }
            )
    dat = [b for b in bang if b["bao_nham_gio"] <= 1.0]
    if not dat:
        return max(bang, key=lambda b: (-b["bao_nham"], b["bat_that"])), bang
    return min(dat, key=lambda b: (-b["bat_that"], -b["bat"], b["lien"], b["nguong"])), bang


def ghi_that(tach: str) -> list[np.ndarray]:
    """Bản ghi "hey nexa" giọng thật để train (tách train / val theo cột `tach` đã khoá)."""
    man = [json.loads(x) for x in open(GHI_TRAIN / "manifest.jsonl", encoding="utf-8")]
    return [doc_wav(GHI_TRAIN / r["tep"]) for r in man if r["tach"] == tach]


def main() -> None:
    import joblib

    t0 = time.time()
    rng = np.random.default_rng(SEED)
    man = [json.loads(x) for x in open(TTS / "manifest.jsonl", encoding="utf-8")]
    am = {d["id"]: doc_wav(TTS / d["tep"]) for d in man if d["tach"] in ("train", "val")}
    fl = doc_fleurs(FLEURS_DEV)
    ten = sorted(fl)
    random.Random(SEED).shuffle(ten)
    n_val = int(0.3 * len(ten))
    fl_val, fl_train = [fl[t] for t in ten[:n_val]], [fl[t] for t in ten[n_val:]]
    print(
        f"nạp: {len(am)} file TTS, FLEURS dev {len(ten)} câu ({n_val} val) — {time.time() - t0:.0f}s",
        flush=True,
    )

    def lay(loai: str, tach: str) -> list[np.ndarray]:
        return [am[d["id"]] for d in man if d["loai"] == loai and d["tach"] == tach]

    dt = DacTrung()
    if CACHE.exists():  # đặc trưng tính một lần (~10 phút), các lần train sau dùng lại
        c = np.load(CACHE)
        X_duong, X_am, V_duong, V_moc, V_dong, T_duong, T_moc = (
            c[k] for k in ("X_duong", "X_am", "V_duong", "V_moc", "V_dong", "T_duong", "T_moc")
        )
    else:
        bo_b = [doc_wav(t) for t in sorted(BO_B.glob("*.wav"))]
        X_duong = np.concatenate(
            [
                dac_trung_duong(dt, lay("duong", "train"), 8, rng, fl_train),
                dac_trung_duong(dt, ghi_that("train"), 20, rng, fl_train),  # giọng thật × 20 biến đổi
            ]
        )
        X_am = np.concatenate(
            [
                dac_trung_am(dt, bo_b, 3, rng, fl_train),  # giọng người dùng nói lệnh xe (bộ B)
                dac_trung_am(dt, lay("gan_giong", "train"), 3, rng, fl_train),
                dac_trung_am(dt, lay("lenh", "train"), 2, rng, fl_train, buoc=2),
                dac_trung_am(dt, fl_train, 1, rng, fl_train, buoc=2),
                dac_trung_am(
                    dt,
                    [
                        on(rng.choice(["trang", "hong", "tram"]), 3 * SR, rng, []) * rng.uniform(0.001, 0.3)
                        for _ in range(150)
                    ],
                    1,
                    rng,
                    fl_train,
                ),
            ]
        )
        rng_val = np.random.default_rng(SEED + 1)
        V = dac_trung_bat(dt, lay("duong", "val"), rng_val, fl_val)
        V_duong, V_moc = np.concatenate(V), np.cumsum([0] + [len(v) for v in V])
        T = dac_trung_bat(dt, ghi_that("val"), rng_val, fl_val, so_lan=4)  # giọng thật VAL
        T_duong, T_moc = np.concatenate(T), np.cumsum([0] + [len(v) for v in T])
        V_dong = dt.chuoi(dong_am(fl_val + lay("gan_giong", "val") + lay("lenh", "val"), rng_val))[BO_DAU:]
        np.savez(
            CACHE,
            X_duong=X_duong,
            X_am=X_am,
            V_duong=V_duong,
            V_moc=V_moc,
            V_dong=V_dong,
            T_duong=T_duong,
            T_moc=T_moc,
        )
    acav = np.load(ACAV, mmap_mode="r").reshape(-1, 1536)
    fl_kho = kho_fleurs_train(dt)
    print(
        f"đặc trưng: dương {len(X_duong)} · âm {len(X_am)} · ACAV100M {len(acav)}"
        f" · FLEURS train {len(fl_kho)} — {time.time() - t0:.0f}s",
        flush=True,
    )

    # âm mỗi lô: 30% kho của mình (câu gần giống, câu lệnh, FLEURS dev, ồn) · 35% ACAV100M · 35% FLEURS train
    mo_hinh = train_lo(X_duong, [(X_am, 0.3), (acav, 0.35), (fl_kho, 0.35)], rng)
    print(f"train xong — {time.time() - t0:.0f}s", flush=True)

    # ── VAL: chọn (k, ngưỡng) ──
    s_val, s_that = diem(mo_hinh, V_duong), diem(mo_hinh, T_duong)
    lan_thu = [s_val[a:b] for a, b in zip(V_moc[:-1], V_moc[1:], strict=True)]
    lan_that = [s_that[a:b] for a, b in zip(T_moc[:-1], T_moc[1:], strict=True)]
    gio = (len(V_dong) + BO_DAU) * 1280 / SR / 3600
    d_am = diem(mo_hinh, V_dong)
    chon, bang = chon_nguong(lan_thu, lan_that, d_am, gio)
    print(
        f"VAL: dòng âm {gio * 60:.1f} phút · CHỌN {chon['lien']} khung liền ≥ {chon['nguong']}"
        f" · bắt giọng thật {chon['bat_that']:.1%} ({len(lan_that)} lần) · giọng máy {chon['bat']:.1%}"
        f" ({len(lan_thu)} lần) · báo nhầm {chon['bao_nham']} ({chon['bao_nham_gio']:.2f}/giờ)"
    )
    for b in bang:
        if round(b["nguong"] * 100) % 10 == 0 or b["nguong"] == 0.95:
            print(
                f"   k={b['lien']} ngưỡng {b['nguong']:.2f}: thật {b['bat_that']:.1%} · máy {b['bat']:.1%}"
                f" · báo nhầm {b['bao_nham']}"
            )
    nguong = chon["nguong"]
    joblib.dump(
        {
            "mo_hinh": mo_hinh,
            "nguong": nguong,
            "lien": chon["lien"],
            "thong_tin": {
                "duong": len(X_duong),
                "am": len(X_am),
                "am_acav100m": len(acav),
                "am_fleurs_train": len(fl_kho),
                "ban": 2,
                "val_phut_am": gio * 60,
                "val": chon,
                "seed": SEED,
            },
        },
        MO_HINH,
    )
    Path("runs/wake").mkdir(parents=True, exist_ok=True)
    Path("runs/wake/val.json").write_text(
        json.dumps({"chon": chon, "bang": bang}, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"lưu {MO_HINH} — tổng {time.time() - t0:.0f}s")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
