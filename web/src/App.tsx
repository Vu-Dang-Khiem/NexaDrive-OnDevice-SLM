/**
 * Giao diện demo — hướng B "Sáng tối giản" (người dùng chọn 2026-10-06, PLAN.md Phase 7):
 *   trên: tên + kết nối + công tắc demo (Đỗ xe / Đang chạy, Khoá trẻ em — luật an toàn đọc hai cờ này);
 *   giữa: xe 3D dựng bằng code (xe/XeLowPoly) — bộ phận chuyển động theo trạng thái xe, camera bay tới bộ phận;
 *   dưới: thanh hội thoại — micro (giọng nói nối ở Phase 6), câu vừa nói + câu trả lời, Có / Không khi hỏi lại,
 *         ô gõ lệnh, dòng "model đã hiểu gì" (lời gọi hàm, thời gian) cho hội đồng xem.
 */

import { type CSSProperties, type FormEvent, useEffect, useRef, useState } from "react";
import sceneActions from "./sceneActions.json";
import { NexaSocket, type ConnState, type ServerEvent } from "./ws";
import { CanhXe } from "./xe/CanhXe";
import type { TrangThaiXe } from "./xe/XeLowPoly";

type Xe = TrangThaiXe & {
  khoa_tre_em?: boolean;
  fuel_percent?: number;
  battery_percent?: number;
  ac_on?: boolean;
  ac_temperature?: number;
  doors_locked?: boolean;
  volume?: number;
};
type Luot = { nguoi: string; troLy: string; hoiLai: null | "co_khong" | "chon"; goiHam: string; ms?: number };

const ACTIONS = (sceneActions as { actions: Record<string, { camera?: string }> }).actions;
const TEN_KINH: Record<string, string> = {
  front_left: "kính ghế lái", front_right: "kính ghế phụ", rear_left: "kính sau trái", rear_right: "kính sau phải",
};
const MAU = { nen: "#eef1f4", the: "#ffffff", vien: "#d5dbe1", chu: "#0f172a", phu: "#475569", nhan: "#1d4ed8" };

/** Góc camera cho một lời gọi: theo schema, riêng kính / cửa bên phụ thì nhìn từ bên phải xe. */
function gocCho(ten: string, args: Record<string, unknown> = {}): string {
  const w = String(args.window ?? args.door ?? "");
  if (ten === "control_window" && (w.endsWith("right") || w === "passenger")) return "passenger_door";
  return ACTIONS[ten]?.camera ?? "overview";
}

/** Câu trạng thái ngắn của bộ phận vừa được nhắc tới (ô nổi góc phải khung 3D). */
function moTaBoPhan(ten: string, xe: Xe): string {
  if (ten === "control_window") {
    const mo = Object.entries(xe.windows ?? {}).filter(([, p]) => p > 0);
    return mo.length ? "Đang mở: " + mo.map(([k, p]) => `${TEN_KINH[k] ?? k} ${p}%`).join(", ") : "Các kính đều đóng";
  }
  if (ten.includes("trunk")) return xe.trunk_open ? "Cốp đang mở" : "Cốp đang đóng";
  if (ten === "operate_sunroof") return xe.sunroof_percent ? `Cửa sổ trời mở ${xe.sunroof_percent}%` : "Cửa sổ trời đóng";
  if (ten === "open_charge_port") return xe.charge_port_open ? "Nắp sạc đang mở" : "Nắp sạc đóng";
  if (ten === "open_fuel_flap") return xe.fuel_flap_open ? "Nắp xăng đang mở" : "Nắp xăng đóng";
  if (ten.includes("doors")) return xe.doors_locked ? "Cửa đã khoá" : "Cửa đang mở khoá";
  if (ten === "set_hazard_lights") return xe.hazards_on ? "Đèn khẩn cấp đang bật" : "Đèn khẩn cấp tắt";
  if (ten === "set_ac") return xe.ac_on ? `Điều hoà ${xe.ac_temperature}°` : "Điều hoà tắt";
  return "";
}

export default function App() {
  const [ket, setKet] = useState<ConnState>("connecting");
  const [xe, setXe] = useState<Xe>({});
  const [luot, setLuot] = useState<Luot[]>([]);
  const [go, setGo] = useState("");
  const [cam, setCam] = useState({ goc: "overview", lan: 0, ham: "" });
  const sock = useRef<NexaSocket | null>(null);
  const cauVuaGui = useRef("");

  useEffect(() => {
    const s = new NexaSocket(
      (e: ServerEvent) => {
        if (e.type === "state") return setXe(e.vehicle as Xe);
        if (e.type === "error") return;
        const nguoi = cauVuaGui.current;
        if (e.type === "assistant_turn") {
          const c = e.calls[0];
          if (c) setCam((v) => ({ goc: gocCho(c.name, c.arguments), lan: v.lan + 1, ham: c.name }));
          setLuot((ds) => [...ds, { nguoi, troLy: e.text, hoiLai: null, goiHam: e.calls.map((x) => x.nexa).join("  ") || e.loai, ms: e.latency_ms }]);
        } else {
          setCam((v) => ({ goc: gocCho(e.function), lan: v.lan + 1, ham: e.function }));
          setLuot((ds) => [...ds, { nguoi, troLy: e.text, hoiLai: e.ly_do === "chon" ? "chon" : e.ly_do === "thieu_tham_so" ? null : "co_khong", goiHam: `hỏi lại (${e.ly_do}) · ${e.functions.join(", ")}` }]);
        }
      },
      (st) => setKet(st),
    );
    s.connect();
    sock.current = s;
    return () => s.close();
  }, []);

  const gui = (cau: string) => {
    const c = cau.trim();
    if (!c) return;
    cauVuaGui.current = c;
    if (!sock.current?.send(c)) setLuot((ds) => [...ds, { nguoi: c, troLy: "Chưa kết nối được backend.", hoiLai: null, goiHam: "" }]);
  };
  const onGo = (e: FormEvent) => {
    e.preventDefault();
    gui(go);
    setGo("");
  };

  const cuoi = luot[luot.length - 1];
  const chay = Boolean(xe.dang_chay);
  const moTa = cam.ham ? moTaBoPhan(cam.ham, xe) : "";
  const cham = ket === "open" ? "#16a34a" : ket === "connecting" ? "#ca8a04" : "#dc2626";

  return (
    <div style={S.trang}>
      <header style={S.dau}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 12, flexWrap: "wrap" }}>
          <div style={{ fontWeight: 700, fontSize: 22 }}>Nexa Drive</div>
          <div style={{ fontSize: 14, color: MAU.phu }}>Trợ lý xe · chạy trên máy</div>
          <div style={{ fontSize: 13, color: MAU.phu, display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: cham, display: "inline-block" }} />
            {ket === "open" ? "đã kết nối" : ket === "connecting" ? "đang kết nối…" : "mất kết nối"}
          </div>
        </div>
        <div role="group" aria-label="Công tắc demo" style={S.nhomCongTac}>
          <button type="button" aria-pressed={!chay} style={chay ? S.nut : S.nutChon} onClick={() => sock.current?.setVehicle({ dang_chay: false })}>Đỗ xe</button>
          <button type="button" aria-pressed={chay} style={chay ? S.nutChon : S.nut} onClick={() => sock.current?.setVehicle({ dang_chay: true })}>Đang chạy</button>
          <button type="button" aria-pressed={Boolean(xe.khoa_tre_em)} style={xe.khoa_tre_em ? S.nutChon : S.nut} onClick={() => sock.current?.setVehicle({ khoa_tre_em: !xe.khoa_tre_em })}>Khoá trẻ em</button>
        </div>
      </header>

      <section aria-label="Xe 3D" style={S.khungXe}>
        {/* canvas phủ kín khung (khung flex không có chiều cao cố định -> height: 100% của Canvas không ăn) */}
        <div style={{ position: "absolute", inset: 0 }}>
          <CanhXe s={xe} goc={cam.goc} lan={cam.lan} />
        </div>
        <div style={S.hud}>
          <O nhan="Xăng" giaTri={xe.fuel_percent !== undefined ? `${xe.fuel_percent}%` : "—"} />
          <O nhan="Pin" giaTri={xe.battery_percent !== undefined ? `${xe.battery_percent}%` : "—"} />
          <O nhan="Điều hoà" giaTri={xe.ac_on ? `Bật · ${xe.ac_temperature}°` : `Tắt · ${xe.ac_temperature ?? "—"}°`} />
          <O nhan="Cửa" giaTri={xe.doors_locked ? "Đã khoá" : "Mở khoá"} />
          <O nhan="Âm lượng" giaTri={String(xe.volume ?? "—")} />
        </div>
        {moTa && <div style={S.oNoi}>{moTa}</div>}
        {chay && <div style={S.dangChay}>● Đang chạy</div>}
      </section>

      <section aria-label="Hội thoại" style={S.thanh}>
        <button type="button" aria-label="Giọng nói — sẽ nối ở bước giọng nói" title="Giọng nói sẽ nối ở bước sau" disabled style={S.mic}>
          <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#ffffff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <rect x="9" y="3" width="6" height="11" rx="3" /><path d="M5 11a7 7 0 0 0 14 0" /><path d="M12 18v3" />
          </svg>
        </button>
        <div style={{ flex: "1 1 420px", minWidth: 0, display: "flex", flexDirection: "column", gap: 6 }}>
          {cuoi ? (
            <>
              <div style={{ fontSize: 15, color: MAU.phu }}>Bạn: «{cuoi.nguoi}»</div>
              <div aria-live="polite" style={{ fontSize: 20, fontWeight: 600, lineHeight: 1.4 }}>{cuoi.troLy}</div>
            </>
          ) : (
            <div style={{ fontSize: 16, color: MAU.phu, lineHeight: 1.5 }}>
              Thử gõ: «mở cốp», «hạ kính ghế phụ một nửa», «nóng quá», «bật đèn khẩn cấp», «mở cửa sổ trời». Bật «Đang chạy» rồi «mở cốp» để thấy luật an toàn.
            </div>
          )}
        </div>
        {cuoi?.hoiLai === "co_khong" && (
          <div style={{ display: "flex", gap: 10 }}>
            <button type="button" style={S.nutCo} onClick={() => gui("có")}>Có</button>
            <button type="button" style={S.nutKhong} onClick={() => gui("không")}>Không</button>
          </div>
        )}
        {cuoi?.hoiLai === "chon" && (
          <div style={{ display: "flex", gap: 10 }}>
            <button type="button" style={S.nutCo} onClick={() => gui("cái đầu")}>Cái đầu</button>
            <button type="button" style={S.nutKhong} onClick={() => gui("cái thứ hai")}>Cái thứ hai</button>
          </div>
        )}
        <div style={S.hangDuoi}>
          <form onSubmit={onGo} style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
            <label htmlFor="go-lenh" style={{ fontSize: 14, color: MAU.phu }}>Gõ lệnh</label>
            <input id="go-lenh" value={go} onChange={(e) => setGo(e.target.value)} placeholder="vd: bật điều hoà 22 độ" style={S.o} />
            <button type="submit" style={S.nutGui}>Gửi</button>
          </form>
          {cuoi?.goiHam && (
            <div style={S.goiHam}>{cuoi.goiHam}{cuoi.ms !== undefined ? ` · ${Math.round(cuoi.ms)} ms` : ""}</div>
          )}
        </div>
      </section>
    </div>
  );
}

function O({ nhan, giaTri }: { nhan: string; giaTri: string }) {
  return (
    <div style={{ background: "rgba(241,245,249,0.92)", borderRadius: 12, padding: "10px 14px" }}>
      <div style={{ fontSize: 12, color: MAU.phu }}>{nhan}</div>
      <div style={{ fontSize: 20, fontWeight: 700 }}>{giaTri}</div>
    </div>
  );
}

const nutCoBan: CSSProperties = { minHeight: 44, padding: "0 16px", border: 0, borderRadius: 999, fontSize: 14, cursor: "pointer", fontFamily: "inherit" };
const S: Record<string, CSSProperties> = {
  trang: { minHeight: "100vh", boxSizing: "border-box", background: MAU.nen, color: MAU.chu, fontFamily: "'Be Vietnam Pro', system-ui, sans-serif", display: "flex", flexDirection: "column", padding: "20px 28px", gap: 18 },
  dau: { display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "space-between", gap: 12 },
  nhomCongTac: { display: "flex", flexWrap: "wrap", gap: 6, background: MAU.the, border: `1px solid ${MAU.vien}`, borderRadius: 999, padding: 4 },
  nut: { ...nutCoBan, background: "transparent", color: "#334155" },
  nutChon: { ...nutCoBan, background: MAU.chu, color: "#ffffff", fontWeight: 600 },
  khungXe: { flex: 1, position: "relative", minHeight: 480, background: MAU.the, border: `1px solid ${MAU.vien}`, borderRadius: 24, overflow: "hidden" },
  hud: { position: "absolute", top: 18, left: 22, display: "flex", flexWrap: "wrap", gap: 10, pointerEvents: "none" },
  oNoi: { position: "absolute", right: 22, top: 18, background: "#eff6ff", border: "1px solid #bfdbfe", color: "#1e3a8a", borderRadius: 12, padding: "10px 14px", fontSize: 14, pointerEvents: "none" },
  dangChay: { position: "absolute", right: 22, bottom: 18, background: "#0f172a", color: "#ffffff", borderRadius: 999, padding: "8px 14px", fontSize: 13, fontWeight: 600, pointerEvents: "none" },
  thanh: { display: "flex", flexWrap: "wrap", alignItems: "center", gap: 18, background: MAU.the, border: `1px solid ${MAU.vien}`, borderRadius: 24, padding: "18px 22px" },
  mic: { width: 76, height: 76, flex: "none", borderRadius: "50%", border: 0, background: MAU.nhan, opacity: 0.45, display: "flex", alignItems: "center", justifyContent: "center", cursor: "not-allowed" },
  nutCo: { minHeight: 48, minWidth: 96, border: 0, borderRadius: 12, background: MAU.nhan, color: "#ffffff", fontSize: 16, fontWeight: 700, cursor: "pointer", fontFamily: "inherit" },
  nutKhong: { minHeight: 48, minWidth: 96, border: "1px solid #cbd5e1", borderRadius: 12, background: "#ffffff", color: MAU.chu, fontSize: 16, cursor: "pointer", fontFamily: "inherit" },
  hangDuoi: { flexBasis: "100%", display: "flex", flexWrap: "wrap", justifyContent: "space-between", gap: 10, borderTop: "1px solid #e2e8f0", paddingTop: 12 },
  o: { minHeight: 40, width: 300, maxWidth: "70vw", border: "1px solid #cbd5e1", borderRadius: 10, padding: "0 12px", fontFamily: "inherit", fontSize: 14 },
  nutGui: { minHeight: 40, padding: "0 16px", border: 0, borderRadius: 10, background: MAU.chu, color: "#ffffff", fontSize: 14, cursor: "pointer", fontFamily: "inherit" },
  goiHam: { fontFamily: "'JetBrains Mono', ui-monospace, monospace", fontSize: 12.5, color: MAU.phu, alignSelf: "center", overflowWrap: "anywhere" },
};
