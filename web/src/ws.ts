/**
 * Client WebSocket tới backend Nexa Drive, có tự kết nối lại.
 *
 * Phase 5: mang `assistant_turn` (câu trả lời + lời gọi hàm + ui_action cho scene 3D),
 * `clarify` (agent hỏi lại), `state` (trạng thái xe sau mỗi thay đổi). Hợp đồng: serve/contracts.py.
 */

export type CallPayload = { name: string; arguments: Record<string, unknown>; nexa: string };

export type ServerEvent =
  | {
      type: "assistant_turn";
      text: string;
      lang: string;
      loai: "thuc_thi" | "tu_choi" | "chua_hieu" | "huy" | "tro_chuyen";
      calls: CallPayload[];
      results: unknown[];
      ui_actions: unknown[];
      latency_ms?: number;
    }
  | {
      type: "clarify";
      text: string;
      ly_do: "de_xuat" | "thieu_tham_so" | "an_toan" | "chon";
      function: string;
      functions: string[];
      missing: string[];
    }
  | { type: "state"; vehicle: Record<string, unknown> }
  | { type: "error"; message: string; detail?: string };

export type ConnState = "connecting" | "open" | "closed";

const WS_URL = "ws://localhost:8000/ws";
const RETRY_MS = 2000;

export class NexaSocket {
  private ws: WebSocket | null = null;
  private retryTimer: number | null = null;
  private closedByUser = false;

  constructor(
    private onEvent: (e: ServerEvent) => void,
    private onStateChange: (s: ConnState) => void,
  ) {}

  connect(): void {
    this.closedByUser = false;
    this.onStateChange("connecting");
    const ws = new WebSocket(WS_URL);
    this.ws = ws;

    ws.onopen = () => this.onStateChange("open");

    ws.onmessage = (ev) => {
      try {
        this.onEvent(JSON.parse(ev.data) as ServerEvent);
      } catch {
        this.onEvent({ type: "error", message: "không parse được dữ liệu từ server" });
      }
    };

    ws.onclose = () => {
      this.onStateChange("closed");
      if (!this.closedByUser) this.scheduleRetry();
    };

    ws.onerror = () => ws.close();
  }

  private scheduleRetry(): void {
    if (this.retryTimer !== null) return;
    this.retryTimer = window.setTimeout(() => {
      this.retryTimer = null;
      this.connect();
    }, RETRY_MS);
  }

  /** `nguon`: "nut" = bấm mic / gõ (tài xế) · "wake_word" = gọi «hey nexa» (ai cũng nói được -> luật an toàn chặt hơn). */
  send(text: string, lang: "vi" | "en" | "auto" = "vi", nguon: "nut" | "wake_word" = "nut"): boolean {
    if (this.ws?.readyState !== WebSocket.OPEN) return false;
    this.ws.send(JSON.stringify({ type: "user_message", text, lang, nguon }));
    return true;
  }

  /** Công tắc demo: luật an toàn đọc hai cờ này. */
  setVehicle(flags: { dang_chay?: boolean; khoa_tre_em?: boolean }): boolean {
    if (this.ws?.readyState !== WebSocket.OPEN) return false;
    this.ws.send(JSON.stringify({ type: "dieu_khien_xe", ...flags }));
    return true;
  }

  close(): void {
    this.closedByUser = true;
    if (this.retryTimer !== null) {
      clearTimeout(this.retryTimer);
      this.retryTimer = null;
    }
    this.ws?.close();
  }
}
