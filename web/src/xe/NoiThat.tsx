/**
 * Nội thất xe (người dùng 2026-10-06: "bật điều hoà nên thiết kế giao diện trong xe nữa cho đẹp"): táp-lô, vô-lăng,
 * màn hình giữa (nhiệt độ điều hoà, mức quạt, bài nhạc, điểm đến), 4 ghế, bệ tỳ tay, CỬA GIÓ CÓ LUỒNG KHÍ chuyển
 * động khi bật điều hoà (xanh = làm mát, cam = sưởi; quạt càng mạnh luồng càng nhanh), vạch sấy kính sáng khi bật.
 * Camera vào khoang ở các lệnh điều hoà / quạt / sấy / âm lượng / nhạc / dẫn đường (CanhXe: góc "dashboard" và
 * "center_console").
 */

import { Html, RoundedBox } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";

export type TrangThaiKhoang = {
  ac_on?: boolean;
  ac_temperature?: number;
  ac_zone?: string;
  ac_mode?: string;
  fan_level?: number;
  defrost_on?: boolean;
  defrost_zone?: string;
  media_playing?: boolean;
  now_playing?: string;
  volume?: number;
  destination?: string | null;
};

const DA = "#26292e"; // da ghế / táp-lô
const NHUA = "#3a3f46";
const SO_HAT = 26; // số "hạt gió" mỗi cửa gió

/** Một cửa gió + luồng khí: các vệt mảnh bay từ miệng gió về phía ghế, mờ dần. */
function CuaGio({ z, bat, mau, toc }: { z: number; bat: boolean; mau: string; toc: number }) {
  const hat = useRef<THREE.InstancedMesh>(null);
  const tuoi = useMemo(() => Array.from({ length: SO_HAT }, (_, i) => i / SO_HAT), []);
  const lech = useMemo(() => Array.from({ length: SO_HAT }, () => [Math.random() - 0.5, Math.random() - 0.5]), []);
  const tam = useMemo(() => new THREE.Object3D(), []);
  const doMo = useRef(0);
  useFrame((_, dt) => {
    const m = hat.current;
    if (!m) return;
    doMo.current += ((bat ? 1 : 0) - doMo.current) * (1 - Math.exp(-3 * dt));
    for (let i = 0; i < SO_HAT; i++) {
      tuoi[i] = (tuoi[i] + dt * (0.25 + 0.12 * toc)) % 1;
      const a = tuoi[i];
      tam.position.set(-a * 1.1, -a * 0.25 + lech[i][1] * 0.12 * a, lech[i][0] * 0.22 * a);
      const co = (1 - a) * doMo.current;
      tam.scale.set(0.06 + a * 0.07, 0.006 * co + 0.0001, 0.006 * co + 0.0001); // vệt gió mảnh

      tam.updateMatrix();
      m.setMatrixAt(i, tam.matrix);
    }
    m.instanceMatrix.needsUpdate = true;
    (m.material as THREE.MeshBasicMaterial).opacity = 0.42 * doMo.current;
  });
  return (
    <group position={[0.92, 1.0, z]}>
      <mesh position={[0.02, 0, 0]}>
        <boxGeometry args={[0.03, 0.06, 0.16]} />
        <meshStandardMaterial color="#14171b" metalness={0.5} roughness={0.4} />
      </mesh>
      <instancedMesh ref={hat} args={[undefined, undefined, SO_HAT]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshBasicMaterial color={mau} transparent opacity={0} depthWrite={false} />
      </instancedMesh>
    </group>
  );
}

function Ghe({ x, z, truoc }: { x: number; z: number; truoc: boolean }) {
  return (
    <group position={[x, 0, z]}>
      <RoundedBox args={[0.5, 0.12, 0.5]} radius={0.05} position={[0, 0.55, 0]} castShadow>
        <meshStandardMaterial color={DA} roughness={0.7} />
      </RoundedBox>
      <RoundedBox args={[0.12, 0.62, 0.5]} radius={0.05} position={[-0.24, 0.9, 0]} rotation={[0, 0, 0.18]}>
        <meshStandardMaterial color={DA} roughness={0.7} />
      </RoundedBox>
      {truoc && (
        <RoundedBox args={[0.1, 0.16, 0.26]} radius={0.04} position={[-0.3, 1.3, 0]}>
          <meshStandardMaterial color={DA} roughness={0.7} />
        </RoundedBox>
      )}
    </group>
  );
}

/** Màn hình giữa: React DOM gắn vào mặt phẳng 3D (drei Html transform) — chữ sắc, đổi theo trạng thái xe. */
/** Gió ấm hay mát: theo chế độ nếu có (sưởi / làm lạnh), không thì đoán theo nhiệt độ đặt. */
function gioAm(s: TrangThaiKhoang): boolean {
  if (s.ac_mode === "heat") return true;
  if (s.ac_mode === "cool") return false;
  return (s.ac_temperature ?? 24) >= 25;
}

function ManHinh({ s }: { s: TrangThaiKhoang }) {
  const nong = gioAm(s);
  const vung = { driver: "ghế lái", passenger: "ghế phụ", rear: "hàng sau", all: "toàn xe" }[s.ac_zone ?? "all"] ?? "toàn xe";
  const cheDo = { heat: " · sưởi", cool: " · làm lạnh" }[s.ac_mode ?? ""] ?? "";
  return (
    <group position={[0.97, 1.12, 0.05]} rotation={[0, -Math.PI / 2, 0]}>
      <mesh>
        <planeGeometry args={[0.42, 0.26]} />
        <meshStandardMaterial color="#05070a" roughness={0.2} metalness={0.3} />
      </mesh>
      <Html transform occlude={false} distanceFactor={0.36} position={[0, 0, 0.002]} style={{ pointerEvents: "none" }}>
        <div style={{ width: 420, height: 260, boxSizing: "border-box", padding: 22, background: "#0b1220", color: "#e2e8f0",
          borderRadius: 14, fontFamily: "'Be Vietnam Pro', sans-serif", display: "flex", flexDirection: "column", gap: 10 }}>
          <div style={{ fontSize: 15, color: "#94a3b8" }}>Điều hoà · {vung}{cheDo}</div>
          <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
            <div style={{ fontSize: 64, fontWeight: 700, color: s.ac_on ? (nong ? "#fb923c" : "#38bdf8") : "#64748b" }}>
              {s.ac_on ? `${s.ac_temperature}°` : "Tắt"}
            </div>
            <div style={{ fontSize: 16, color: "#cbd5e1" }}>Quạt {s.fan_level ?? 0}/7{s.defrost_on ? " · sấy kính" : ""}</div>
          </div>
          <div style={{ display: "flex", gap: 4 }}>
            {Array.from({ length: 7 }, (_, i) => (
              <div key={i} style={{ flex: 1, height: 8, borderRadius: 4, background: i < (s.fan_level ?? 0) ? "#38bdf8" : "#1e293b" }} />
            ))}
          </div>
          <div style={{ marginTop: "auto", fontSize: 15, color: "#cbd5e1", display: "flex", justifyContent: "space-between", gap: 12 }}>
            <span>{s.media_playing ? `♪ ${s.now_playing}` : "Không phát nhạc"}</span>
            <span>Âm lượng {s.volume ?? 0}</span>
          </div>
          {s.destination && <div style={{ fontSize: 15, color: "#93c5fd" }}>→ {s.destination}</div>}
        </div>
      </Html>
    </group>
  );
}

/** `trongXe`: camera đang ở trong khoang — màn hình (DOM gắn 3D) chỉ hiện lúc đó, không thì lộ xuyên thân xe. */
export function NoiThat({ s, trongXe }: { s: TrangThaiKhoang; trongXe: boolean }) {
  const mauGio = gioAm(s) ? "#fb923c" : "#7dd3fc";
  // MỘT vật liệu dùng chung cho mọi vạch sấy của mỗi kính -> đổi độ sáng một lần là cả dải sáng theo
  const say = useMemo(() => new THREE.MeshBasicMaterial({ color: "#fdba74", transparent: true, opacity: 0, side: THREE.DoubleSide }), []);
  const sayHau = useMemo(() => new THREE.MeshBasicMaterial({ color: "#fdba74", transparent: true, opacity: 0, side: THREE.DoubleSide }), []);
  useFrame((_, dt) => {
    const sayTruoc = s.defrost_on && s.defrost_zone !== "rear";
    const saySau = s.defrost_on && (s.defrost_zone === "rear" || s.defrost_zone === "both");
    say.opacity += ((sayTruoc ? 0.7 : 0) - say.opacity) * (1 - Math.exp(-3 * dt));
    sayHau.opacity += ((saySau ? 0.7 : 0) - sayHau.opacity) * (1 - Math.exp(-3 * dt));
  });
  return (
    <group>
      {/* đèn trần nội thất: trần quay mặt xuống nên đèn từ trên không chiếu tới — thiếu đèn này khoang tối đen */}
      <pointLight position={[0.1, 1.3, 0]} intensity={trongXe ? 2.2 : 0.8} distance={3.2} decay={1.6} color="#fff4e6" />
      {/* sàn + táp-lô + bệ tỳ tay */}
      <mesh position={[-0.2, 0.36, 0]}>
        <boxGeometry args={[2.9, 0.04, 1.76]} />
        <meshStandardMaterial color="#1c1f24" roughness={0.95} />
      </mesh>
      <RoundedBox args={[0.42, 0.26, 1.74]} radius={0.08} position={[1.05, 0.93, 0]}>
        <meshStandardMaterial color={NHUA} roughness={0.75} />
      </RoundedBox>
      <RoundedBox args={[0.9, 0.22, 0.24]} radius={0.06} position={[0.45, 0.62, 0]}>
        <meshStandardMaterial color={DA} roughness={0.7} />
      </RoundedBox>
      {/* vô-lăng bên trái (ghế lái = −Z) */}
      <group position={[0.72, 1.05, -0.42]} rotation={[0, 0, 0.45]}>
        <mesh rotation={[0, Math.PI / 2, 0]}>
          <torusGeometry args={[0.17, 0.022, 12, 40]} />
          <meshStandardMaterial color="#111317" roughness={0.6} />
        </mesh>
        <mesh rotation={[0, Math.PI / 2, 0]}>
          <cylinderGeometry args={[0.05, 0.05, 0.04, 20]} />
          <meshStandardMaterial color="#2b2f36" metalness={0.5} roughness={0.4} />
        </mesh>
      </group>
      {trongXe ? (
        <ManHinh s={s} />
      ) : (
        <mesh position={[0.97, 1.12, 0.05]} rotation={[0, -Math.PI / 2, 0]}>
          <planeGeometry args={[0.42, 0.26]} />
          <meshStandardMaterial color="#05070a" roughness={0.2} metalness={0.3} />
        </mesh>
      )}
      {/* 4 cửa gió trên táp-lô */}
      {[-0.7, -0.15, 0.15, 0.7].map((z) => (
        <CuaGio key={z} z={z} bat={Boolean(s.ac_on)} mau={mauGio} toc={s.fan_level ?? 2} />
      ))}
      <Ghe x={0.1} z={-0.42} truoc />
      <Ghe x={0.1} z={0.42} truoc />
      <Ghe x={-0.85} z={-0.42} truoc={false} />
      <Ghe x={-0.85} z={0.42} truoc={false} />
      {/* vạch sấy kính lái / kính hậu */}
      <group position={[1.0, 1.17, 0]} rotation={[0, 0, Math.atan2(0.55, 0.34)]}>
        {[-0.18, -0.06, 0.06, 0.18].map((y) => (
          <mesh key={y} position={[-0.01, y, 0]} rotation={[0, Math.PI / 2, 0]} material={say}>
            <planeGeometry args={[1.4, 0.012]} />
          </mesh>
        ))}
      </group>
      <group position={[-1.34, 1.19, 0]} rotation={[0, 0, -Math.atan2(0.52, 0.39)]}>
        {[-0.16, -0.05, 0.06, 0.17].map((y) => (
          <mesh key={y} position={[0.01, y, 0]} rotation={[0, -Math.PI / 2, 0]} material={sayHau}>
            <planeGeometry args={[1.3, 0.012]} />
          </mesh>
        ))}
      </group>
    </group>
  );
}
