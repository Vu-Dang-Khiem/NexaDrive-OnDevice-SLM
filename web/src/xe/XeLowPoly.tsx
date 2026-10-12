/**
 * Xe DỰNG BẰNG CODE (PLAN.md Phase 7, cách 1): mỗi bộ phận là một khối riêng nên mở / đóng được hết
 * — 4 cửa (bản lề trước), 4 kính (hạ theo đúng % `windows`), cốp, cửa sổ trời, nắp sạc, nắp xăng, đèn pha,
 * đèn khẩn cấp, bánh xe quay khi "Đang chạy". Không cần mô hình tải về hay Blender.
 * Bản 2 (người dùng: "xấu, không thật"): dáng cong mượt, sơn kim loại có lớp bóng + phản chiếu (đèn studio ở
 * CanhXe), kính phản chiếu, bánh nằm trong vòm tối, gương, lưới tản nhiệt, khe cửa, đèn LED mảnh.
 *
 * Toạ độ: đầu xe hướng +X, trục đứng +Y, bên PHẢI xe là +Z (ghế lái bên trái = −Z, xe Việt Nam).
 * Mọi chuyển động đi MƯỢT tới giá trị đích (giảm dần theo thời gian), không nhảy cóc.
 */

import { useFrame } from "@react-three/fiber";
import { type ReactNode, useMemo, useRef } from "react";
import * as THREE from "three";
import { NoiThat, type TrangThaiKhoang } from "./NoiThat";

export type TrangThaiXe = TrangThaiKhoang & {
  dang_chay?: boolean;
  doors_open?: string[];
  trunk_open?: boolean;
  charge_port_open?: boolean;
  fuel_flap_open?: boolean;
  windows?: Record<string, number>;
  sunroof_state?: string;
  sunroof_percent?: number;
  headlights?: string;
  hazards_on?: boolean;
};

const HONG = 0.94; // nửa bề rộng thân xe (mặt hông)
const MAU = { kinh: "#0b1220", den: "#10141a", vien: "#c9d1da", hau: "#e11d2a", xinhan: "#f59e0b" };

/** Tiến `hien` tới `dich` mượt, độc lập tốc độ khung hình. */
export function damp(hien: number, dich: number, toc_do: number, dt: number): number {
  return hien + (dich - hien) * (1 - Math.exp(-toc_do * dt));
}

/** Mép trên khung kính hông theo x (đi theo đường nóc / kính lái / kính hậu của mặt cắt thân xe). */
function tren(x: number): number {
  if (x >= 0.7) return 1.33 - ((x - 0.7) / 0.55) * 0.34;
  if (x >= 0.35) return 1.45 - ((x - 0.35) / 0.35) * 0.12;
  if (x >= -0.6) return 1.45;
  if (x >= -1.08) return 1.38 + ((x + 1.08) / 0.48) * 0.07;
  return 1.38 - ((-1.08 - x) / 0.52) * 0.39;
}
const GAY = [-1.08, -0.6, 0.35, 0.7]; // các điểm gãy của đường mép trên

/** Mảng kính / khoang hông giữa x0..x1, mép trên lùi `le` cho có khung. Gốc toạ độ dời về (gocX, gocY):
 *  khoang dùng (0, 0) — toạ độ xe; kính cửa dùng (x1, 1,0) — bản lề cửa + đường gờ, để co kính từ gờ lên. */
function hinhKinhHong(x0: number, x1: number, gocX: number, gocY: number, le = 0.05): THREE.ShapeGeometry {
  const s = new THREE.Shape();
  s.moveTo(x1 - gocX, 1.0 - gocY);
  s.lineTo(x0 - gocX, 1.0 - gocY);
  for (const x of [x0, ...GAY.filter((g) => g > x0 && g < x1), x1]) s.lineTo(x - gocX, tren(x) - le - gocY);
  s.closePath();
  return new THREE.ShapeGeometry(s, 4);
}

const X_BANH_TRUOC = 1.6;
const X_BANH_SAU = -1.45;
const X_CUA_TRUOC = 1.15; // mép trước cửa trước (bản lề)
const X_COT_B = -0.05;
const X_CUA_SAU = -0.98; // mép sau cửa sau — ngay sau vòm bánh sau

/** Đùn một mặt cắt dọc theo bề rộng [z0, z1], cạnh vát tròn nằm TRONG biên dạng (bevelOffset = −bevelSize —
 *  không thì khối phình ra mỗi phía và lệch khỏi kính / cửa: lỗi có từ bản 1). */
function dun(s: THREE.Shape, z0: number, z1: number, vat = 0.06): THREE.ExtrudeGeometry {
  const g = new THREE.ExtrudeGeometry(s, {
    depth: Math.max(0.001, z1 - z0 - 2 * vat), bevelEnabled: true, bevelSize: vat, bevelOffset: -vat,
    bevelThickness: vat, bevelSegments: 5, curveSegments: 24,
  });
  g.translate(0, 0, z0 + vat);
  g.computeVertexNormals();
  return g;
}

/** Thân xe RỖNG có ô cửa sổ thật (bản 3 — người dùng muốn camera vào trong xe): khối đầu (capo, mũi), khối đuôi
 *  (cốp, vè sau, mảng trụ C), mui, gờ dưới cửa. Khoang lái để trống: từ ngoài nhìn qua kính thấy ghế, từ trong
 *  nhìn ra thấy đường, mở cửa / hạ kính là mở thật. */
function useThanXe() {
  return useMemo(() => {
    const dau = new THREE.Shape();
    dau.moveTo(X_CUA_TRUOC, 0.3);
    dau.lineTo(X_CUA_TRUOC, 0.99);
    dau.lineTo(1.25, 0.99);
    dau.quadraticCurveTo(1.8, 0.93, 2.25, 0.83);
    dau.quadraticCurveTo(2.44, 0.78, 2.43, 0.6);
    dau.lineTo(2.37, 0.36);
    dau.quadraticCurveTo(2.32, 0.3, 2.1, 0.3);
    dau.closePath();

    const duoi = new THREE.Shape();
    duoi.moveTo(X_CUA_SAU, 0.3);
    duoi.lineTo(-2.1, 0.3);
    duoi.quadraticCurveTo(-2.3, 0.3, -2.32, 0.34);
    duoi.lineTo(-2.38, 0.62);
    duoi.quadraticCurveTo(-2.4, 0.92, -2.2, 0.95);
    duoi.lineTo(X_CUA_SAU, 0.99);
    duoi.closePath();

    // mảng trụ C (giữa kính hậu và kính cửa sau), mỗi bên một tấm mỏng
    const cotC = new THREE.Shape();
    cotC.moveTo(-1.6, 0.99);
    cotC.lineTo(X_CUA_SAU, 0.99);
    cotC.lineTo(X_CUA_SAU, tren(X_CUA_SAU) + 0.02);
    cotC.lineTo(-1.08, 1.4);
    cotC.closePath();

    // mui: tấm cong theo đường nóc
    const mai = new THREE.Shape();
    mai.moveTo(-1.08, 1.38);
    mai.quadraticCurveTo(-0.95, 1.455, -0.6, 1.455);
    mai.lineTo(0.35, 1.45);
    mai.quadraticCurveTo(0.56, 1.44, 0.7, 1.33);
    mai.lineTo(0.66, 1.3);
    mai.quadraticCurveTo(0.52, 1.4, 0.35, 1.405);
    mai.lineTo(-0.6, 1.41);
    mai.quadraticCurveTo(-0.92, 1.41, -1.04, 1.34);
    mai.closePath();

    // gờ dưới cửa (bậu) chạy dọc khoang
    const bau = new THREE.Shape();
    bau.moveTo(X_CUA_SAU, 0.3);
    bau.lineTo(X_CUA_TRUOC, 0.3);
    bau.lineTo(X_CUA_TRUOC, 0.39);
    bau.lineTo(X_CUA_SAU, 0.39);
    bau.closePath();

    return {
      dau: dun(dau, -HONG, HONG, 0.09),
      duoi: dun(duoi, -HONG, HONG, 0.09),
      cotC: [dun(cotC, -HONG, -HONG + 0.07, 0.02), dun(cotC, HONG - 0.07, HONG, 0.02)],
      mai: dun(mai, -0.9, 0.9, 0.05),
      bau: [dun(bau, -HONG, -HONG + 0.1, 0.03), dun(bau, HONG - 0.1, HONG, 0.03)],
    };
  }, []);
}

/** Thanh nối hai điểm (trụ A dọc mép kính lái). */
function Thanh({ a, b, z, day = 0.07, rong = 0.1, children }:
  { a: [number, number]; b: [number, number]; z: number; day?: number; rong?: number; children: ReactNode }) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  return (
    <mesh position={[(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, z]} rotation={[0, 0, Math.atan2(dy, dx)]}>
      <boxGeometry args={[Math.hypot(dx, dy) + 0.04, day, rong]} />
      {children}
    </mesh>
  );
}

/** Sơn kim loại có lớp bóng — phản chiếu đèn studio dựng trong CanhXe. */
function Son({ mau, haiMat = false }: { mau: string; haiMat?: boolean }) {
  return (
    <meshPhysicalMaterial color={mau} metalness={0.55} roughness={0.32} clearcoat={1} clearcoatRoughness={0.08}
      envMapIntensity={1.2} side={haiMat ? THREE.DoubleSide : THREE.FrontSide} />
  );
}
function Kinh({ trongSuot = 0.94 }: { trongSuot?: number }) {
  return <meshPhysicalMaterial color={MAU.kinh} metalness={0.9} roughness={0.05} envMapIntensity={1.6} transparent opacity={trongSuot} side={THREE.DoubleSide} />;
}

type CuaProps = { x0: number; x1: number; phai: boolean; mo: boolean; phanTramKinh: number; mauSon: string };

/** Một cánh cửa: tấm cửa + kính (hạ theo %) + tay nắm, bản lề ở mép trước, mở ra ngoài ~62°. */
function Cua({ x0, x1, phai, mo, phanTramKinh, mauSon }: CuaProps) {
  const nhom = useRef<THREE.Group>(null);
  const kinh = useRef<THREE.Mesh>(null);
  const z = phai ? HONG + 0.004 : -HONG - 0.004;
  const dai = x1 - x0;
  const kinhHinh = useMemo(() => hinhKinhHong(x0, x1, x1, 1.0), [x0, x1]);
  useFrame((_, dt) => {
    if (nhom.current) nhom.current.rotation.y = damp(nhom.current.rotation.y, mo ? (phai ? 1.08 : -1.08) : 0, 5, dt);
    if (kinh.current) {
      // kính co từ mép trên xuống gờ: phần còn lại = 1 − %mở (giữ tối thiểu 2% cho khỏi biến mất)
      kinh.current.scale.y = damp(kinh.current.scale.y, Math.max(0.02, 1 - phanTramKinh / 100), 4, dt);
    }
  });
  const ra = phai ? 1 : -1;
  return (
    <group ref={nhom} position={[x1, 0, z]}>
      <mesh position={[-dai / 2, 0.68, 0]} castShadow>
        <boxGeometry args={[dai - 0.015, 0.6, 0.02]} />
        <Son mau={mauSon} />
      </mesh>
      {/* khe cửa: viền tối mảnh ở hai mép đứng */}
      {[0, -dai].map((x) => (
        <mesh key={x} position={[x, 0.68, ra * 0.011]}>
          <boxGeometry args={[0.012, 0.6, 0.004]} />
          <meshBasicMaterial color="#20262e" />
        </mesh>
      ))}
      <mesh position={[-dai * 0.8, 0.86, ra * 0.016]}>
        <boxGeometry args={[0.2, 0.03, 0.012]} />
        <meshStandardMaterial color={MAU.vien} metalness={0.9} roughness={0.2} />
      </mesh>
      {/* nẹp chrome dọc mép dưới kính */}
      <mesh position={[-dai / 2, 0.995, ra * 0.012]}>
        <boxGeometry args={[dai - 0.03, 0.015, 0.01]} />
        <meshStandardMaterial color={MAU.vien} metalness={0.95} roughness={0.15} />
      </mesh>
      <mesh ref={kinh} geometry={kinhHinh} position={[0, 1.0, ra * 0.014]}>
        <Kinh trongSuot={0.5} />
      </mesh>
    </group>
  );
}

function BanhXe({ x, z, chay }: { x: number; z: number; chay: boolean }) {
  const quay = useRef<THREE.Group>(null);
  const toc = useRef(0);
  const ra = z > 0 ? 1 : -1;
  useFrame((_, dt) => {
    toc.current = damp(toc.current, chay ? 9 : 0, 1.5, dt);
    if (quay.current) quay.current.rotation.z -= toc.current * dt;
  });
  return (
    <group position={[x, 0.36, z]}>
      <group ref={quay}>
        <mesh rotation={[Math.PI / 2, 0, 0]} castShadow>
          <cylinderGeometry args={[0.36, 0.36, 0.25, 40]} />
          <meshStandardMaterial color="#141619" roughness={0.85} />
        </mesh>
        {/* hông lốp bo tròn — torus mặc định nằm trong mặt XY, trục Z = trục bánh, KHÔNG xoay */}
        <mesh position={[0, 0, ra * 0.1]}>
          <torusGeometry args={[0.31, 0.05, 12, 40]} />
          <meshStandardMaterial color="#1b1e22" roughness={0.8} />
        </mesh>
        <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, 0, ra * 0.118]}>
          <cylinderGeometry args={[0.25, 0.25, 0.02, 40]} />
          <meshStandardMaterial color="#3b4048" metalness={0.8} roughness={0.35} />
        </mesh>
        {Array.from({ length: 10 }, (_, i) => (
          <mesh key={i} rotation={[0, 0, (i * Math.PI) / 5]} position={[0, 0, ra * 0.13]}>
            <boxGeometry args={[0.035, 0.46, 0.014]} />
            <meshStandardMaterial color="#d6dbe1" metalness={0.95} roughness={0.18} />
          </mesh>
        ))}
        <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, 0, ra * 0.138]}>
          <cylinderGeometry args={[0.055, 0.055, 0.02, 20]} />
          <meshStandardMaterial color="#d6dbe1" metalness={0.95} roughness={0.18} />
        </mesh>
      </group>
    </group>
  );
}

/** Hốc bánh: nửa đĩa tối trên hông xe sau bánh (bánh trông nằm TRONG vòm) + viền vòm. */
function VomBanh({ x, phai }: { x: number; phai: boolean }) {
  const z = phai ? HONG + 0.001 : -HONG - 0.001;
  return (
    <group position={[x, 0.36, z]} rotation={[0, phai ? 0 : Math.PI, 0]}>
      <mesh>
        <circleGeometry args={[0.46, 40, 0, Math.PI]} />
        <meshBasicMaterial color="#07090b" />
      </mesh>
      <mesh position={[0, 0, 0.002]}>
        <ringGeometry args={[0.46, 0.5, 40, 1, 0, Math.PI]} />
        <meshStandardMaterial color="#1a1d22" roughness={0.6} />
      </mesh>
    </group>
  );
}

/** Nắp nhỏ (sạc / xăng) trên hông sau, bản lề dọc, mở ra ngoài. */
function NapNho({ phai, mo, mau, mauSon }: { phai: boolean; mo: boolean; mau: string; mauSon: string }) {
  const nhom = useRef<THREE.Group>(null);
  useFrame((_, dt) => {
    if (nhom.current) nhom.current.rotation.y = damp(nhom.current.rotation.y, mo ? (phai ? 1.3 : -1.3) : 0, 6, dt);
  });
  return (
    <group position={[-1.5, 0.84, phai ? HONG + 0.002 : -HONG - 0.002]}>
      <mesh position={[-0.1, 0, 0]}>
        <boxGeometry args={[0.18, 0.15, 0.004]} />
        <meshBasicMaterial color={mo ? mau : "#2a3038"} />
      </mesh>
      <group ref={nhom}>
        <mesh position={[-0.1, 0, phai ? 0.008 : -0.008]}>
          <boxGeometry args={[0.2, 0.17, 0.014]} />
          <Son mau={mauSon} />
        </mesh>
      </group>
    </group>
  );
}

export function XeLowPoly({ s, nhapNhay, trongXe = false, mauSon = "#1d3a5c" }:
  { s: TrangThaiXe; nhapNhay: boolean; trongXe?: boolean; mauSon?: string }) {
  const than = useThanXe();
  const cop = useRef<THREE.Group>(null);
  const troi = useRef<THREE.Group>(null);
  const toanXe = useRef<THREE.Group>(null);
  const t = useRef(0);
  const cua = new Set(s.doors_open ?? []);
  const kinh = s.windows ?? {};
  const chay = Boolean(s.dang_chay);

  // kính lái: chân (1,25; 0,99) -> đỉnh (0,7; 1,33) — mặt phẳng đứng nghiêng về sau một góc atan(0,55 / 0,34)
  const kinhLai = useMemo(() => {
    const g = new THREE.PlaneGeometry(1.56, 0.64, 1, 1);
    g.rotateY(Math.PI / 2);
    g.rotateZ(Math.atan2(0.55, 0.34));
    return g;
  }, []);
  // kính hậu: chân (−1,6; 0,99) -> đỉnh (−1,08; 1,38) — nghiêng về trước
  const kinhHau = useMemo(() => {
    const g = new THREE.PlaneGeometry(1.5, 0.62, 1, 1);
    g.rotateY(-Math.PI / 2);
    g.rotateZ(-Math.atan2(0.52, 0.39));
    return g;
  }, []);

  useFrame((_, dt) => {
    t.current += dt;
    if (cop.current) cop.current.rotation.z = damp(cop.current.rotation.z, s.trunk_open ? -1.15 : 0, 4, dt);
    if (troi.current) {
      const pt = (s.sunroof_percent ?? 0) / 100;
      const nghieng = s.sunroof_state === "tilt";
      troi.current.position.x = damp(troi.current.position.x, nghieng ? 0 : -0.7 * pt, 4, dt);
      troi.current.rotation.z = damp(troi.current.rotation.z, nghieng ? -0.18 * Math.max(pt, 0.4) : 0, 4, dt);
    }
    if (toanXe.current) toanXe.current.position.y = chay ? Math.sin(t.current * 14) * 0.005 : 0; // rung nhẹ khi chạy
  });

  const pha = s.headlights ?? "auto";
  const sangDenPha = pha === "low_beam" || pha === "high_beam" ? 3 : pha === "auto" ? 1.2 : pha === "parking" ? 0.6 : 0.05;
  const xiNhan = s.hazards_on && nhapNhay ? 4 : 0.05;

  return (
    <group ref={toanXe}>
      {/* thân RỖNG: khối đầu, khối đuôi, mui, mảng trụ C, bậu cửa — khoang lái để trống */}
      {[than.dau, than.duoi, than.mai, ...than.cotC, ...than.bau].map((g, i) => (
        <mesh key={i} geometry={g} castShadow receiveShadow>
          <Son mau={mauSon} />
        </mesh>
      ))}
      {/* trần xe (mặt dưới mui) màu sáng — nhìn từ trong khoang */}
      <mesh position={[-0.15, 1.395, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <planeGeometry args={[1.5, 1.74]} />
        <meshStandardMaterial color="#c9ced5" roughness={0.95} />
      </mesh>
      {/* trụ A dọc hai mép kính lái, trụ B giữa hai cửa */}
      {[-1, 1].map((ra) => (
        <group key={ra}>
          <Thanh a={[1.25, 0.99]} b={[0.7, 1.33]} z={ra * 0.88}><Son mau={mauSon} /></Thanh>
          <mesh position={[X_COT_B, 1.2, ra * (HONG - 0.035)]}>
            <boxGeometry args={[0.09, 0.44, 0.07]} />
            <meshStandardMaterial color="#14171b" roughness={0.5} />
          </mesh>
        </group>
      ))}
      <NoiThat s={s} trongXe={trongXe} />
      <mesh geometry={kinhLai} position={[1.0, 1.185, 0]}>
        <Kinh trongSuot={0.32} />
      </mesh>
      <mesh geometry={kinhHau} position={[-1.365, 1.205, 0]}>
        <Kinh trongSuot={0.45} />
      </mesh>
      {/* 4 cửa: trước (cột B → 1,15), sau (sau vòm bánh −0,98 → cột B); trái = ghế lái */}
      <Cua x0={X_COT_B} x1={X_CUA_TRUOC} phai={false} mo={cua.has("driver")} phanTramKinh={kinh.front_left ?? 0} mauSon={mauSon} />
      <Cua x0={X_COT_B} x1={X_CUA_TRUOC} phai={true} mo={cua.has("passenger")} phanTramKinh={kinh.front_right ?? 0} mauSon={mauSon} />
      <Cua x0={X_CUA_SAU} x1={X_COT_B} phai={false} mo={cua.has("rear_left")} phanTramKinh={kinh.rear_left ?? 0} mauSon={mauSon} />
      <Cua x0={X_CUA_SAU} x1={X_COT_B} phai={true} mo={cua.has("rear_right")} phanTramKinh={kinh.rear_right ?? 0} mauSon={mauSon} />
      {/* gương chiếu hậu */}
      {[-1, 1].map((ra) => (
        <group key={ra} position={[1.08, 1.02, ra * (HONG + 0.08)]}>
          <mesh position={[0, 0, -ra * 0.05]}>
            <boxGeometry args={[0.08, 0.04, 0.1]} />
            <meshStandardMaterial color="#111418" />
          </mesh>
          <mesh>
            <boxGeometry args={[0.14, 0.11, 0.08]} />
            <Son mau={mauSon} />
          </mesh>
        </group>
      ))}
      {/* cốp: bản lề ở mép trước nắp cốp (x −1,6) */}
      <group ref={cop} position={[-1.6, 1.0, 0]}>
        <mesh position={[-0.3, 0, 0]} castShadow>
          <boxGeometry args={[0.62, 0.035, 2 * HONG - 0.12]} />
          <Son mau={mauSon} />
        </mesh>
      </group>
      {/* cửa sổ trời: trượt về sau theo %, hoặc hé (nghiêng) */}
      <group position={[0.0, 1.462, 0]}>
        <group ref={troi}>
          <mesh>
            <boxGeometry args={[0.8, 0.012, 0.92]} />
            <Kinh trongSuot={1} />
          </mesh>
        </group>
      </group>
      <NapNho phai={false} mo={Boolean(s.charge_port_open)} mau="#22c55e" mauSon={mauSon} />
      <NapNho phai={true} mo={Boolean(s.fuel_flap_open)} mau="#3b82f6" mauSon={mauSon} />
      {/* mặt trước: lưới tản nhiệt + hốc gió dưới; đèn LED mảnh */}
      <mesh position={[2.42, 0.5, 0]}>
        <boxGeometry args={[0.04, 0.14, 0.9]} />
        <meshStandardMaterial color={MAU.den} metalness={0.4} roughness={0.5} />
      </mesh>
      <mesh position={[2.38, 0.36, 0]}>
        <boxGeometry args={[0.04, 0.06, 1.3]} />
        <meshStandardMaterial color={MAU.den} />
      </mesh>
      {[-0.6, 0.6].map((z) => (
        <mesh key={z} position={[2.385, 0.72, z]} rotation={[0, 0, 0.35]}>
          <boxGeometry args={[0.03, 0.05, 0.42]} />
          <meshStandardMaterial color="#f8fafc" emissive="#fff7d6" emissiveIntensity={sangDenPha} />
        </mesh>
      ))}
      {/* đuôi: dải đèn hậu chạy ngang + hai cụm góc */}
      <mesh position={[-2.395, 0.84, 0]}>
        <boxGeometry args={[0.02, 0.035, 1.5]} />
        <meshStandardMaterial color={MAU.hau} emissive={MAU.hau} emissiveIntensity={chay ? 2 : 0.7} />
      </mesh>
      {[-0.68, 0.68].map((z) => (
        <mesh key={z} position={[-2.39, 0.8, z]}>
          <boxGeometry args={[0.03, 0.09, 0.3]} />
          <meshStandardMaterial color={MAU.hau} emissive={MAU.hau} emissiveIntensity={chay ? 1.6 : 0.5} />
        </mesh>
      ))}
      {/* xi-nhan 4 góc (đèn khẩn cấp nháy) */}
      {[-0.86, 0.86].map((z) => (
        <group key={z}>
          <mesh position={[2.35, 0.66, z]}>
            <boxGeometry args={[0.06, 0.04, 0.1]} />
            <meshStandardMaterial color={MAU.xinhan} emissive={MAU.xinhan} emissiveIntensity={xiNhan} />
          </mesh>
          <mesh position={[-2.36, 0.72, z]}>
            <boxGeometry args={[0.05, 0.05, 0.1]} />
            <meshStandardMaterial color={MAU.xinhan} emissive={MAU.xinhan} emissiveIntensity={xiNhan} />
          </mesh>
        </group>
      ))}
      {/* vệt sáng: chiếu xa dài + thẳng; chiếu gần ngắn, chúc xuống mặt đường (trước đây chiếu gần không có vệt) */}
      {pha === "high_beam" && (
        <mesh position={[3.7, 0.55, 0]} rotation={[0, 0, Math.PI / 2]}>
          <coneGeometry args={[0.9, 2.4, 24, 1, true]} />
          <meshBasicMaterial color="#fff3b0" transparent opacity={0.14} side={THREE.DoubleSide} depthWrite={false} />
        </mesh>
      )}
      {pha === "low_beam" && (
        <mesh position={[3.15, 0.5, 0]} rotation={[0, 0, Math.PI / 2 - 0.22]}>
          <coneGeometry args={[0.6, 1.5, 24, 1, true]} />
          <meshBasicMaterial color="#fff3b0" transparent opacity={0.12} side={THREE.DoubleSide} depthWrite={false} />
        </mesh>
      )}
      {[X_BANH_TRUOC, X_BANH_SAU].map((x) => (
        <group key={x}>
          <VomBanh x={x} phai={true} />
          <VomBanh x={x} phai={false} />
          <BanhXe x={x} z={0.82} chay={chay} />
          <BanhXe x={x} z={-0.82} chay={chay} />
        </group>
      ))}
    </group>
  );
}
