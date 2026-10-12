/**
 * Khung cảnh quanh xe (người dùng 2026-10-06: "tạo khung cảnh xung quanh cho nó thật đẹp"): bầu trời có mặt trời,
 * núi xa trong sương, đường nhựa hai làn + vạch kẻ, vỉa hè, hàng cây, cột đèn, dãy nhà phố nhiều màu, cỏ.
 * Khi "Đang chạy": cây / cột đèn / nhà / vạch đường TRÔI về sau rồi quay vòng — xe đứng yên, cảnh chạy.
 * Toàn bộ dựng bằng code (không tải ảnh / mô hình) -> chạy offline.
 *
 * Toạ độ như XeLowPoly: xe ở gốc, đầu xe +X, bên phải xe +Z. Xe chạy làn phải: tim đường ở z = −1,75.
 */

import { Sky } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { type MutableRefObject, useMemo, useRef } from "react";
import * as THREE from "three";

type Ref = MutableRefObject<number>;
/** Bỏ màu riêng của từng vật — cho mảng đi kèm (thân cây, kính nhà) không ăn màu tán / tường. */
const khongMau = <T extends { mau?: THREE.Color }>(ds: T[]): T[] => ds.map((v) => ({ ...v, mau: undefined }));

const DAI = 160; // chiều dài một vòng cảnh (m) — vật đi hết thì quay lại đầu
const TIM = -1.75; // tim đường (vạch giữa)
const LE_PHAI = 1.75; // mép đường bên phải
const LE_TRAI = -5.25;

/** Ngẫu nhiên có hạt giống — cảnh giống nhau mỗi lần mở trang. */
function ngauNhien(hat: number) {
  let s = hat;
  return () => {
    s = (s * 16807) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

type VatTroi = { x: number; z: number; co: number; xoay: number; mau?: THREE.Color };

/** Một loạt vật giống nhau (cây, đèn, nhà) dùng chung một InstancedMesh, trôi theo `quangDuong`. */
function useTroi(vat: VatTroi[], quangDuong: Ref, dung: (o: THREE.Object3D, v: VatTroi) => void) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const tam = useMemo(() => new THREE.Object3D(), []);
  const daMau = useRef(false);
  useFrame(() => {
    const m = ref.current;
    if (!m) return;
    vat.forEach((v, i) => {
      const x = ((((v.x - quangDuong.current) % DAI) + DAI) % DAI) - DAI / 2;
      tam.position.set(x, 0, v.z);
      tam.rotation.set(0, v.xoay, 0);
      tam.scale.setScalar(1);
      dung(tam, v);
      tam.updateMatrix();
      m.setMatrixAt(i, tam.matrix);
      if (!daMau.current && v.mau) m.setColorAt(i, v.mau);
    });
    m.instanceMatrix.needsUpdate = true;
    if (!daMau.current && m.instanceColor) m.instanceColor.needsUpdate = true;
    daMau.current = true;
  });
  return ref;
}

function HangCay({ quangDuong }: { quangDuong: Ref }) {
  const cay = useMemo(() => {
    const r = ngauNhien(7);
    const ds: VatTroi[] = [];
    for (let x = 0; x < DAI; x += 9) {
      for (const z of [LE_PHAI + 3.6, LE_TRAI - 3.6]) { // mép ngoài vỉa hè — camera quanh xe ít bị cây che
        const xanh = new THREE.Color().setHSL(0.27 + r() * 0.08, 0.45 + r() * 0.2, 0.3 + r() * 0.12);
        ds.push({ x: x + r() * 3, z: z + (r() - 0.5) * 0.6, co: 0.85 + r() * 0.5, xoay: r() * 6, mau: xanh });
      }
    }
    return ds;
  }, []);
  const tan = useTroi(cay, quangDuong, (o, v) => {
    o.position.y = 2.6 * v.co;
    o.scale.setScalar(v.co);
  });
  const cayTrongThan = useMemo(() => khongMau(cay), [cay]);
  const than = useTroi(cayTrongThan, quangDuong, (o, v) => {
    o.position.y = 0.9 * v.co;
    o.scale.set(v.co, v.co, v.co);
  });
  return (
    <group>
      <instancedMesh ref={tan} args={[undefined, undefined, cay.length]} castShadow>
        <icosahedronGeometry args={[1.3, 0]} />
        <meshStandardMaterial flatShading roughness={0.9} />
      </instancedMesh>
      <instancedMesh ref={than} args={[undefined, undefined, cay.length]}>
        <cylinderGeometry args={[0.12, 0.18, 1.8, 6]} />
        <meshStandardMaterial color="#6b4f3a" roughness={0.9} />
      </instancedMesh>
    </group>
  );
}

function CotDen({ quangDuong }: { quangDuong: Ref }) {
  const den = useMemo(() => {
    const ds: VatTroi[] = [];
    for (let x = 4; x < DAI; x += 18) ds.push({ x, z: LE_PHAI + 0.7, co: 1, xoay: 0 });
    for (let x = 13; x < DAI; x += 18) ds.push({ x, z: LE_TRAI - 0.7, co: 1, xoay: Math.PI });
    return ds;
  }, []);
  const cot = useTroi(den, quangDuong, (o) => { o.position.y = 2.6; });
  const canh = useTroi(den, quangDuong, (o, v) => {
    o.position.y = 5.1;
    o.position.z += v.xoay ? 0.6 : -0.6;
  });
  return (
    <group>
      <instancedMesh ref={cot} args={[undefined, undefined, den.length]}>
        <cylinderGeometry args={[0.06, 0.09, 5.2, 8]} />
        <meshStandardMaterial color="#4b5563" metalness={0.6} roughness={0.4} />
      </instancedMesh>
      <instancedMesh ref={canh} args={[undefined, undefined, den.length]}>
        <boxGeometry args={[0.35, 0.12, 1.3]} />
        <meshStandardMaterial color="#374151" metalness={0.5} roughness={0.4} emissive="#fff3c4" emissiveIntensity={0.15} />
      </instancedMesh>
    </group>
  );
}

const MAU_NHA = ["#f4e3c1", "#e8c9a0", "#f2d0c4", "#cfe0e8", "#e9edf0", "#f6e7a8", "#d9c7e6", "#c9dcc0"];

function DayNha({ quangDuong }: { quangDuong: Ref }) {
  const nha = useMemo(() => {
    const r = ngauNhien(31);
    const ds: (VatTroi & { cao: number; rong: number; sau: number })[] = [];
    for (const [z, chieu] of [[LE_PHAI + 9, 1], [LE_TRAI - 9, -1]] as const) {
      let x = 0;
      while (x < DAI) {
        const rong = 4 + r() * 4;
        ds.push({ x: x + rong / 2, z: z + chieu * r() * 1.5, co: 1, xoay: 0, cao: 5 + r() * 14, rong, sau: 6 + r() * 3,
          mau: new THREE.Color(MAU_NHA[Math.floor(r() * MAU_NHA.length)]) });
        x += rong + 0.4 + r() * 1.5;
      }
    }
    return ds;
  }, []);
  const khoi = useTroi(nha, quangDuong, (o, v) => {
    const n = v as (typeof nha)[number];
    o.position.y = n.cao / 2;
    o.scale.set(n.rong, n.cao, n.sau);
  });
  // dải cửa sổ: tấm kính tối phủ mặt nhà hướng ra đường
  const nhaKinh = useMemo(() => khongMau(nha), [nha]);
  const cuaSo = useTroi(nhaKinh, quangDuong, (o, v) => {
    const n = v as (typeof nha)[number];
    o.position.y = n.cao / 2 + 0.4;
    o.position.z += (n.z > 0 ? -1 : 1) * (n.sau / 2 + 0.02);
    o.scale.set(n.rong * 0.8, n.cao - 1.8, 1);
  });
  return (
    <group>
      <instancedMesh ref={khoi} args={[undefined, undefined, nha.length]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial roughness={0.85} />
      </instancedMesh>
      <instancedMesh ref={cuaSo} args={[undefined, undefined, nha.length]}>
        <planeGeometry args={[1, 1]} />
        <meshStandardMaterial color="#5b7a96" metalness={0.4} roughness={0.25} side={THREE.DoubleSide} transparent opacity={0.55} />
      </instancedMesh>
    </group>
  );
}

/** Vạch giữa đường (đứt khúc) — trôi như cây, chu kỳ 6 m. */
function VachGiua({ quangDuong }: { quangDuong: Ref }) {
  const nhom = useRef<THREE.Group>(null);
  useFrame(() => {
    if (nhom.current) nhom.current.position.x = -(((quangDuong.current % 6) + 6) % 6);
  });
  return (
    <group ref={nhom}>
      {Array.from({ length: 40 }, (_, i) => (
        <mesh key={i} rotation={[-Math.PI / 2, 0, 0]} position={[-117 + i * 6, 0.012, TIM]}>
          <planeGeometry args={[3, 0.14]} />
          <meshStandardMaterial color="#f5f5f4" roughness={0.6} />
        </mesh>
      ))}
    </group>
  );
}

function DayNui() {
  const nui = useMemo(() => {
    const r = ngauNhien(99);
    return Array.from({ length: 14 }, (_, i) => ({
      x: -140 + i * 22 + r() * 10, z: (i % 2 ? 1 : -1) * (70 + r() * 25), cao: 18 + r() * 22, rong: 26 + r() * 18,
    }));
  }, []);
  return (
    <group>
      {nui.map((n, i) => (
        <mesh key={i} position={[n.x, n.cao / 2 - 1, n.z]}>
          <coneGeometry args={[n.rong, n.cao, 5]} />
          <meshStandardMaterial color="#8fa6b8" flatShading roughness={1} />
        </mesh>
      ))}
    </group>
  );
}

export function CanhQuan({ chay }: { chay: boolean }) {
  const quangDuong = useRef(0);
  const toc = useRef(0);
  useFrame((_, dt) => {
    toc.current += ((chay ? 11 : 0) - toc.current) * (1 - Math.exp(-0.8 * dt)); // ~40 km/h, tăng / giảm tốc êm
    quangDuong.current += toc.current * Math.min(dt, 0.1);
  });
  return (
    <group>
      <Sky distance={4000} sunPosition={[80, 45, 60]} turbidity={5} rayleigh={1.2} mieCoefficient={0.004} mieDirectionalG={0.85} />
      {/* cỏ */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]} receiveShadow>
        <planeGeometry args={[600, 600]} />
        <meshStandardMaterial color="#8fb36a" roughness={1} />
      </mesh>
      {/* mặt đường nhựa + vạch mép liền */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.002, TIM]} receiveShadow>
        <planeGeometry args={[DAI * 2, LE_PHAI - LE_TRAI]} />
        <meshStandardMaterial color="#4a4f57" roughness={0.92} />
      </mesh>
      {[LE_PHAI - 0.25, LE_TRAI + 0.25].map((z) => (
        <mesh key={z} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.01, z]}>
          <planeGeometry args={[DAI * 2, 0.12]} />
          <meshStandardMaterial color="#f5f5f4" roughness={0.6} />
        </mesh>
      ))}
      <VachGiua quangDuong={quangDuong} />
      {/* vỉa hè hai bên (cao hơn mặt đường 15 cm) */}
      {[LE_PHAI + 2, LE_TRAI - 2].map((z) => (
        <mesh key={z} position={[0, 0.075, z]} receiveShadow>
          <boxGeometry args={[DAI * 2, 0.15, 4]} />
          <meshStandardMaterial color="#cfcac0" roughness={0.95} />
        </mesh>
      ))}
      <HangCay quangDuong={quangDuong} />
      <CotDen quangDuong={quangDuong} />
      <DayNha quangDuong={quangDuong} />
      <DayNui />
    </group>
  );
}
