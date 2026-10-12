/**
 * Sân khấu 3D: ánh sáng, mặt đường (vạch kẻ trôi khi "Đang chạy"), bóng, và CAMERA BAY TỚI BỘ PHẬN theo lệnh
 * (`ui_action.camera` trong schema — PLAN.md Phase 7). Không có lệnh mới một lúc -> xe tự xoay chậm; người xem
 * vẫn kéo chuột xoay / cuộn phóng to được (OrbitControls).
 */

import { ContactShadows, Environment, Lightformer, OrbitControls } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { CanhQuan } from "./CanhQuan";
import { type TrangThaiXe, XeLowPoly } from "./XeLowPoly";

/** Góc máy theo tên trong schema (`cameraPresets`) + 2 góc thêm cho kính bên phụ. [vị trí, điểm nhìn]. */
export const GOC: Record<string, [[number, number, number], [number, number, number]]> = {
  // nhìn từ phía làn đối diện (bên lái): hàng cây trái cách xa (z ≈ −8,85), dãy nhà bên kia đường làm nền
  overview: [[6.4, 2.6, -4.6], [0, 0.75, 0]],
  // hai góc TRONG KHOANG (điều hoà, quạt, sấy, âm lượng, nhạc, dẫn đường): ngồi hàng sau nhìn lên táp-lô
  dashboard: [[-0.55, 1.32, 0.0], [1.0, 1.02, 0]],
  center_console: [[-0.25, 1.28, 0.32], [1.0, 1.08, 0.04]],
  rear: [[-6.4, 2.4, 2.6], [-1.8, 0.85, 0]],
  driver_door: [[1.2, 1.7, -5.4], [-0.1, 0.9, -0.9]],
  passenger_door: [[1.4, 1.6, 4.4], [-0.1, 0.9, 0.9]], // trước hàng cây vỉa hè phải (z ≈ 5,35)
  wheel_fl: [[3.2, 1.0, -3.1], [1.6, 0.35, -0.9]],
  charge_port: [[-3.3, 1.5, -3.7], [-1.5, 0.85, -0.9]],
  fuel_flap: [[-3.3, 1.5, 3.7], [-1.5, 0.85, 0.9]],
  roof: [[1.8, 5.8, 3.2], [-0.2, 1.3, 0]],
  front: [[6.4, 1.6, 2.2], [2.2, 0.7, 0]],
};

/** Lượn camera tới góc `goc` mỗi khi `lan` đổi; xong thì trả quyền cho chuột. Rảnh lâu -> tự xoay. */
function DieuCamera({ goc, lan }: { goc: string; lan: number }) {
  const { camera, controls } = useThree() as unknown as {
    camera: THREE.PerspectiveCamera;
    controls: { target: THREE.Vector3; autoRotate: boolean; update: () => void } | null;
  };
  const dangLuon = useRef(false);
  const lanCuoi = useRef(performance.now());
  useEffect(() => {
    dangLuon.current = true;
    lanCuoi.current = performance.now();
  }, [goc, lan]);
  useFrame((_, dt) => {
    if (!controls) return;
    const [vt, nhin] = GOC[goc] ?? GOC.overview;
    const ranh = performance.now() - lanCuoi.current > 9000;
    controls.autoRotate = ranh && !dangLuon.current;
    if (dangLuon.current) {
      const k = 1 - Math.exp(-3 * dt);
      camera.position.lerp(new THREE.Vector3(...vt), k);
      controls.target.lerp(new THREE.Vector3(...nhin), k);
      controls.update();
      if (camera.position.distanceTo(new THREE.Vector3(...vt)) < 0.03) dangLuon.current = false;
    }
  });
  return null;
}

const GOC_TRONG_XE = new Set(["dashboard", "center_console"]);

export function CanhXe({ s, goc, lan }: { s: TrangThaiXe; goc: string; lan: number }) {
  const [nhapNhay, setNhapNhay] = useState(false);
  const trongXe = GOC_TRONG_XE.has(goc);
  useEffect(() => {
    if (!s.hazards_on) return;
    const id = window.setInterval(() => setNhapNhay((v) => !v), 450);
    return () => window.clearInterval(id);
  }, [s.hazards_on]);
  return (
    <Canvas shadows camera={{ position: GOC.overview[0], fov: 40 }} dpr={[1, 2]}>
      {/* sương xa cùng màu chân trời của bầu trời -> nhà / núi xa mờ dần tự nhiên */}
      <fog attach="fog" args={["#d4e4f0", 40, 170]} />
      <hemisphereLight args={["#ffffff", "#cbd5e1", 0.6]} />
      <directionalLight position={[5, 8, 4]} intensity={1.4} castShadow shadow-mapSize={[2048, 2048]} />
      <directionalLight position={[-6, 4, -5]} intensity={0.4} />
      {/* "đèn studio" cho sơn bóng / kính phản chiếu — dựng bằng code, KHÔNG tải ảnh HDR (chạy offline) */}
      <Environment resolution={256}>
        <Lightformer form="rect" intensity={2.2} position={[0, 6, 0]} rotation-x={Math.PI / 2} scale={[12, 3, 1]} />
        <Lightformer form="rect" intensity={1.4} position={[0, 3, 7]} scale={[14, 2.5, 1]} />
        <Lightformer form="rect" intensity={1.4} position={[0, 3, -7]} rotation-y={Math.PI} scale={[14, 2.5, 1]} />
        <Lightformer form="rect" intensity={1} position={[8, 2, 0]} rotation-y={-Math.PI / 2} scale={[8, 2, 1]} />
        <Lightformer form="rect" intensity={0.8} position={[-8, 2, 0]} rotation-y={Math.PI / 2} scale={[8, 2, 1]} />
      </Environment>
      <CanhQuan chay={Boolean(s.dang_chay)} />
      <XeLowPoly s={s} nhapNhay={nhapNhay} trongXe={trongXe} />
      <ContactShadows position={[0, 0.006, 0]} opacity={0.45} scale={10} blur={2.4} far={2} />
      {/* trong khoang cho lại gần (camera chỉ cách táp-lô ~1,5 m); ngoài xe giữ tối thiểu 3 m */}
      <OrbitControls makeDefault enablePan={false} minDistance={trongXe ? 0.3 : 3} maxDistance={12}
        maxPolarAngle={Math.PI / 2.05} autoRotateSpeed={0.6} target={GOC.overview[1]} />
      <DieuCamera goc={goc} lan={lan} />
    </Canvas>
  );
}
