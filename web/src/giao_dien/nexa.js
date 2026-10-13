/* =====================================================================
   Giao diện demo Nexa Drive (người dùng chọn 2026-10-07, PLAN.md Phase 7).

   Khung, cảnh 3D và hoạt cảnh lấy từ bản demo người dùng thiết kế. Ở bản demo gốc, câu trả lời đến
   từ vài luật từ khoá giả; ở đây ĐÃ BỎ — mọi câu trả lời, lời gọi hàm và trạng thái xe đến từ
   backend (serve/app.py) qua WebSocket, đúng hợp đồng cũ (serve/contracts.py, web/src/ws.ts):
     • assistant_turn -> câu trả lời + các lời gọi ĐÃ chạy (đã qua luật an toàn) -> hoạt cảnh 3D;
     • clarify        -> trợ lý hỏi lại (nút Có / Không, Cái đầu / Cái thứ hai);
     • state          -> trạng thái xe của backend: nguồn chân lý cho mọi thứ vẽ trên xe;
     • công tắc Đỗ xe / Đang chạy, Khoá trẻ em gửi lên backend (luật an toàn đọc hai cờ này).
   Giọng nói (PLAN.md Phase 6): micro -> dịch vụ giọng nói (serve/giong_noi.py, cổng 8001: PhoWhisper-small)
   -> chữ -> gửi qua WebSocket như câu gõ tay; câu trả lời -> VieNeu đọc to. Backend agent không đổi.
   ===================================================================== */
const CONFIG = {
  // Mô hình xe: tệp tải sẵn trong web/public (chạy không cần mạng), CDN chỉ là dự phòng.
  CAR_URLS: [
    "/car.glb",
    "https://cdn.jsdelivr.net/gh/KhronosGroup/glTF-Sample-Assets@main/Models/CarConcept/glTF-Binary/CarConcept.glb",
  ],
  GIONG_NOI_URL: "http://localhost:8001",   // uvicorn serve.giong_noi:tao_app_that --factory --port 8001
};

import "@fontsource/be-vietnam-pro/400.css";
import "@fontsource/be-vietnam-pro/500.css";
import "@fontsource/be-vietnam-pro/600.css";
import "./nexa.css";
import { NexaSocket } from "../ws";
import * as THREE from "three";
import { Timer } from "three/addons/misc/Timer.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { Sky } from "three/addons/objects/Sky.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { mergeVertices } from "three/addons/utils/BufferGeometryUtils.js";

const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ---------------- Renderer, cảnh, camera ---------------- */
const stage = document.getElementById("stage");
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;
stage.prepend(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(34, 1, 0.5, 6000);
camera.position.set(5.6, 2.0, -6.4);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(-0.1, 0.72, 0);
controls.enableDamping = true;
controls.enablePan = false;
controls.minDistance = 1.2;
controls.maxDistance = 16;
controls.maxPolarAngle = 1.48;
controls.minPolarAngle = 0.35;

const pmrem = new THREE.PMREMGenerator(renderer);
const roomEnv = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

/* ---------------- Tiện ích tạo texture ---------------- */
function canvasTex(size, draw, repeat) {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  draw(c.getContext("2d"), size);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  if (repeat) { t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(...repeat); }
  return t;
}
function noise(ctx, s, base, spread, n) {
  ctx.fillStyle = base; ctx.fillRect(0, 0, s, s);
  for (let i = 0; i < n; i++) {
    const v = (Math.random() - .5) * spread;
    ctx.fillStyle = `rgba(${v > 0 ? 255 : 0},${v > 0 ? 255 : 0},${v > 0 ? 255 : 0},${Math.abs(v)})`;
    ctx.fillRect(Math.random() * s, Math.random() * s, 1 + Math.random() * 2, 1 + Math.random() * 2);
  }
}

/* ---------------- Mặt đất, đường ---------------- */
const ROAD_R = 1.95;          // mép phải của làn xe (phía vỉa hè)
const ROAD_L = -5.75;         // mép đường phía bên kia
const ROAD_W = ROAD_R - ROAD_L;
const world = new THREE.Group();   // những thứ trôi khi xe chạy
scene.add(world);

const asphaltTex = canvasTex(512, (c, s) => noise(c, s, "#47484a", .16, 26000), [350, 1.9]);
const road = new THREE.Mesh(
  new THREE.PlaneGeometry(1400, ROAD_W),
  new THREE.MeshStandardMaterial({ map: asphaltTex, roughness: .92, metalness: 0 })
);
road.rotation.x = -Math.PI / 2;
road.position.set(0, 0, (ROAD_R + ROAD_L) / 2);
road.receiveShadow = true;
scene.add(road);

const grassTex = canvasTex(256, (c, s) => noise(c, s, "#56653f", .25, 12000), [260, 260]);
const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(3000, 3000),
  new THREE.MeshStandardMaterial({ map: grassTex, roughness: 1 })
);
ground.rotation.x = -Math.PI / 2;
ground.position.y = -0.06;
ground.receiveShadow = true;
scene.add(ground);

// vạch kẻ đường
const paint = new THREE.MeshStandardMaterial({ color: "#d9d6cc", roughness: .7 });
const edgeGeo = new THREE.PlaneGeometry(1400, .14);
for (const z of [ROAD_R - .22, ROAD_L + .22]) {
  const m = new THREE.Mesh(edgeGeo, paint);
  m.rotation.x = -Math.PI / 2; m.position.set(0, .006, z); m.receiveShadow = true;
  scene.add(m);
}
const DASH_SPAN = 9, DASH_N = 70;
const dashes = new THREE.InstancedMesh(new THREE.PlaneGeometry(3, .13), paint, DASH_N);
dashes.receiveShadow = true;
const dashX = [];
for (let i = 0; i < DASH_N; i++) dashX.push(-300 + i * DASH_SPAN);
scene.add(dashes);
const tmpM = new THREE.Matrix4(), tmpQ = new THREE.Quaternion().setFromEuler(new THREE.Euler(-Math.PI / 2, 0, 0)), tmpS = new THREE.Vector3(1, 1, 1);
function layoutDashes() {
  for (let i = 0; i < DASH_N; i++) {
    tmpM.compose(new THREE.Vector3(dashX[i], .006, -1.9), tmpQ, tmpS);
    dashes.setMatrixAt(i, tmpM);
  }
  dashes.instanceMatrix.needsUpdate = true;
}
layoutDashes();

// vỉa hè + bó vỉa
const concrete = new THREE.MeshStandardMaterial({ color: "#8f8d86", roughness: .95 });
const curbMat = new THREE.MeshStandardMaterial({ color: "#a9a69e", roughness: .9 });
const tileTex = canvasTex(256, (c, s) => {
  noise(c, s, "#8b8982", .18, 5000);
  c.strokeStyle = "rgba(40,40,40,.35)"; c.lineWidth = 2;
  for (let i = 0; i <= 4; i++) { c.beginPath(); c.moveTo(i * s / 4, 0); c.lineTo(i * s / 4, s); c.stroke(); c.beginPath(); c.moveTo(0, i * s / 4); c.lineTo(s, i * s / 4); c.stroke(); }
}, [700, 1.5]);
const walk = new THREE.Mesh(new THREE.BoxGeometry(1400, .14, 3.2), new THREE.MeshStandardMaterial({ map: tileTex, roughness: .95 }));
walk.position.set(0, .07, ROAD_R + .18 + 1.6); walk.receiveShadow = true; scene.add(walk);
for (const [z, w] of [[ROAD_R + .09, .18], [ROAD_L - .09, .18]]) {
  const c = new THREE.Mesh(new THREE.BoxGeometry(1400, .16, w), curbMat);
  c.position.set(0, .08, z); c.receiveShadow = true; scene.add(c);
}
const farWalk = new THREE.Mesh(new THREE.BoxGeometry(1400, .12, 1.6), concrete);
farWalk.position.set(0, .06, ROAD_L - .18 - .8); farWalk.receiveShadow = true; scene.add(farWalk);

// đồi xa làm đường chân trời
const hillMat = new THREE.MeshStandardMaterial({ color: "#4a5546", roughness: 1 });
for (let i = 0; i < 14; i++) {
  const h = new THREE.Mesh(lumpy(1, i * 1.3, .12, 4), hillMat);
  const a = -2.4 + i * .42, r = 520 + Math.random() * 140;
  h.position.set(Math.cos(a) * r - 80, -10, Math.sin(a) * r * .9 + 140);
  h.scale.set(120 + Math.random() * 90, 40 + Math.random() * 45, 90 + Math.random() * 60);
  scene.add(h);
}

/* ---------------- Đèn đường và cây ---------------- */
const SCROLL_MIN = -230, SCROLL_LEN = 360;   // vật thể trôi trong khoảng [-230, 130]
const scrollers = [];   // { obj, x }
const poleMat = new THREE.MeshStandardMaterial({ color: "#5d6267", roughness: .5, metalness: .6 });
const lampHeadMat = new THREE.MeshStandardMaterial({ color: "#2b2f33", roughness: .6 });
const lampGlassMat = new THREE.MeshStandardMaterial({ color: "#fff3df", emissive: "#ffd7a1", emissiveIntensity: 0 });
const lampLights = [];
function makeLamp() {
  const g = new THREE.Group();
  const pole = new THREE.Mesh(new THREE.CylinderGeometry(.07, .1, 7.2, 12), poleMat);
  pole.position.y = 3.6; pole.castShadow = true; g.add(pole);
  const arm = new THREE.Mesh(new THREE.CylinderGeometry(.05, .05, 1.6, 8), poleMat);
  arm.rotation.x = Math.PI / 2; arm.position.set(0, 7.1, -.8); g.add(arm);
  const head = new THREE.Mesh(new RoundedBoxGeometry(.7, .16, .34, 3, .06), lampHeadMat);
  head.position.set(0, 7.08, -1.6); g.add(head);
  const glass = new THREE.Mesh(new THREE.PlaneGeometry(.6, .26), lampGlassMat);
  glass.rotation.x = Math.PI / 2; glass.position.set(0, 6.995, -1.6); g.add(glass);
  const light = new THREE.PointLight("#ffd29a", 0, 26, 1.6);
  light.position.set(0, 6.8, -1.6); g.add(light);
  lampLights.push(light);
  return g;
}
for (let i = 0; i < 12; i++) {
  const lamp = makeLamp();
  const x = SCROLL_MIN + i * 30 - 4;
  lamp.position.set(x, .14, ROAD_R + .8);
  world.add(lamp);
  scrollers.push({ obj: lamp, x });
}
const trunkMat = new THREE.MeshStandardMaterial({ color: "#4a3f35", roughness: .95 });
const leafMats = ["#46573a", "#506240", "#3d4e35"].map(c => new THREE.MeshStandardMaterial({ color: c, roughness: .9 }));
// khối tán lá: cầu lồi lõm theo một hàm nhiễu của vị trí, đổ bóng mềm
function lumpy(r, seed, amp = .2, detail = 3) {
  let g = new THREE.IcosahedronGeometry(r, detail);
  g.deleteAttribute("normal"); g.deleteAttribute("uv");
  g = mergeVertices(g);
  const p = g.attributes.position, v = new THREE.Vector3();
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i);
    const n = Math.sin(v.x * 3.1 / r + seed) * Math.sin(v.y * 2.3 / r + seed * 1.7) * Math.sin(v.z * 2.9 / r + seed * .6)
            + .5 * Math.sin(v.x * 7.3 / r + seed * 2.1) * Math.sin(v.z * 6.1 / r - seed);
    v.multiplyScalar(1 + amp * n);
    p.setXYZ(i, v.x, v.y, v.z);
  }
  g.computeVertexNormals();
  return g;
}
function makeTree() {
  const g = new THREE.Group();
  const h = 2.2 + Math.random() * 1.6;
  const t = new THREE.Mesh(new THREE.CylinderGeometry(.12, .17, h, 7), trunkMat);
  t.position.y = h / 2; t.castShadow = true; g.add(t);
  const n = 2 + (Math.random() * 2 | 0);
  for (let i = 0; i < n; i++) {
    const r = 1.1 + Math.random() * .7;
    const f = new THREE.Mesh(lumpy(r, Math.random() * 10), leafMats[(i + (Math.random() * 3 | 0)) % 3]);
    f.scale.y = .85;
    f.position.set((Math.random() - .5) * 1.1, h + r * .55 + i * .45, (Math.random() - .5) * 1.1);
    f.castShadow = true; g.add(f);
  }
  return g;
}
for (let i = 0; i < 36; i++) {
  const t = makeTree();
  const x = SCROLL_MIN + i * 10 + Math.random() * 4;
  const side = i % 3 === 0 ? ROAD_L - 3.2 - Math.random() * 3 : ROAD_R + 4.8 + Math.random() * 5;
  t.position.set(x, 0, side);
  world.add(t);
  scrollers.push({ obj: t, x });
}

/* ---------------- Bầu trời ---------------- */
const sky = new Sky();
sky.scale.setScalar(4500);
const skyU = sky.material.uniforms;
skyU.turbidity.value = 4; skyU.rayleigh.value = 1.4; skyU.mieCoefficient.value = .004; skyU.mieDirectionalG.value = .82;
const sunDir = new THREE.Vector3().setFromSphericalCoords(1, THREE.MathUtils.degToRad(90 - 34), THREE.MathUtils.degToRad(125));
skyU.sunPosition.value.copy(sunDir);
scene.add(sky);

const nightDome = new THREE.Mesh(
  new THREE.SphereGeometry(4000, 32, 16),
  new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false, fog: false,
    uniforms: { top: { value: new THREE.Color("#060c18") }, bottom: { value: new THREE.Color("#22314a") } },
    vertexShader: "varying vec3 vP; void main(){ vP = normalize(position); gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.); }",
    fragmentShader: "uniform vec3 top; uniform vec3 bottom; varying vec3 vP; void main(){ float h = clamp(vP.y*1.6+.08,0.,1.); gl_FragColor = vec4(mix(bottom, top, pow(h,.7)),1.); }",
  })
);
scene.add(nightDome);

// sao
const starGeo = new THREE.BufferGeometry();
const starPos = [], starCol = [];
for (let i = 0; i < 2200; i++) {
  const th = Math.random() * Math.PI * 2, ph = Math.acos(Math.random() * .92 + .06);
  const r = 3500;
  starPos.push(r * Math.sin(ph) * Math.cos(th), r * Math.cos(ph), r * Math.sin(ph) * Math.sin(th));
  const b = .55 + Math.random() * .45, warm = Math.random() < .2;
  starCol.push(b, b * (warm ? .92 : 1), b * (warm ? .82 : 1));
}
starGeo.setAttribute("position", new THREE.Float32BufferAttribute(starPos, 3));
starGeo.setAttribute("color", new THREE.Float32BufferAttribute(starCol, 3));
const stars = new THREE.Points(starGeo, new THREE.PointsMaterial({ size: 1.7, sizeAttenuation: false, vertexColors: true, fog: false, transparent: true, depthWrite: false }));
scene.add(stars);

// trăng
const moonTex = canvasTex(256, (c, s) => {
  c.fillStyle = "#e9e5da"; c.fillRect(0, 0, s, s);
  for (let i = 0; i < 70; i++) {
    const r = 4 + Math.random() * 22;
    c.fillStyle = `rgba(120,118,110,${.08 + Math.random() * .18})`;
    c.beginPath(); c.arc(Math.random() * s, Math.random() * s, r, 0, Math.PI * 2); c.fill();
  }
});
const moonPos = new THREE.Vector3().setFromSphericalCoords(3200, THREE.MathUtils.degToRad(90 - 7.5), Math.atan2(-.5, .87));
const moon = new THREE.Mesh(new THREE.SphereGeometry(70, 48, 24), new THREE.MeshBasicMaterial({ map: moonTex, fog: false }));
moon.position.copy(moonPos);
scene.add(moon);
const glowTex = canvasTex(128, (c, s) => {
  const g = c.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
  g.addColorStop(0, "rgba(220,228,255,.55)"); g.addColorStop(.3, "rgba(170,190,240,.18)"); g.addColorStop(1, "rgba(150,170,230,0)");
  c.fillStyle = g; c.fillRect(0, 0, s, s);
});
const moonGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, transparent: true, depthWrite: false, fog: false, blending: THREE.AdditiveBlending }));
moonGlow.scale.setScalar(900); moonGlow.position.copy(moonPos);
scene.add(moonGlow);

/* ---------------- Ánh sáng ---------------- */
const hemi = new THREE.HemisphereLight("#cfe0f2", "#5b5a50", .6);
scene.add(hemi);
const key = new THREE.DirectionalLight("#fff4e6", 2.4);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
key.shadow.camera.left = -9; key.shadow.camera.right = 9; key.shadow.camera.top = 9; key.shadow.camera.bottom = -9;
key.shadow.camera.near = 1; key.shadow.camera.far = 80;
key.shadow.bias = -.0004; key.shadow.normalBias = .02;
scene.add(key, key.target);

/* ---------------- Chiếc xe: mô hình glTF ----------------
   Mô hình "Car Concept" của Khronos glTF Sample Assets (CC-BY 4.0),
   dựng từ mẫu public domain của Unity Fan. Logo Khronos đã được thay.
   Trong mô hình: đầu xe là +z, bên trái xe là +x. Ở cảnh này xe xoay
   để đầu xe hướng +x, bên trái (ghế lái) là -z. */
const car = new THREE.Group();
scene.add(car);
const model = new THREE.Group();
model.rotation.y = Math.PI / 2;
model.position.set(-.24, 0, 0);
car.add(model);

const trunkState = { open: 0, target: 0 };
const flapState = { open: 0, target: 0 };
const sunroofState = { slide: 0, tilt: 0 };
const WIN_IDS = ["front_left", "front_right", "rear_left", "rear_right"];
const windows = Object.fromEntries(WIN_IDS.map(id => [id, { open: 0, target: 0, pivot: null }]));
const rig = { hatch: null, roof: null, wheels: [], ready: false };
const MAT = {};   // vật liệu đèn lấy từ mô hình
const carMats = [];   // vật liệu của xe: phản chiếu bầu trời mạnh hơn mặt đường
const WHEEL_R = .54;

// đèn pha chiếu ra đường, đèn hậu hắt xuống mặt đường
const beams = [];
for (const s of [-1, 1]) {
  const sp = new THREE.SpotLight("#fff2dd", 0, 60, .42, .55, 1.4);
  sp.position.set(s * .62, .66, 2.25);
  sp.target.position.set(s * .7, -1.3, 24);
  model.add(sp, sp.target);
  beams.push(sp);
}
const tailGlow = new THREE.PointLight("#ff2a1f", 0, 4, 2);
tailGlow.position.set(0, .9, -2.3); model.add(tailGlow);

// quầng sáng xi nhan ở bốn góc xe
const amberTex = canvasTex(64, (c, s) => {
  const g = c.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
  g.addColorStop(0, "rgba(255,214,140,1)"); g.addColorStop(.25, "rgba(255,150,30,.75)"); g.addColorStop(1, "rgba(255,120,0,0)");
  c.fillStyle = g; c.fillRect(0, 0, s, s);
});
const blinkers = [];
for (const [x, y, z] of [[.84, .66, 2.2], [-.84, .66, 2.2], [.8, .98, -1.4], [-.8, .98, -1.4]]) {
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: amberTex, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, opacity: 0 }));
  sp.position.set(x, y, z); sp.scale.setScalar(.42);
  model.add(sp); blinkers.push(sp);
}

// bóng mờ dưới gầm
const blob = new THREE.Mesh(new THREE.PlaneGeometry(5.2, 2.7), new THREE.MeshBasicMaterial({
  map: canvasTex(128, (c, s) => { const g = c.createRadialGradient(s / 2, s / 2, 4, s / 2, s / 2, s / 2); g.addColorStop(0, "rgba(0,0,0,.6)"); g.addColorStop(1, "rgba(0,0,0,0)"); c.fillStyle = g; c.fillRect(0, 0, s, s); }),
  transparent: true, depthWrite: false,
}));
blob.rotation.x = -Math.PI / 2; blob.position.y = .012; car.add(blob);

function plateTexture() {
  const t = canvasTex(512, (c, s) => {
    c.fillStyle = "#eeeeea"; c.fillRect(0, 0, s, s);
    c.strokeStyle = "#1b1b1b"; c.lineWidth = 14; c.strokeRect(10, s * .3, s - 20, s * .4);
    c.fillStyle = "#1b1b1b"; c.font = "700 92px 'Be Vietnam Pro', Arial, sans-serif";
    c.textAlign = "center"; c.textBaseline = "middle"; c.fillText("30K-123.45", s / 2, s / 2 + 4);
  });
  t.flipY = false;
  return t;
}

// gói một bộ phận vào khớp xoay đặt tại `at` (toạ độ mô hình), giữ nguyên vị trí hiện tại
function pivotAt(at, nodes) {
  const p = new THREE.Group(); p.position.copy(at);
  model.add(p); model.updateMatrixWorld(true);
  for (const n of nodes) if (n) p.attach(n);
  return p;
}

function setupModel(root) {
  model.add(root);
  root.updateMatrixWorld(true);
  const byName = n => root.getObjectByName(n);

  root.traverse(o => {
    if (!o.isMesh) return;
    o.castShadow = o.material.name !== "Glass";
    o.receiveShadow = true;
  });
  // màu giản dị: thân bạc ánh xám, phần ốp dưới và nóc màu than, nội thất ghi
  const recolor = {
    "Paint 1 Carmine": m => { m.color.set("#979da3"); m.metalness = .6; m.roughness = .3; m.clearcoat = 1; m.clearcoatRoughness = .05; MAT.paint = m; },
    "Paint 2 Carmine": m => { m.color.set("#26292d"); m.metalness = .4; m.roughness = .4; },
    "Interior 3 Carmine": m => { m.color.set("#6b655e"); },
    "Brake": m => { m.color.set("#3b3e42"); },
    // hông lốp có chữ/logo Khronos: bỏ texture, giữ màu cao su
    "Tireside": m => { m.map = null; m.color.set("#1b1c1e"); m.roughness = .85; m.needsUpdate = true; },
  };
  const seen = new Set();
  root.traverse(o => {
    if (!o.isMesh) return;
    const list = Array.isArray(o.material) ? o.material : [o.material];
    for (const m of list) {
      if (seen.has(m)) continue; seen.add(m);
      recolor[m.name]?.(m);
      carMats.push(m);
      if (m.name === "Headlight") MAT.head = m;
      if (m.name === "Brakelight") MAT.tail = m;
      if (m.name === "Signallight") MAT.sig = m;
      if (m.name === "License") { m.map = plateTexture(); m.needsUpdate = true; }
    }
  });
  if (MAT.head) MAT.head.emissive.set("#fff4e6");
  if (MAT.tail) MAT.tail.emissive.set("#ff2216");
  if (MAT.sig) MAT.sig.emissive.set("#ff9a1a");
  const emblem = byName("InteriorSteeringEmblem"); if (emblem) emblem.visible = false;

  // cốp: kính sau + khung nắp, bản lề ở mép trên phía trước
  rig.hatch = pivotAt(new THREE.Vector3(0, 1.14, -.44), [byName("BodyRearwindow"), byName("InteriorRearHatch")]);
  // cửa sổ trời: tấm nóc, bản lề ở mép sau để hé
  rig.roof = pivotAt(new THREE.Vector3(0, 1.135, -.45), [byName("BodyRoofPanel")]);
  rig.roofBase = rig.roof.position.clone();
  // kính cửa: xe 2 cửa nên kính sau dùng chung kính trước cùng bên
  const wl = pivotAt(new THREE.Vector3(), [byName("BodyDoorLWindow")]);
  const wr = pivotAt(new THREE.Vector3(), [byName("BodyDoorRWindow")]);
  windows.front_left.pivot = wl; windows.front_right.pivot = wr;
  // bánh xe: khớp quay theo trục ngang của xe
  for (const n of ["WheelFrontL", "WheelFrontR", "WheelRearL", "WheelRearR"]) {
    const w = byName(n); if (!w) continue;
    // bánh trước trong mô hình gốc bị đánh lái và nghiêng sẵn: trả về thẳng để quay đúng trục
    if (n.startsWith("WheelFront")) { w.rotation.set(0, 0, 0); w.updateMatrixWorld(true); }
    const at = w.getWorldPosition(new THREE.Vector3()); model.worldToLocal(at);
    rig.wheels.push(pivotAt(at, [w]));
  }
  buildExtras(root, byName);
  rig.ready = true;
  applyTime(isNight);
}

async function loadCar() {
  const loader = new GLTFLoader();
  setChip("load", "Đang tải mô hình xe…");
  for (const url of CONFIG.CAR_URLS) {
    try {
      const gltf = await loader.loadAsync(url);
      setupModel(gltf.scene);
      setChip("load", null);
      if (xeBackend) dongBo(xeBackend);   // trạng thái đã về trước khi xe tải xong -> vẽ lại cho khớp
      return;
    } catch (e) { console.warn("Không tải được", url, e); }
  }
  setChip("load", "Không tải được mô hình xe. Kiểm tra kết nối mạng rồi tải lại trang.");
}

/* ---------------- Ngày / đêm ---------------- */
let isNight = true;
function applyTime(night) {
  isNight = night;
  sky.visible = !night;
  nightDome.visible = stars.visible = moon.visible = moonGlow.visible = night;
  if (night) {
    scene.background = null;
    scene.environment = roomEnv; scene.environmentIntensity = .14;
    carMats.forEach(m => { m.envMap = roomEnv; m.envMapIntensity = .35; });
    scene.fog = new THREE.Fog("#1b2639", 25, 330);
    hemi.color.set("#3a4d6e"); hemi.groundColor.set("#101318"); hemi.intensity = .55;
    key.color.set("#a9bde0"); key.intensity = .55;
    key.position.copy(moonPos).normalize().multiplyScalar(40);
    renderer.toneMappingExposure = 1.0;
    lampLights.forEach(l => l.intensity = 38); lampGlassMat.emissiveIntensity = 2.2;
    hillMat.color.set("#1d2530");
  } else {
    scene.environment = dayEnv || (dayEnv = pmrem.fromScene(skyEnvScene(), 0).texture); scene.environmentIntensity = .3;
    carMats.forEach(m => { m.envMap = dayEnv; m.envMapIntensity = .9; });
    scene.fog = new THREE.Fog("#c4cfd8", 140, 900);
    hemi.color.set("#eef0ec"); hemi.groundColor.set("#6b6450"); hemi.intensity = .9;
    key.color.set("#fff1de"); key.intensity = 2.8;
    key.position.copy(sunDir).multiplyScalar(40);
    renderer.toneMappingExposure = .72;
    lampLights.forEach(l => l.intensity = 0); lampGlassMat.emissiveIntensity = 0;
    hillMat.color.set("#5d6a57");
  }
  updateLights();
  document.querySelectorAll("[data-time]").forEach(b => b.setAttribute("aria-pressed", String((b.dataset.time === "night") === night)));
}
let skyEnvCache, dayEnv;
function skyEnvScene() {
  if (!skyEnvCache) { skyEnvCache = new THREE.Scene(); const s = new Sky(); s.scale.setScalar(4500); Object.assign(s.material.uniforms, {}); s.material.uniforms.sunPosition.value.copy(sunDir); for (const k of ["turbidity", "rayleigh", "mieCoefficient", "mieDirectionalG"]) s.material.uniforms[k].value = skyU[k].value; skyEnvCache.add(s); }
  return skyEnvCache;
}

/* ---------------- Trạng thái xe ---------------- */
const vehicle = {
  mode: "parked", child_lock: false, locked: true,
  headlights: "off", hazard: false,
  sunroof: { action: "close", intensity: 0 },
  ac: null, fan: null, defrost: null, volume: 15, media: null, nav: null, navAvoid: "none", nearby: null,
  doorLocks: { driver: true, passenger: true, rear_left: true, rear_right: true },
};
let lockBlinkUntil = 0;
function updateLights() {
  const mode = vehicle.headlights;
  const on = mode === "low_beam" || mode === "high_beam" || (mode === "auto" && isNight);
  const parking = mode === "parking" || on;
  if (MAT.head) MAT.head.emissiveIntensity = on ? (mode === "high_beam" ? 4 : 2.6) : (parking ? .5 : 0);
  if (MAT.tail) MAT.tail.emissiveIntensity = parking ? 2.2 : 0;
  tailGlow.intensity = parking && isNight ? 1.2 : 0;
  beams.forEach(b => {
    b.intensity = on ? (mode === "high_beam" ? 420 : 160) : 0;
    b.angle = mode === "high_beam" ? .36 : .46;
    b.target.position.y = mode === "high_beam" ? .2 : -1.2;
    b.distance = mode === "high_beam" ? 110 : 55;
  });
}

/* ---------------- Camera di chuyển theo hành động ---------------- */
const VIEWS = {
  home: { p: [5.4, 1.9, -6.2], t: [0, .6, 0] },
  rear: { p: [-6.4, 2.6, -3.4], t: [-1.1, .8, 0] },
  left: { p: [.3, 1.45, -4.9], t: [-.4, .85, 0] },
  right: { p: [.3, 1.45, 4.9], t: [-.4, .85, 0] },
  top: { p: [1.5, 4.0, -2.6], t: [-.35, 1.0, 0] },
  front: { p: [7.0, 1.3, -2.8], t: [1.4, .5, 0] },
  fuel: { p: [-1.9, 1.2, 2.9], t: [-1.0, .74, 1.05] },
  charge: { p: [-1.9, 1.2, -2.9], t: [-1.0, .74, -1.05] },
  doorLeft: { p: [1.4, 1.55, -4.6], t: [-.3, .85, -.6] },
  doorRight: { p: [1.4, 1.55, 4.6], t: [-.3, .85, .6] },
  windshield: { p: [4.3, 2.4, -1.5], t: [.75, .95, 0] },
  rearGlass: { p: [-4.4, 2.9, -1.3], t: [-1.0, 1.0, 0] },
  hazard: { p: [-5.4, 1.8, -4.4], t: [-.3, .65, 0] },
  status: { p: [4.6, 3.7, -5.4], t: [-.1, .6, 0] },
  tires: { p: [3.6, 3.4, -4.8], t: [-.1, .3, 0] },
  tire_front_left: { p: [3.4, 1.1, -3.4], t: [1.25, .42, -.9] },
  tire_front_right: { p: [3.4, 1.1, 3.4], t: [1.25, .42, .9] },
  tire_rear_left: { p: [-3.7, 1.1, -3.2], t: [-1.55, .42, -.9] },
  tire_rear_right: { p: [-3.7, 1.1, 3.2], t: [-1.55, .42, .9] },
  // trong xe: tầm mắt người ngồi ghế phụ, nhìn về táp-lô
  cabin: { p: [.36, .95, .22], t: [1.0, .76, .3], tPortrait: [.97, .79, .45], inside: true },
};
let camTween = null, wantInside = false;
const viewBtn = document.getElementById("viewBtn");
const fovFor = inside => inside ? (camera.aspect < 1 ? 74 : 60) : (camera.aspect < 1 ? 46 : 34);
// Ở trong xe: điểm nhìn đặt sát trước mắt, kéo chuột là quay đầu nhìn quanh
function applyViewLimits(inside) {
  controls.minDistance = inside ? .12 : 1.2;
  controls.maxDistance = inside ? .16 : 16;
  controls.enableZoom = !inside;
  controls.rotateSpeed = inside ? .45 : 1;
  controls.minPolarAngle = inside ? .7 : .35;
  controls.maxPolarAngle = inside ? 2.1 : 1.48;
  camera.near = inside ? .02 : .25;
  camera.fov = fovFor(inside);
  camera.updateProjectionMatrix();
}
function looseLimits() { controls.minDistance = .01; controls.maxDistance = 100; controls.minPolarAngle = 0; controls.maxPolarAngle = Math.PI; }
function focus(name, instant) {
  const v = VIEWS[name]; if (!v) return;
  const inside = !!v.inside;
  const to = { p: new THREE.Vector3(...v.p), t: new THREE.Vector3(...(camera.aspect < 1 && v.tPortrait ? v.tPortrait : v.t)) };
  if (inside) to.t.sub(to.p).setLength(.14).add(to.p);
  // màn hình dọc (điện thoại): lùi camera ra để thấy trọn xe
  else if (camera.aspect < 1) to.p.sub(to.t).multiplyScalar(1.35).add(to.t);
  if (inside === wantInside && name === "cabin" && !camTween && camera.position.distanceTo(to.p) < .05) return;
  wantInside = inside;
  if (viewBtn) { viewBtn.textContent = inside ? "Ra ngoài xe" : "Vào trong xe"; viewBtn.setAttribute("aria-pressed", String(inside)); }
  if (instant || reduceMotion) { camera.position.copy(to.p); controls.target.copy(to.t); applyViewLimits(inside); camTween = null; return; }
  looseLimits();
  if (inside) { camera.near = .02; camera.updateProjectionMatrix(); }
  camTween = { from: { p: camera.position.clone(), t: controls.target.clone(), fov: camera.fov }, to: { ...to, fov: fovFor(inside) }, inside, t0: performance.now(), dur: 1400 };
}
function finishTween() {
  if (!camTween) return;
  camera.position.copy(camTween.to.p); controls.target.copy(camTween.to.t);
  applyViewLimits(camTween.inside); camTween = null;
}

/* ---------------- Áp lệnh của model lên xe ---------------- */
const VN = {
  zone: { driver: "ghế lái", passenger: "ghế phụ", rear: "hàng ghế sau", all: "cả xe", front: "kính trước", both: "kính trước và sau" },
  cat: { gas_station: "cây xăng", charging_station: "trạm sạc", parking: "bãi đỗ", restaurant: "quán ăn", hospital: "bệnh viện", atm: "ATM", coffee: "quán cà phê" },
  source: { radio: "radio", bluetooth: "Bluetooth", usb: "USB", streaming: "trực tuyến" },
};
const DEF_VN = { front: "kính trước", rear: "kính sau", both: "kính trước và sau" };
const chipsEl = document.getElementById("chips");
const chipData = new Map();
let ketNoi = "connecting";   // kết nối WebSocket tới backend: connecting | open | closed
let xeBackend = null;        // ảnh chụp trạng thái xe mới nhất từ backend (StateEvent.vehicle)
function setChip(key, text) {
  if (text == null) chipData.delete(key); else chipData.set(key, text);
  renderChips(key);
}
// Chip cố định: trạng thái kết nối + số liệu chính (xăng, pin, điều hoà, cửa, âm lượng) như ô HUD của giao diện trước
function renderChips(fresh) {
  chipsEl.innerHTML = "";
  const add = (text, cls = "") => { const d = document.createElement("div"); d.className = "chip " + cls; d.textContent = text; chipsEl.append(d); };
  if (ketNoi !== "open") add(ketNoi === "connecting" ? "Đang kết nối backend…" : "Mất kết nối backend — kiểm tra serve.app đang chạy", ketNoi === "closed" ? "off" : "mock");
  const s = xeBackend;
  if (s) add(`Xăng ${s.fuel_percent}% · Pin ${s.battery_percent}% · Điều hoà ${s.ac_on ? s.ac_temperature + "°" : "tắt"} · Cửa ${s.doors_locked ? "đã khoá" : "mở khoá"} · Âm lượng ${s.volume}`);
  for (const [k, t] of chipData) add(t, k === fresh ? "flash" : "");
}
// tên kính chấp nhận cả cách gọi gộp: 2 kính trước / 2 kính sau / ghế lái / ghế phụ
const WINDOW_ALIAS = {
  all: WIN_IDS, front: ["front_left", "front_right"], front_both: ["front_left", "front_right"],
  rear: ["rear_left", "rear_right"], rear_both: ["rear_left", "rear_right"],
  driver: ["front_left"], passenger: ["front_right"],
};
const DOOR_VN = { driver: "Cửa ghế lái", passenger: "Cửa ghế phụ", rear_left: "Cửa sau trái", rear_right: "Cửa sau phải" };
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
function blinkLock(times) { lockBlinkUntil = performance.now() + 550 * times; }
function applyCall({ name, args = {} }, kq = {}) {
  switch (name) {
    /* ---- tra cứu ---- */
    case "get_fuel_level": case "get_battery_range": case "get_tire_pressure": case "get_vehicle_status":
    case "get_odometer": case "get_range_estimate": case "get_weather":
      runQuery(name, args, kq); break;

    /* ---- điều hoà ---- */
    case "set_ac": {
      const prev = vehicle.ac || {};
      const on = args.state ? args.state !== "off" : true;
      let temp = args.temperature != null ? +args.temperature : (prev.temperature ?? 22);
      if (args.delta != null) temp += +args.delta;
      vehicle.ac = { state: on ? "on" : "off", temperature: clamp(temp, 16, 30), zone: args.zone || prev.zone || "all", mode: args.mode || prev.mode || "auto" };
      showMain("climate"); focus("cabin");
      setChip("ac", on ? `Điều hoà ${vehicle.ac.temperature}°C${vehicle.ac.zone !== "all" ? ", " + (VN.zone[vehicle.ac.zone] || vehicle.ac.zone) : ""}` : "Điều hoà tắt");
      break;
    }
    case "set_fan_speed": {
      let lv = args.level != null ? +args.level : (vehicle.fan ?? 3);
      if (args.delta != null) lv += +args.delta;
      vehicle.fan = clamp(Math.round(lv), 0, 7);
      showMain("climate"); focus("cabin");
      setChip("fan", vehicle.fan === 0 ? "Quạt gió tắt" : `Quạt gió mức ${vehicle.fan}`); break;
    }
    case "set_defrost": {
      const zone = args.zone || "front", on = args.state !== "off";
      vehicle.defrost = { state: on ? "on" : "off", zone };
      if (on) for (const k of zone === "both" ? ["front", "rear"] : [zone]) { const f = extras.fog[k]; if (f) { f.amt = 1; f.clear = -.15; } }
      showMain("climate");
      focus(zone === "rear" ? "rearGlass" : zone === "both" ? "rearGlass" : "windshield");
      setChip("defrost", on ? `Sấy ${VN.zone[zone] || "kính"} bật` : null); break;
    }
    case "set_wipers": {   // chưa có trong bảng hàm, để sẵn nếu thêm sau
      wipers.state = args.state || "on"; wipers.speed = args.speed || "low";
      focus("windshield"); setChip("wipers", wipers.state === "off" ? null : "Gạt mưa đang chạy"); break;
    }

    /* ---- cửa, cốp, nắp ---- */
    case "lock_doors":
      vehicle.locked = true; vehicle.doorLocks = { driver: true, passenger: true, rear_left: true, rear_right: true };
      blinkLock(1); focus("doorLeft");
      showCallouts([{ at: ANCHOR.driver, text: "Đã khoá" }, { at: ANCHOR.passenger, text: "Đã khoá" }], 5000);
      setChip("lock", "Đã khoá cửa"); break;
    case "unlock_doors": {
      const door = args.door || "all";
      const ids = door === "all" ? ["driver", "passenger", "rear_left", "rear_right"] : [door];
      for (const id of ids) vehicle.doorLocks[id] = false;
      vehicle.locked = Object.values(vehicle.doorLocks).every(Boolean);
      blinkLock(2);
      focus(door === "passenger" || door === "rear_right" ? "doorRight" : "doorLeft");
      showCallouts(ids.map(id => ({
        at: ANCHOR[id],
        text: id.startsWith("rear") && vehicle.child_lock ? `${DOOR_VN[id]}: mở khoá, khoá trẻ em vẫn bật` : `${DOOR_VN[id]}: đã mở khoá`,
        warn: id.startsWith("rear") && vehicle.child_lock,
      })), 6000);
      setChip("lock", door === "all" ? "Đã mở khoá cửa" : `${DOOR_VN[door] || door} đã mở khoá`); break;
    }
    case "open_trunk": trunkState.target = 1; focus("rear"); break;
    case "close_trunk": trunkState.target = 0; focus("rear"); break;
    case "open_charge_port": case "close_charge_port":
      if (extras.ports.charge) extras.ports.charge.target = name === "open_charge_port" ? 1 : 0;
      focus("charge"); setChip("charge", name === "open_charge_port" ? "Cổng sạc mở" : null);
      showCallouts([{ at: ANCHOR.charge, text: name === "open_charge_port" ? "Cổng sạc: đã mở" : "Cổng sạc: đã đóng" }], 5000); break;
    case "open_fuel_flap": case "close_fuel_flap":
      if (extras.ports.fuel) extras.ports.fuel.target = name === "open_fuel_flap" ? 1 : 0;
      focus("fuel"); setChip("fuel", name === "open_fuel_flap" ? "Nắp bình xăng mở" : null);
      showCallouts([{ at: ANCHOR.fuel, text: name === "open_fuel_flap" ? "Nắp bình xăng: đã mở" : "Nắp bình xăng: đã đóng" }], 5000); break;

    /* ---- kính, cửa sổ trời ---- */
    case "control_window": {
      const ids = WINDOW_ALIAS[args.window || "all"] || [args.window];
      const amt = args.amount != null ? clamp(+args.amount, 0, 100) / 100 : null;
      for (const id of ids) {
        const w = windows[id]; if (!w) continue;
        w.target = args.position === "up" ? (amt != null ? Math.max(0, w.target - amt) : 0) : (amt ?? 1);
      }
      const right = ids.every(id => id.endsWith("right")), left = ids.every(id => id.endsWith("left"));
      const WIN_VN = { front_left: "Kính ghế lái", front_right: "Kính ghế phụ", rear_left: "Kính sau trái", rear_right: "Kính sau phải" };
      showCallouts(ids.filter(id => windows[id]).map(id => ({ at: ANCHOR["win_" + id], text: `${WIN_VN[id]}: ${windows[id].target ? "hạ " + Math.round(windows[id].target * 100) + "%" : "đóng"}` })), 6000);
      focus(right ? "right" : left ? "left" : "status"); break;
    }
    case "operate_sunroof": {
      const action = args.action || "open";
      vehicle.sunroof = { action, intensity: action === "open" ? clamp(+(args.intensity ?? 100), 0, 100) : action === "tilt" ? 100 : 0 };
      focus("top"); break;
    }

    /* ---- đèn ---- */
    case "set_headlights": vehicle.headlights = args.mode || "low_beam"; updateLights(); focus("front"); break;
    case "set_hazard_lights": vehicle.hazard = args.state !== "off"; focus("hazard"); setChip("hazard", vehicle.hazard ? "Đèn khẩn cấp bật" : null); break;

    /* ---- giải trí ---- */
    case "set_volume": {
      if (args.set_value != null) vehicle.volume = clamp(+args.set_value, 0, 40);
      else if (args.delta != null) vehicle.volume = clamp(vehicle.volume + +args.delta, 0, 40);
      showMain("audio"); focus("cabin");
      setChip("vol", `Âm lượng ${vehicle.volume}`); break;
    }
    case "play_media": vehicle.media = args; showMain("audio"); focus("cabin"); setChip("media", `Đang phát${args.query ? ": " + args.query : ""}${args.source ? " (" + (VN.source[args.source] || args.source) + ")" : ""}`); break;

    /* ---- dẫn đường ---- */
    case "navigate_to": vehicle.nav = args.destination; vehicle.navAvoid = args.avoid || "none"; vehicle.nearby = null; showMain("nav"); focus("cabin"); setChip("nav", `Dẫn đường tới ${args.destination}`); break;
    case "find_nearby": vehicle.nearby = args; vehicle.nav = null; showMain("nav"); focus("cabin"); setChip("near", `Tìm ${VN.cat[args.category] || args.category} gần đây`); break;

    default: break;   // irrelevant_function: chỉ có câu trả lời
  }
}

/* ---------------- Khoang lái: màn hình giữa, luồng gió, đèn trần ----------------
   Toạ độ theo xe: đầu xe +x, bên trái (ghế lái) -z. Vô lăng của mẫu xe nằm giữa táp-lô. */
const cabin = new THREE.Group();
car.add(cabin);

// màn hình giải trí đặt trên táp-lô, lệch sang ghế phụ
const SCREEN_W = 1024, SCREEN_H = 560;
const screenCanvas = document.createElement("canvas");
screenCanvas.width = SCREEN_W; screenCanvas.height = SCREEN_H;
const sctx = screenCanvas.getContext("2d");
const screenTex = new THREE.CanvasTexture(screenCanvas);
screenTex.colorSpace = THREE.SRGBColorSpace; screenTex.anisotropy = 8;
const screenGroup = new THREE.Group();
screenGroup.position.set(.93, .8, .43);
screenGroup.rotation.order = "YXZ";
screenGroup.rotation.y = -Math.PI / 2 - .22;
screenGroup.rotation.x = -.16;
const SCR = { w: .42, h: .23 };
const bezel = new THREE.Mesh(new RoundedBoxGeometry(SCR.w + .02, SCR.h + .02, .014, 3, .006),
  new THREE.MeshStandardMaterial({ color: "#14181c", roughness: .35, metalness: .3 }));
bezel.position.z = -.008;
const screenMesh = new THREE.Mesh(new THREE.PlaneGeometry(SCR.w, SCR.h), new THREE.MeshBasicMaterial({ map: screenTex, toneMapped: false }));
screenGroup.add(bezel, screenMesh);
cabin.add(screenGroup);
const screenGlow = new THREE.PointLight("#bcd7ff", .25, 1.2, 2);
screenGlow.position.set(.8, .82, .38); cabin.add(screenGlow);

// đèn trần: chỉ sáng khi camera ở trong xe
const domeLight = new THREE.PointLight("#ffe4c2", 0, 2.4, 1.6);
domeLight.position.set(.05, .96, 0); cabin.add(domeLight);

// luồng gió từ cửa gió điều hoà và sấy kính: hạt sáng mờ, màu theo nhiệt độ
const softTex = canvasTex(64, (c, s) => {
  const g = c.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
  g.addColorStop(0, "rgba(255,255,255,.9)"); g.addColorStop(.4, "rgba(255,255,255,.35)"); g.addColorStop(1, "rgba(255,255,255,0)");
  c.fillStyle = g; c.fillRect(0, 0, s, s);
});
const AIR_N = 420;
const airPos = new Float32Array(AIR_N * 3), airCol = new Float32Array(AIR_N * 3);
const airGeo = new THREE.BufferGeometry();
airGeo.setAttribute("position", new THREE.BufferAttribute(airPos, 3));
airGeo.setAttribute("color", new THREE.BufferAttribute(airCol, 3));
const airPts = new THREE.Points(airGeo, new THREE.PointsMaterial({
  size: .028, map: softTex, vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, sizeAttenuation: true,
}));
airPts.frustumCulled = false;
cabin.add(airPts);
const air = Array.from({ length: AIR_N }, () => ({ age: 1, life: 1, p: new THREE.Vector3(), v: new THREE.Vector3(), c: new THREE.Color() }));
const VENTS = { driver: [new THREE.Vector3(.92, .73, -.4)], passenger: [new THREE.Vector3(.92, .73, .4)], rear: [new THREE.Vector3(-.2, .62, 0)] };
let airSpawn = 0, airNext = 0;
const COLD = new THREE.Color("#7cc0ff"), WARM = new THREE.Color("#ffa65c"), MILD = new THREE.Color("#dfe8f0");
function acColor(t, mode) {
  if (mode === "heat") return WARM;
  if (mode === "cool") return COLD;
  if (t == null) return MILD;
  return t <= 22 ? MILD.clone().lerp(COLD, Math.min(1, (22 - t) / 4 + .55)) : MILD.clone().lerp(WARM, Math.min(1, (t - 22) / 5 + .3));
}
function spawnAir(pos, dir, spd, color, life) {
  const a = air[airNext]; airNext = (airNext + 1) % AIR_N;
  a.p.copy(pos).add(new THREE.Vector3((Math.random() - .5) * .05, (Math.random() - .5) * .03, (Math.random() - .5) * .08));
  a.v.copy(dir).add(new THREE.Vector3((Math.random() - .5) * .25, (Math.random() - .5) * .18, (Math.random() - .5) * .3)).normalize().multiplyScalar(spd * (.8 + Math.random() * .4));
  a.c.copy(color); a.age = 0; a.life = life * (.8 + Math.random() * .4);
}
function updateAir(dt) {
  const ac = vehicle.ac, acOn = ac && ac.state !== "off";
  const fan = acOn ? (vehicle.fan ?? 3) : (vehicle.fan || 0);
  const fanOn = fan > 0;
  const def = vehicle.defrost && vehicle.defrost.state !== "off" && vehicle.defrost.zone !== "rear";
  // tốc độ phát hạt
  airSpawn += dt * ((fanOn ? 30 + fan * 20 : 0) + (def ? 70 : 0));
  const col = acOn ? acColor(ac.temperature, ac.mode) : MILD;
  const zone = acOn ? (ac.zone || "all") : "all";
  const vents = VENTS[zone] || [...VENTS.driver, ...VENTS.passenger];
  const ventDir = zone === "rear" ? new THREE.Vector3(-1, .25, 0) : new THREE.Vector3(-1, .06, 0);
  while (airSpawn >= 1) {
    airSpawn -= 1;
    const useDef = def && (!fanOn || Math.random() < .45);
    if (useDef) spawnAir(new THREE.Vector3(1.12, .86, (Math.random() - .5) * 1.1), new THREE.Vector3(-.75, .45, 0), .42, WARM, 1.5);
    else if (fanOn) spawnAir(vents[(Math.random() * vents.length) | 0], ventDir, .45 + fan * .1, col, 1.6);
  }
  for (let i = 0; i < AIR_N; i++) {
    const a = air[i];
    if (a.age < a.life) { a.age += dt; a.p.addScaledVector(a.v, dt); a.v.multiplyScalar(1 - .6 * dt); }
    const k = a.age < a.life ? Math.sin(Math.PI * a.age / a.life) * .55 : 0;
    airPos[i * 3] = a.p.x; airPos[i * 3 + 1] = a.p.y; airPos[i * 3 + 2] = a.p.z;
    airCol[i * 3] = a.c.r * k; airCol[i * 3 + 1] = a.c.g * k; airCol[i * 3 + 2] = a.c.b * k;
  }
  airGeo.attributes.position.needsUpdate = true; airGeo.attributes.color.needsUpdate = true;
}

// nội dung màn hình
const screenState = { info: "", touched: null, touchedAt: 0, lastDraw: 0, page: "main", card: null };
function touchPanel(key) { screenState.touched = key; screenState.touchedAt = performance.now(); drawScreen(); }
function showMain(key) { screenState.page = "main"; touchPanel(key); }
// trang thông tin lớn cho các hàm tra cứu: { title, big, lines[], bar 0–1, warn, small }
function showInfo(card) { screenState.page = "info"; screenState.card = card; touchPanel("info"); }
const FONT = "'Be Vietnam Pro', system-ui, sans-serif";
function rr(c, x, y, w, h, r) { c.beginPath(); c.roundRect(x, y, w, h, r); }
function wrapText(c, text, x, y, maxW, lineH, maxLines) {
  const words = String(text).split(/\s+/); let line = "", n = 0;
  for (let i = 0; i < words.length; i++) {
    const test = line ? line + " " + words[i] : words[i];
    if (c.measureText(test).width > maxW && line) {
      if (n === maxLines - 1) { while (c.measureText(line + "…").width > maxW && line) line = line.slice(0, -1); c.fillText(line + "…", x, y + n * lineH); return; }
      c.fillText(line, x, y + n * lineH); n++; line = words[i];
    } else line = test;
  }
  if (line) c.fillText(line, x, y + n * lineH);
}
function drawScreen() {
  const c = sctx, W = SCREEN_W, H = SCREEN_H, now = performance.now();
  screenState.lastDraw = now;
  const hl = screenState.touched && now - screenState.touchedAt < 4000 ? screenState.touched : null;
  c.fillStyle = "#0a0f14"; c.fillRect(0, 0, W, H);
  // thanh trên
  c.textBaseline = "alphabetic"; c.textAlign = "left";
  c.fillStyle = "#7f8b97"; c.font = `500 24px ${FONT}`; c.fillText("Nexa Drive", 34, 52);
  c.textAlign = "center"; c.fillStyle = "#aab4be";
  const kmh = Math.round(speed * 3.6);
  c.fillText(vehicle.mode === "driving" ? `Đang chạy · ${kmh} km/h` : "Đang đỗ", W / 2, 52);
  c.textAlign = "right"; c.fillStyle = "#e6eaee"; c.font = `600 28px ${FONT}`;
  const d = new Date(); c.fillText(`${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`, W - 34, 52);
  const PX = 26, GAP = 18, TOP = 82, PH = 318, PW = (W - PX * 2 - GAP * 2) / 3;
  if (screenState.page === "info" && screenState.card) {
    const k = screenState.card, x = PX, w = W - PX * 2;
    rr(c, x, TOP, w, PH, 20); c.fillStyle = "#131a22"; c.fill();
    if (hl === "info") { c.lineWidth = 4; c.strokeStyle = "#e2a04a"; c.stroke(); }
    c.textAlign = "left"; c.fillStyle = "#7f8b97"; c.font = `500 26px ${FONT}`; c.fillText(k.title, x + 30, TOP + 52);
    c.fillStyle = k.warn ? "#e2a04a" : "#e6eaee"; c.font = `600 ${k.small ? 64 : 96}px ${FONT}`;
    wrapText(c, k.big, x + 28, TOP + (k.small ? 132 : 150), w - 60, 70, 1);
    c.font = `400 27px ${FONT}`;
    (k.lines || []).forEach((line, i) => { c.fillStyle = i === (k.lines.length - 1) && k.warn ? "#e2a04a" : "#aab4be"; c.fillText(line, x + 30, TOP + 200 + i * 38); });
    if (k.bar != null) {
      rr(c, x + 30, TOP + PH - 44, w - 60, 14, 7); c.fillStyle = "#26303b"; c.fill();
      rr(c, x + 30, TOP + PH - 44, Math.max(14, (w - 60) * k.bar), 14, 7); c.fillStyle = k.warn ? "#e2a04a" : "#dfe8f0"; c.fill();
    }
    c.textAlign = "right"; c.fillStyle = "#56616c"; c.font = `400 19px ${FONT}`; c.fillText("Số liệu mô phỏng", x + w - 28, TOP + 52);
  } else {
  // ba ô
  const panels = ["climate", "audio", "nav"];
  panels.forEach((key, i) => {
    const x = PX + i * (PW + GAP);
    rr(c, x, TOP, PW, PH, 20); c.fillStyle = "#131a22"; c.fill();
    if (hl === key) { c.lineWidth = 4; c.strokeStyle = "#e2a04a"; c.stroke(); }
  });
  const label = (t, x) => { c.textAlign = "left"; c.fillStyle = "#7f8b97"; c.font = `500 23px ${FONT}`; c.fillText(t, x + 24, TOP + 46); };
  // điều hoà
  {
    const x = PX; label("Điều hoà", x);
    const ac = vehicle.ac, on = ac && ac.state !== "off";
    c.textAlign = "left";
    if (on) {
      const t = ac.temperature ?? 22;
      c.fillStyle = "#" + acColor(t, ac.mode).getHexString(); c.font = `600 104px ${FONT}`; c.fillText(`${t}°`, x + 22, TOP + 168);
      c.fillStyle = "#aab4be"; c.font = `400 22px ${FONT}`;
      const modeVn = { heat: "Sưởi", cool: "Làm lạnh", auto: "Tự động" }[ac.mode] || "Tự động";
      c.fillText(`${modeVn} · ${ac.zone && ac.zone !== "all" ? (VN.zone[ac.zone] || ac.zone) : "Cả xe"}`, x + 26, TOP + 206);
    } else { c.fillStyle = "#56616c"; c.font = `600 64px ${FONT}`; c.fillText("Tắt", x + 24, TOP + 150); }
    const fan = on ? (vehicle.fan ?? 3) : (vehicle.fan || 0);
    c.fillStyle = "#7f8b97"; c.font = `500 21px ${FONT}`; c.fillText("Quạt", x + 26, TOP + 254);
    for (let b = 0; b < 7; b++) {
      const bh = 10 + b * 4;
      c.fillStyle = b < fan ? "#dfe8f0" : "#26303b";
      rr(c, x + 92 + b * 22, TOP + 258 - bh, 14, bh, 3); c.fill();
    }
    const def = vehicle.defrost && vehicle.defrost.state !== "off";
    c.fillStyle = def ? "#e2a04a" : "#56616c"; c.font = `500 21px ${FONT}`;
    c.fillText(def ? `Sấy ${DEF_VN[vehicle.defrost.zone] || "kính trước"}` : "Sấy kính tắt", x + 26, TOP + 296);
  }
  // âm thanh
  {
    const x = PX + PW + GAP; label("Âm thanh", x);
    const m = vehicle.media; c.textAlign = "left";
    if (m) {
      c.fillStyle = "#e6eaee"; c.font = `600 32px ${FONT}`;
      wrapText(c, m.query || "Danh sách phát", x + 24, TOP + 104, PW - 48, 40, 2);
      c.fillStyle = "#aab4be"; c.font = `400 21px ${FONT}`;
      c.fillText(m.source ? (VN.source[m.source] || m.source) : "Đang phát", x + 24, TOP + 182);
      for (let b = 0; b < 16; b++) {
        const hgt = 8 + Math.abs(Math.sin(now / 180 + b * 1.7) * Math.sin(now / 410 + b)) * 34;
        c.fillStyle = "#8fb3d6"; rr(c, x + 24 + b * 15, TOP + 236 - hgt, 9, hgt, 2); c.fill();
      }
    } else { c.fillStyle = "#56616c"; c.font = `600 32px ${FONT}`; c.fillText("Chưa phát", x + 24, TOP + 120); }
    c.fillStyle = "#7f8b97"; c.font = `500 21px ${FONT}`; c.fillText(`Âm lượng ${vehicle.volume}`, x + 24, TOP + 274);
    rr(c, x + 24, TOP + 286, PW - 48, 10, 5); c.fillStyle = "#26303b"; c.fill();
    rr(c, x + 24, TOP + 286, Math.max(10, (PW - 48) * vehicle.volume / 40), 10, 5); c.fillStyle = "#dfe8f0"; c.fill();
  }
  // dẫn đường
  {
    const x = PX + (PW + GAP) * 2; label("Dẫn đường", x);
    c.textAlign = "left";
    if (vehicle.nav) {
      c.fillStyle = "#e6eaee"; c.font = `600 32px ${FONT}`;
      wrapText(c, vehicle.nav, x + 24, TOP + 104, PW - 48, 40, 2);
      if (vehicle.navAvoid && vehicle.navAvoid !== "none") { c.fillStyle = "#aab4be"; c.font = `400 20px ${FONT}`; c.fillText(vehicle.navAvoid === "tolls" ? "Tránh trạm thu phí" : "Tránh cao tốc", x + 24, TOP + 300); }
      // phác hoạ tuyến đường
      c.strokeStyle = "#2d3946"; c.lineWidth = 2;
      for (let g = 0; g < 4; g++) { c.beginPath(); c.moveTo(x + 24, TOP + 170 + g * 30); c.lineTo(x + PW - 24, TOP + 160 + g * 34); c.stroke(); }
      c.strokeStyle = "#8fb3d6"; c.lineWidth = 6; c.lineCap = "round"; c.beginPath();
      c.moveTo(x + 40, TOP + 290); c.bezierCurveTo(x + 110, TOP + 270, x + 90, TOP + 190, x + 170, TOP + 200); c.lineTo(x + PW - 46, TOP + 168); c.stroke();
      c.fillStyle = "#e2a04a"; c.beginPath(); c.arc(x + PW - 46, TOP + 168, 9, 0, Math.PI * 2); c.fill();
      c.fillStyle = "#e6eaee"; c.beginPath(); c.arc(x + 40, TOP + 290, 7, 0, Math.PI * 2); c.fill();
    } else if (vehicle.nearby) {
      c.fillStyle = "#e6eaee"; c.font = `600 32px ${FONT}`;
      wrapText(c, `Tìm ${VN.cat[vehicle.nearby.category] || vehicle.nearby.category}`, x + 24, TOP + 104, PW - 48, 40, 2);
      c.fillStyle = "#aab4be"; c.font = `400 21px ${FONT}`;
      c.fillText(vehicle.nearby.radius_km ? `Trong bán kính ${vehicle.nearby.radius_km} km` : "Quanh vị trí hiện tại", x + 24, TOP + 182);
    } else { c.fillStyle = "#56616c"; c.font = `600 32px ${FONT}`; wrapText(c, "Chưa có điểm đến", x + 24, TOP + 120, PW - 48, 40, 2); }
  }
  }
  // dòng trợ lý
  rr(c, PX, TOP + PH + 18, W - PX * 2, H - (TOP + PH + 18) - 22, 18);
  c.fillStyle = "#131a22"; c.fill();
  if (hl === "info") { c.lineWidth = 4; c.strokeStyle = "#e2a04a"; c.stroke(); }
  c.textAlign = "left"; c.fillStyle = "#7f8b97"; c.font = `500 21px ${FONT}`; c.fillText("Trợ lý", PX + 24, TOP + PH + 56);
  c.fillStyle = screenState.info ? "#e6eaee" : "#56616c"; c.font = `500 25px ${FONT}`;
  wrapText(c, screenState.info || "Nói hoặc gõ lệnh để điều khiển xe", PX + 24, TOP + PH + 92, W - PX * 2 - 48, 32, 2);
  screenTex.needsUpdate = true;
}
document.fonts?.ready?.then(drawScreen);
setInterval(() => { if (performance.now() - screenState.lastDraw > 20000) drawScreen(); }, 30000);

/* ---------------- Bộ phận bổ sung: mỗi hàm của model đều có chỗ hiện trên xe ----------------
   Gồm: kính sau tách trái/phải, nóc kính (cửa sổ trời), lớp hơi nước + dây sấy kính sau,
   cần gạt nước, gương gập khi khoá, nắp cổng sạc (bên trái) và nắp bình xăng (bên phải).
   Vị trí đặt các chi tiết được dò bằng tia chiếu lên bề mặt thân xe sau khi tải mô hình. */
const extras = {
  ready: false,
  fog: { front: { amt: 0, clear: 1, mats: [] }, rear: { amt: 0, clear: 1, mats: [] } },
  heaterMat: new THREE.MeshBasicMaterial({ color: "#3a2c24" }),
  heat: 0,
  wiper: null, wiperAxis: new THREE.Vector3(0, .97, .23).normalize(), wiperPhase: 0, wiperAngle: 0,
  mirrors: [], fold: 0,
  ports: {},
};
const wipers = { state: "off", speed: "low" };
const HEATER_OFF = new THREE.Color("#3a2c24"), HEATER_ON = new THREE.Color("#ff7a2e");

function fogMaterial(y0, y1) {
  return new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, side: THREE.DoubleSide,
    polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
    uniforms: { uAmt: { value: 0 }, uClear: { value: 1 }, uY0: { value: y0 }, uY1: { value: y1 } },
    vertexShader: `varying vec3 vW; void main(){ vec4 w = modelMatrix * vec4(position,1.); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
    fragmentShader: `uniform float uAmt, uClear, uY0, uY1; varying vec3 vW;
      float h(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453); }
      float n(vec2 p){ vec2 i = floor(p), f = fract(p); f = f*f*(3.-2.*f);
        return mix(mix(h(i), h(i+vec2(1,0)), f.x), mix(h(i+vec2(0,1)), h(i+vec2(1,1)), f.x), f.y); }
      void main(){
        float t = clamp((vW.y - uY0) / (uY1 - uY0), 0., 1.);
        float edge = uClear + (n(vW.xz * 5.) - .5) * .12;
        float grain = .82 + .18 * n(vW.xz * 22.) * n(vW.xz * 61.);
        float a = uAmt * smoothstep(edge - .05, edge + .05, t) * grain;
        gl_FragColor = vec4(vec3(.83, .87, .9), a * .58);
      }`,
  });
}

function buildExtras(root, byName) {
  const rc = new THREE.Raycaster();
  model.updateMatrixWorld(true);
  const meshesOf = o => { const out = []; o?.traverse(c => { if (c.isMesh) out.push(c); }); return out; };

  // 1. Kính hông sau: tách lưới làm hai nửa trái/phải để hạ riêng
  const rearSides = byName("BodyWindowsRearSides");
  if (rearSides) {
    for (const mesh of meshesOf(rearSides)) {
      const g = mesh.geometry, pos = g.attributes.position, idx = g.index;
      const v = new THREE.Vector3(), halves = { left: [], right: [] };
      const triCount = idx ? idx.count / 3 : pos.count / 3;
      for (let t = 0; t < triCount; t++) {
        let cx = 0;
        for (let k = 0; k < 3; k++) { const i = idx ? idx.getX(t * 3 + k) : t * 3 + k; v.fromBufferAttribute(pos, i).applyMatrix4(mesh.matrixWorld); model.worldToLocal(v); cx += v.x; }
        for (let k = 0; k < 3; k++) (cx > 0 ? halves.left : halves.right).push(idx ? idx.getX(t * 3 + k) : t * 3 + k);
      }
      for (const side of ["left", "right"]) {
        const ng = g.clone(); ng.setIndex(halves[side]);
        const m = new THREE.Mesh(ng, mesh.material);
        const pivot = new THREE.Group(); model.add(pivot);
        new THREE.Matrix4().copy(model.matrixWorld).invert().multiply(mesh.matrixWorld).decompose(m.position, m.quaternion, m.scale);
        pivot.add(m);
        windows[side === "left" ? "rear_left" : "rear_right"].pivot = pivot;
      }
      mesh.visible = false;
    }
  }

  // 2. Nóc xe thành tấm kính: cửa sổ trời nhìn được vào khoang lái
  const roofGlass = new THREE.MeshPhysicalMaterial({ color: "#0d1317", metalness: .15, roughness: .06, clearcoat: 1, transparent: true, opacity: .6 });
  carMats.push(roofGlass);
  for (const m of meshesOf(byName("BodyRoofPanel"))) { m.material = roofGlass; m.castShadow = false; }

  // 3. Hơi nước trên kính lái và kính sau (tan dần khi sấy)
  for (const [key, node, y0, y1] of [["front", "BodyWindshield", .8, 1.14], ["rear", "BodyRearwindow", .96, 1.16]]) {
    for (const m of meshesOf(byName(node))) {
      const mat = fogMaterial(y0, y1);
      const fog = new THREE.Mesh(m.geometry, mat);
      fog.renderOrder = 3; fog.castShadow = false; fog.receiveShadow = false;
      m.add(fog);
      extras.fog[key].mats.push(mat);
    }
  }

  // 4. Dây sấy kính sau: dò bề mặt kính sau rồi đặt các sợi mảnh bám theo
  const rearGlassMeshes = meshesOf(rig.hatch).filter(m => (m.material.name || "") === "Glass");
  if (rearGlassMeshes.length) {
    rig.hatch.updateMatrixWorld(true);
    for (let k = 0; k < 8; k++) {
      const zm = -.62 - k * .115;   // toạ độ mô hình dọc theo xe
      const pts = [];
      for (let i = 0; i <= 14; i++) {
        const xm = -.58 + i * (1.16 / 14);
        const o = new THREE.Vector3(xm, 3, zm); model.localToWorld(o);
        rc.set(o, new THREE.Vector3(0, -1, 0));
        const h = rc.intersectObjects(rearGlassMeshes, false)[0];
        if (!h) continue;
        const nrm = h.face.normal.clone().transformDirection(h.object.matrixWorld);
        pts.push(rig.hatch.worldToLocal(h.point.clone().addScaledVector(nrm, .003)));
      }
      if (pts.length < 4) continue;
      const tube = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 40, .0016, 4), extras.heaterMat);
      rig.hatch.add(tube);
    }
  }

  // 5. Cần gạt nước: quay quanh chân cần, trục vuông góc với kính lái
  const wiperNode = byName("BodyWindshieldWipers");
  if (wiperNode) extras.wiper = pivotAt(new THREE.Vector3(.682, .849, 1.427), [wiperNode]);

  // 6. Gương chiếu hậu gập vào khi khoá xe
  for (const [side, sign] of [["L", 1], ["R", -1]]) {
    const nodes = ["Mirror", "MirrorColor1", "MirrorColor2"].map(n => byName(`BodyDoor${side}${n}`)).filter(Boolean);
    if (nodes.length) extras.mirrors.push({ pivot: pivotAt(new THREE.Vector3(sign * .95, .95, .45), nodes), sign });
  }

  // 7. Nắp cổng sạc (hông sau bên trái) và nắp bình xăng (hông sau bên phải)
  const body = []; root.traverse(o => { if (o.isMesh && /^Paint 1/.test(o.material.name || "")) body.push(o); });
  extras.ports.charge = makePort("charge", -1, body, rc);
  extras.ports.fuel = makePort("fuel", 1, body, rc);

  extras.ready = true;
}

// side: -1 = bên trái xe (phía -z), 1 = bên phải
function makePort(kind, side, body, rc) {
  const xw = -.8 - .24;          // giữa cửa và vòm bánh sau
  let hit = null;
  for (const y of [.77, .8, .74, .83]) {
    rc.set(new THREE.Vector3(xw, y, side * 3), new THREE.Vector3(0, 0, -side));
    hit = rc.intersectObjects(body, false)[0];
    if (hit) break;
  }
  if (!hit) return null;
  const n = hit.face.normal.clone().transformDirection(hit.object.matrixWorld).normalize();
  const yAxis = new THREE.Vector3(0, 1, 0).addScaledVector(n, -n.y).normalize();
  const xAxis = new THREE.Vector3().crossVectors(yAxis, n).normalize();
  const g = new THREE.Group();
  g.matrix.makeBasis(xAxis, yAxis, n).setPosition(hit.point.clone().addScaledVector(n, .0015));
  g.matrix.decompose(g.position, g.quaternion, g.scale);
  car.add(g);
  const W = .12, H = .1, R = .026;
  const rect = (w, h, r) => { const s = new THREE.Shape(); s.moveTo(-w / 2 + r, -h / 2); s.lineTo(w / 2 - r, -h / 2); s.quadraticCurveTo(w / 2, -h / 2, w / 2, -h / 2 + r); s.lineTo(w / 2, h / 2 - r); s.quadraticCurveTo(w / 2, h / 2, w / 2 - r, h / 2); s.lineTo(-w / 2 + r, h / 2); s.quadraticCurveTo(-w / 2, h / 2, -w / 2, h / 2 - r); s.lineTo(-w / 2, -h / 2 + r); s.quadraticCurveTo(-w / 2, -h / 2, -w / 2 + r, -h / 2); return s; };
  const dark = new THREE.MeshStandardMaterial({ color: "#0c0d0f", roughness: .8 });
  const seam = new THREE.Mesh(new THREE.ShapeGeometry(rect(W + .006, H + .006, R + .003)), dark); seam.position.z = .0003;
  const recess = new THREE.Mesh(new THREE.ShapeGeometry(rect(W - .004, H - .004, R)), new THREE.MeshStandardMaterial({ color: "#16181b", roughness: .9 })); recess.position.z = .0008;
  g.add(seam, recess);
  let led = null;
  if (kind === "charge") {
    const sock = new THREE.Mesh(new THREE.CylinderGeometry(.027, .027, .012, 28), new THREE.MeshStandardMaterial({ color: "#4a4e55", roughness: .45, metalness: .35 }));
    sock.rotation.x = Math.PI / 2; sock.position.z = -.0035; sock.scale.y = .67; g.add(sock);
    for (let i = 0; i < 5; i++) {
      const a = i / 5 * Math.PI * 2 + Math.PI / 2;
      const pin = new THREE.Mesh(new THREE.CircleGeometry(.0055, 14), dark); pin.position.set(Math.cos(a) * .014, Math.sin(a) * .014, .0003); g.add(pin);
    }
    led = new THREE.Mesh(new THREE.TorusGeometry(.033, .0022, 8, 40), new THREE.MeshBasicMaterial({ color: "#1f3b2a" }));
    led.position.z = -.0008; g.add(led);
  } else {
    const cap = new THREE.Mesh(new THREE.CylinderGeometry(.024, .024, .014, 24), new THREE.MeshStandardMaterial({ color: "#55595f", roughness: .45, metalness: .4 }));
    cap.rotation.x = Math.PI / 2; cap.position.z = -.0052; g.add(cap);
    const grip = new THREE.Mesh(new THREE.BoxGeometry(.03, .006, .002), new THREE.MeshStandardMaterial({ color: "#2a2c30" })); grip.position.z = .0028; g.add(grip);
  }
  // nắp: bản lề ở mép phía đầu xe
  const front = Math.sign(xAxis.x) || 1;
  const hinge = new THREE.Group(); hinge.position.set(front * W / 2, 0, .0012); g.add(hinge);
  const flapGeo = new THREE.ExtrudeGeometry(rect(W, H, R), { depth: .003, bevelEnabled: true, bevelThickness: .0008, bevelSize: .0008, bevelSegments: 2 });
  const flap = new THREE.Mesh(flapGeo, MAT.paint || new THREE.MeshStandardMaterial({ color: "#979da3" }));
  flap.position.x = -front * W / 2; flap.castShadow = true;
  hinge.add(flap);
  return { group: g, hinge, front, open: 0, target: 0, led, at: g.position.clone(), normal: n };
}

// Nhãn chú thích bám theo vị trí trên xe (áp suất lốp, trạng thái cửa…)
const calloutsEl = document.getElementById("callouts");
let callouts = [], calloutsUntil = 0;
function showCallouts(list, ms = 10000) {
  calloutsEl.innerHTML = "";
  callouts = list.map(c => {
    const el = document.createElement("div");
    el.className = "callout" + (c.warn ? " warn" : "");
    const label = document.createElement("span"); label.textContent = c.text; el.append(label);
    calloutsEl.append(el);
    return { el, at: new THREE.Vector3(...c.at) };
  });
  calloutsUntil = performance.now() + ms;
}
function clearCallouts() { calloutsEl.innerHTML = ""; callouts = []; statusEl.hidden = true; }
// thẻ tổng quan trạng thái xe (get_vehicle_status)
const statusEl = document.getElementById("statusCard");
let statusTimer = 0;
function showStatusCard(rows) {
  statusEl.innerHTML = "";
  const h = document.createElement("h3"); h.textContent = "Tổng quan xe"; statusEl.append(h);
  const dl = document.createElement("dl");
  for (const [k, v, warn] of rows) {
    const dt = document.createElement("dt"); dt.textContent = k;
    const dd = document.createElement("dd"); dd.textContent = v; if (warn) dd.className = "warn";
    dl.append(dt, dd);
  }
  statusEl.append(dl); statusEl.hidden = false;
  clearTimeout(statusTimer); statusTimer = setTimeout(() => { if (!window.__nexa?.hold) statusEl.hidden = true; }, 15000);
}
const _cv = new THREE.Vector3();
function updateCallouts(now) {
  if (!callouts.length) return;
  if (now > calloutsUntil && !window.__nexa?.hold) { clearCallouts(); return; }
  const w = renderer.domElement.clientWidth, h = renderer.domElement.clientHeight;
  for (const c of callouts) {
    _cv.copy(c.at); car.localToWorld(_cv);
    const behind = _cv.clone().sub(camera.position).dot(camera.getWorldDirection(new THREE.Vector3())) < .2;
    _cv.project(camera);
    const vis = !behind && Math.abs(_cv.x) < 1.05 && Math.abs(_cv.y) < 1.05;
    c.el.style.display = vis ? "" : "none";
    if (vis) c.el.style.transform = `translate(${(_cv.x * .5 + .5) * w}px, ${(-_cv.y * .5 + .5) * h}px)`;
  }
}

// Điểm neo chú thích (toạ độ theo xe: đầu xe +x, bên trái -z)
const ANCHOR = {
  driver: [-.6, 1.08, -.92], passenger: [-.6, 1.08, .92],
  rear_left: [-1.3, .98, -1.0], rear_right: [-1.3, .98, 1.0],
  trunk: [-1.45, 1.2, 0], sunroof: [-.3, 1.2, 0], headlights: [1.95, .72, -.72], ac: [.55, 1.08, 0],
  win_front_left: [-.15, 1.06, -.9], win_front_right: [-.15, 1.06, .9],
  win_rear_left: [-1.0, 1.08, -.92], win_rear_right: [-1.0, 1.08, .92],
  fuel: [-1.04, .84, 1.12], charge: [-1.04, .84, -1.12],
  tire_front_left: [1.25, .82, -1.0], tire_front_right: [1.25, .82, 1.0],
  tire_rear_left: [-1.55, .82, -1.0], tire_rear_right: [-1.55, .82, 1.0],
};

function updateExtras(dt, now) {
  if (!extras.ready) return;
  // hơi nước: tan từ dưới lên khi đang sấy, sau đó biến mất
  for (const key of ["front", "rear"]) {
    const f = extras.fog[key];
    if (f.amt > 0) {
      f.clear = Math.min(1.25, f.clear + dt * .42);
      if (f.clear >= 1.2) f.amt = Math.max(0, f.amt - dt * .8);
    }
    for (const m of f.mats) { m.uniforms.uAmt.value = f.amt; m.uniforms.uClear.value = f.clear; }
  }
  const d = vehicle.defrost, rearOn = d && d.state !== "off" && (d.zone === "rear" || d.zone === "both");
  extras.heat = approach(extras.heat, rearOn ? 1 : 0, 1.4, dt);
  extras.heaterMat.color.copy(HEATER_OFF).lerp(HEATER_ON, extras.heat * (.85 + .15 * Math.sin(now / 300)));
  // cần gạt
  if (extras.wiper) {
    const running = wipers.state === "on" || wipers.state === "auto";
    const rate = wipers.speed === "high" ? 7.5 : 4.2;
    if (running || extras.wiperPhase % (Math.PI * 2) > .05) extras.wiperPhase += dt * rate;
    if (!running && extras.wiperPhase % (Math.PI * 2) < dt * rate + .02) extras.wiperPhase = 0;
    const a = -.85 * (.5 - .5 * Math.cos(extras.wiperPhase));
    extras.wiper.quaternion.setFromAxisAngle(extras.wiperAxis, a);
  }
  // gương gập khi khoá
  extras.fold = approach(extras.fold, vehicle.locked ? 1 : 0, 2.5, dt);
  for (const m of extras.mirrors) m.pivot.rotation.y = m.sign * extras.fold * 1.05;
  // nắp sạc / nắp xăng
  for (const p of Object.values(extras.ports)) {
    if (!p) continue;
    p.open = approach(p.open, p.target, 3, dt);
    p.hinge.rotation.y = p.front * ease(Math.min(1, p.open)) * 1.75;
    if (p.led) p.led.material.color.set(p.open > .6 ? new THREE.Color("#2bd67b").multiplyScalar(.6 + .4 * Math.sin(now / 260) ** 2) : "#1f3b2a");
  }
}

// Số liệu cảm biến — GHI ĐÈ bằng trạng thái xe của backend (dongBo); ngưỡng theo serve/xe_mo_phong.py
// (đổ xăng < 15%, lốp non < 30 PSI; quãng đường = 4,2 km / 1% xăng, 3,5 km / 1% pin)
const sensors = {
  fuel_pct: 62, fuel_range_km: 260, fuel_low_pct: 15,
  battery_pct: 80, ev_range_km: 280,
  odometer_km: 23450,
  tire_psi: { front_left: 32.5, front_right: 32, rear_left: 29.5, rear_right: 31.8 }, tire_ok: [30, 38],
};
const WHEEL_VN = { front_left: "Lốp trước trái", front_right: "Lốp trước phải", rear_left: "Lốp sau trái", rear_right: "Lốp sau phải" };
const fmtNum = n => Math.round(n).toLocaleString("vi-VN");

// kq: kết quả hàm do backend trả (AssistantTurn.results) — dùng cho thời tiết / quãng đường thay vì số tự bịa
function runQuery(name, args, kq = {}) {
  switch (name) {
    case "get_fuel_level":
      showInfo({ title: "Nhiên liệu", big: `${sensors.fuel_pct}%`, bar: sensors.fuel_pct / 100,
        lines: [`Đi được khoảng ${sensors.fuel_range_km} km`, sensors.fuel_pct <= sensors.fuel_low_pct ? "Đã dưới ngưỡng, nên đổ xăng sớm" : "Chưa cần đổ xăng"], warn: sensors.fuel_pct <= sensors.fuel_low_pct });
      focus("cabin"); break;
    case "get_battery_range":
      showInfo({ title: "Pin cao áp", big: `${sensors.battery_pct}%`, bar: sensors.battery_pct / 100, lines: [`Chạy điện được khoảng ${sensors.ev_range_km} km`] });
      focus("cabin"); break;
    case "get_odometer": {
      const mi = args.unit === "mi";
      showInfo({ title: "Tổng quãng đường (ODO)", big: `${fmtNum(mi ? sensors.odometer_km * .621371 : sensors.odometer_km)} ${mi ? "dặm" : "km"}`, lines: [mi ? `Tương đương ${fmtNum(sensors.odometer_km)} km` : `Tương đương ${fmtNum(sensors.odometer_km * .621371)} dặm`] });
      focus("cabin"); break;
    }
    case "get_range_estimate": {
      const total = sensors.fuel_range_km + sensors.ev_range_km, trip = args.trip_km != null ? +args.trip_km : null;
      const lines = [`Xăng ${sensors.fuel_range_km} km · Điện ${sensors.ev_range_km} km`];
      let warn = false;
      if (trip != null) { warn = trip > total; lines.push(warn ? `Chuyến ${fmtNum(trip)} km: không đủ, thiếu ${fmtNum(trip - total)} km` : `Chuyến ${fmtNum(trip)} km: đủ, còn dư ${fmtNum(total - trip)} km`); }
      showInfo({ title: "Còn đi được", big: `${fmtNum(total)} km`, lines, warn, bar: trip != null ? Math.min(1, trip / total) : null });
      focus("cabin"); break;
    }
    case "get_weather": {
      const day = +(args.day ?? 0);
      const lines = [day === 0 ? "Hôm nay" : day === 1 ? "Ngày mai" : day === 2 ? "Ngày kia" : `${day} ngày nữa`];
      if (kq.temp_c != null) lines.push(`${kq.description ?? ""} · ${Math.round(kq.temp_c)}°C`.replace(/^ · /, ""));
      showInfo({ title: "Thời tiết", big: kq.location || args.location || "Vị trí hiện tại", lines, small: true });
      focus("cabin"); break;
    }
    case "get_tire_pressure": {
      const ids = args.wheel ? [args.wheel] : Object.keys(sensors.tire_psi);
      showCallouts(ids.filter(id => sensors.tire_psi[id] != null).map(id => {
        const psi = sensors.tire_psi[id], low = psi < sensors.tire_ok[0], high = psi > sensors.tire_ok[1];
        return { at: ANCHOR["tire_" + id], text: `${WHEEL_VN[id]}: ${psi} PSI${low ? " · non hơi" : high ? " · quá căng" : ""}`, warn: low || high };
      }));
      focus(args.wheel ? "tire_" + args.wheel : "tires"); break;
    }
    case "get_vehicle_status": {
      const winTxt = id => { const v = Math.round(windows[id].target * 100); return v ? `hạ ${v}%` : "đóng"; };
      const openWins = WIN_IDS.filter(id => windows[id].target > 0);
      const sr = vehicle.sunroof;
      const lights = { off: "tắt", auto: "tự động", low_beam: "chiếu gần", high_beam: "chiếu xa", parking: "đèn đỗ" }[vehicle.headlights] || vehicle.headlights;
      const lowTires = Object.entries(sensors.tire_psi).filter(([, p]) => p < sensors.tire_ok[0]);
      const unlocked = Object.entries(vehicle.doorLocks).filter(([, v]) => !v).map(([k]) => DOOR_VN[k].replace("Cửa ", ""));
      showStatusCard([
        ["Cửa", unlocked.length ? `mở khoá: ${unlocked.join(", ")}` : "đã khoá hết", unlocked.length > 0],
        ["Cốp", trunkState.target ? "đang mở" : "đóng", !!trunkState.target],
        ["Kính", openWins.length ? openWins.map(id => `${{ front_left: "lái", front_right: "phụ", rear_left: "sau trái", rear_right: "sau phải" }[id]} ${winTxt(id)}`).join(", ") : "đóng hết", openWins.length > 0],
        ["Cửa sổ trời", sr.action === "open" ? `mở ${sr.intensity ?? 100}%` : sr.action === "tilt" ? "hé" : "đóng"],
        ["Đèn", lights],
        ["Điều hoà", vehicle.ac && vehicle.ac.state !== "off" ? `${vehicle.ac.temperature}°C` : "tắt"],
        ["Xăng", `${sensors.fuel_pct}% · ${sensors.fuel_range_km} km`, sensors.fuel_pct <= sensors.fuel_low_pct],
        ["Pin", `${sensors.battery_pct}% · ${sensors.ev_range_km} km`],
        ["Lốp", lowTires.length ? lowTires.map(([id, p]) => `${WHEEL_VN[id].replace("Lốp ", "")} ${p} PSI`).join(", ") : "bình thường", lowTires.length > 0],
      ]);
      // chỉ gắn nhãn lên xe cho những chỗ bất thường
      showCallouts([
        ...lowTires.map(([id, p]) => ({ at: ANCHOR["tire_" + id], text: `${WHEEL_VN[id]}: ${p} PSI`, warn: true })),
        ...(trunkState.target ? [{ at: ANCHOR.trunk, text: "Cốp đang mở", warn: true }] : []),
        ...openWins.map(id => ({ at: ANCHOR["win_" + id], text: `Kính ${winTxt(id)}`, warn: true })),
      ], 12000);
      focus("status"); break;
    }
  }
}

/* ---------------- Vòng lặp hoạt ảnh ---------------- */
const timer = new Timer();
let speed = 0;
const ease = t => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
function approach(cur, target, rate, dt) { return cur + (target - cur) * Math.min(1, rate * dt); }
function frame() {
  timer.update(); const dt = Math.min(timer.getDelta(), .1), now = performance.now();
  const targetSpeed = vehicle.mode === "driving" ? 13 : 0;
  speed = approach(speed, targetSpeed, 1.2, dt);
  if (speed > .01) {
    const dx = speed * dt;
    for (const s of scrollers) { s.x -= dx; if (s.x < SCROLL_MIN) s.x += SCROLL_LEN; s.obj.position.x = s.x; }
    for (let i = 0; i < DASH_N; i++) { dashX[i] -= dx; if (dashX[i] < -300) dashX[i] += DASH_N * DASH_SPAN; }
    layoutDashes();
    asphaltTex.offset.x += dx / 10; tileTex.offset.x += dx / 2;
    for (const w of rig.wheels) w.rotation.x += dx / WHEEL_R;
    car.position.y = Math.sin(now / 90) * .003;
  }
  // cốp, kính, cửa sổ trời
  trunkState.open = approach(trunkState.open, trunkState.target, 2.4, dt);
  const sr = vehicle.sunroof;
  sunroofState.slide = approach(sunroofState.slide, sr.action === "open" ? (sr.intensity ?? 100) / 100 : 0, 2, dt);
  sunroofState.tilt = approach(sunroofState.tilt, sr.action === "tilt" ? 1 : 0, 2, dt);
  for (const id of WIN_IDS) { const w = windows[id]; w.open = approach(w.open, w.target, 2.2, dt); }
  if (rig.ready) {
    rig.hatch.rotation.x = ease(trunkState.open) * .85;
    rig.roof.position.set(0, rig.roofBase.y + sunroofState.slide * .045 + sunroofState.tilt * .01, rig.roofBase.z - sunroofState.slide * .6);
    rig.roof.rotation.x = -sunroofState.tilt * .09;
    windows.front_left.pivot.position.y = -windows.front_left.open * .23;
    windows.front_right.pivot.position.y = -windows.front_right.open * .23;
    // kính hông sau (xe mẫu 2 cửa): hạ thấp hơn vì ô kính nhỏ
    if (windows.rear_left.pivot) windows.rear_left.pivot.position.y = -windows.rear_left.open * .15;
    if (windows.rear_right.pivot) windows.rear_right.pivot.position.y = -windows.rear_right.open * .15;
  }
  // xi nhan: khẩn cấp hoặc nháy khi khoá/mở khoá
  let blink = 0;
  if (vehicle.hazard) blink = (now % 800) < 420 ? 1 : 0;
  else if (now < lockBlinkUntil) blink = ((lockBlinkUntil - now) % 550) > 270 ? 1 : 0;
  if (MAT.sig) MAT.sig.emissiveIntensity = blink * 6;
  for (const b of blinkers) b.material.opacity = blink;
  // camera
  if (camTween) {
    const k = Math.min(1, (now - camTween.t0) / camTween.dur), e = ease(k);
    camera.position.lerpVectors(camTween.from.p, camTween.to.p, e);
    controls.target.lerpVectors(camTween.from.t, camTween.to.t, e);
    camera.fov = camTween.from.fov + (camTween.to.fov - camTween.from.fov) * e; camera.updateProjectionMatrix();
    if (k >= 1) finishTween();
  }
  // bộ phận bổ sung, chú thích, đồng hồ quãng đường
  updateExtras(dt, now);
  sensors.odometer_km += speed * dt / 1000;
  // khoang lái
  updateAir(dt);
  domeLight.intensity = approach(domeLight.intensity, wantInside ? (isNight ? .6 : .25) : 0, 3, dt);
  if (wantInside && now - screenState.lastDraw > (vehicle.media ? 110 : 1000)) drawScreen();
  controls.update();
  renderer.render(scene, camera);
  updateCallouts(now);
  requestAnimationFrame(frame);
}
controls.addEventListener("start", finishTween);

function resize() {
  const r = stage.getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / r.height;
  if (!camTween) camera.fov = fovFor(wantInside);
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(stage);
window.__nexa = { THREE, camera, controls, scene, model, rig, focus, screenGroup, extras };
viewBtn.addEventListener("click", () => focus(wantInside ? "home" : "cabin"));   // tiện kiểm tra trong DevTools
resize();
focus("home", true);
applyTime(true);
renderChips();
frame();
loadCar();

/* ---------------- Giao diện: nối backend qua WebSocket (web/src/ws.ts) ---------------- */
const saidEl = document.getElementById("said"), replyEl = document.getElementById("reply"), callEl = document.getElementById("call");
const input = document.getElementById("text");
const answersEl = document.getElementById("answers"), ans1 = document.getElementById("ans1"), ans2 = document.getElementById("ans2");
const childBtn = document.getElementById("childLock");

function setText(el, text, empty) { el.textContent = text; el.classList.toggle("empty", !!empty); }
const fmtCall = c => `${c.name}(${Object.entries(c.arguments || {}).map(([k, v]) => `${k}=${typeof v === "string" ? JSON.stringify(v) : v}`).join(", ")})`;

// Góc camera khi trợ lý HỎI LẠI về một hàm (chưa chạy gì): nhìn trước tới bộ phận sắp đụng tới
const FOCUS_HAM = {
  set_ac: "cabin", set_fan_speed: "cabin", set_defrost: "windshield", lock_doors: "doorLeft", unlock_doors: "doorLeft",
  open_trunk: "rear", close_trunk: "rear", open_charge_port: "charge", open_fuel_flap: "fuel", control_window: "left",
  operate_sunroof: "top", set_headlights: "front", set_hazard_lights: "hazard", set_volume: "cabin", play_media: "cabin",
  navigate_to: "cabin", find_nearby: "cabin",
};

/* Trạng thái xe của backend -> mọi thứ vẽ trên xe (backend là nguồn chân lý: luật an toàn có thể đã chặn
   hoặc sửa lời gọi). Gọi sau mỗi StateEvent, và một lần nữa khi mô hình xe tải xong. */
function dongBo(s) {
  xeBackend = s;
  vehicle.mode = s.dang_chay ? "driving" : "parked";
  vehicle.child_lock = !!s.khoa_tre_em;
  document.querySelectorAll("[data-mode]").forEach(x => x.setAttribute("aria-pressed", String((x.dataset.mode === "driving") === !!s.dang_chay)));
  childBtn.setAttribute("aria-pressed", String(vehicle.child_lock));
  // backend chỉ có cờ khoá chung: khoá -> khoá hết; mở khoá -> giữ cửa nào vừa mở (applyCall), không thì mở hết
  if (s.doors_locked) for (const k in vehicle.doorLocks) vehicle.doorLocks[k] = true;
  else if (Object.values(vehicle.doorLocks).every(Boolean)) for (const k in vehicle.doorLocks) vehicle.doorLocks[k] = false;
  vehicle.locked = !!s.doors_locked;
  trunkState.target = s.trunk_open ? 1 : 0;
  if (extras.ports.charge) extras.ports.charge.target = s.charge_port_open ? 1 : 0;
  if (extras.ports.fuel) extras.ports.fuel.target = s.fuel_flap_open ? 1 : 0;
  for (const id of WIN_IDS) windows[id].target = (s.windows?.[id] ?? 0) / 100;
  vehicle.sunroof = { action: s.sunroof_state, intensity: s.sunroof_percent };
  vehicle.ac = { state: s.ac_on ? "on" : "off", temperature: s.ac_temperature, zone: s.ac_zone, mode: s.ac_mode };
  vehicle.fan = s.fan_level;
  vehicle.defrost = { state: s.defrost_on ? "on" : "off", zone: s.defrost_zone };
  vehicle.headlights = s.headlights; updateLights();
  vehicle.hazard = !!s.hazards_on;
  vehicle.volume = s.volume;
  vehicle.media = s.media_playing ? { source: s.media_source, query: s.now_playing } : null;
  if (s.destination) vehicle.nav = s.destination;
  sensors.fuel_pct = s.fuel_percent; sensors.fuel_range_km = Math.round(s.fuel_percent * 4.2);
  sensors.battery_pct = s.battery_percent; sensors.ev_range_km = Math.round(s.battery_percent * 3.5);
  sensors.odometer_km = s.odometer_km; sensors.tire_psi = { ...s.tire_psi };
  renderChips(); drawScreen();
}

// Nút trả lời nhanh khi trợ lý hỏi lại — gửi đúng câu như người lái tự nói ("có" / "không" / "cái đầu"…)
let kieuHoiLai = null;
function hienHoiLai(kieu) {
  kieuHoiLai = kieu;
  answersEl.hidden = !kieu;
  if (kieu === "chon") { ans1.textContent = "Cái đầu"; ans2.textContent = "Cái thứ hai"; }
  else { ans1.textContent = "Có"; ans2.textContent = "Không"; }
}
ans1.addEventListener("click", () => gui(kieuHoiLai === "chon" ? "cái đầu" : "có"));
ans2.addEventListener("click", () => gui(kieuHoiLai === "chon" ? "cái thứ hai" : "không"));

function onEvent(e) {
  if (e.type === "state") return dongBo(e.vehicle);
  if (e.type === "error") { setText(replyEl, `Backend báo lỗi: ${e.message}`); return; }
  hienHoiLai(null);
  if (e.type === "assistant_turn") {
    // chỉ những lời gọi ĐÃ chạy (đã qua luật an toàn) mới có ở đây -> hoạt cảnh; trạng thái chuẩn đến ngay sau (state)
    e.calls.forEach((c, i) => applyCall({ name: c.name, args: c.arguments }, e.results[i] || {}));
    setText(replyEl, e.text);
    // dòng "model đã hiểu gì" cho hội đồng xem: lời gọi + functional token + thời gian
    callEl.textContent = (e.calls.length ? e.calls.map(c => `${fmtCall(c)}  ${c.nexa}`).join("\n") : e.loai)
      + (e.latency_ms != null ? ` · ${Math.round(e.latency_ms)} ms` : "");
  } else {   // clarify: trợ lý hỏi lại, chưa làm gì
    setText(replyEl, e.text);
    callEl.textContent = `hỏi lại (${e.ly_do}) · ${e.functions.join(", ")}`;
    if (FOCUS_HAM[e.function]) focus(FOCUS_HAM[e.function]);
    hienHoiLai(e.ly_do === "chon" ? "chon" : e.ly_do === "thieu_tham_so" ? null : "co_khong");
  }
  callEl.hidden = false;
  screenState.info = e.text; drawScreen();
  // lệnh đến bằng giọng qua «hey nexa» -> đọc xong thì NGHE TIẾP câu kế tiếp (không cần gọi lại)
  docTo(e.text, cheDoNoi && !!wake);
}

/* ---- Giọng nói ra: VieNeu đọc câu trả lời TỪNG ĐOẠN (/tts_stream): đoạn nào tạo xong thì phát luôn, đoạn sau xếp
   lịch nối tiếp -> nghe tiếng sau ~0,2–0,3 s thay vì chờ cả câu. Tắt dịch vụ giọng nói thì im lặng, không báo lỗi. ---- */
let ctxLoa = null;   // một AudioContext dùng chung cho loa
let dangDoc = null;  // { huy: AbortController, nguon: AudioBufferSourceNode[] }
// Người dùng thử 2026-10-13: tiếng "nói một chữ, ngừng, nói tiếp". Âm thanh server gửi liền mạch (đã soi), nên sửa
// khâu phát: (1) loa mở đúng 48 kHz của VieNeu — nếu loa chạy tần số khác thì mỗi đoạn bị đổi tần số RIÊNG, lệch ở
// từng chỗ nối; (2) đệm trước 0,25 s (trước 0,05 s): các đoạn đầu chỉ dài ~85 ms, trang bận vẽ 3D là đoạn sau tới trễ.
const TAN_SO_LOA = 48000;
const DEM_TRUOC = 0.25;
function moLoa() {
  if (ctxLoa) return ctxLoa;
  try { ctxLoa = new AudioContext({ sampleRate: TAN_SO_LOA }); } catch { ctxLoa = new AudioContext(); }
  return ctxLoa;
}
function dungDoc() {   // câu mới / bấm mic / «hey nexa» thức: dừng ngay câu đang đọc
  if (!dangDoc) return;
  const d = dangDoc; dangDoc = null;
  d.huy.abort();
  d.nguon.forEach(s => { try { s.stop(); } catch { /* đoạn chưa tới lịch phát */ } });
  guiWake({ type: "tiep_tuc" });
}
async function docTo(text, ngheTiep = false) {
  if (!text) return;
  dungDoc();
  const d = { huy: new AbortController(), nguon: [] };
  dangDoc = d;
  // đọc xong: nghe «hey nexa» lại như thường, hoặc (hội thoại bằng giọng) nghe luôn câu lệnh tiếp theo
  const xong = () => {
    if (dangDoc !== d) return;
    dangDoc = null;
    if (ngheTiep && wake) batNgheTiep(); else guiWake({ type: "tiep_tuc" });
  };
  try {
    const r = await fetch(`${CONFIG.GIONG_NOI_URL}/tts_stream`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }),
      signal: d.huy.signal,
    });
    if (!r.ok) throw new Error(String(r.status));
    const tanSo = Number(r.headers.get("X-Tan-So")) || 48000;
    moLoa();
    // trình duyệt chỉ cho phát tiếng sau lần click đầu tiên trên trang: chưa click thì resume() treo mãi ->
    // chờ tối đa 0,5 s rồi bỏ qua tiếng (vẫn hiện chữ, vẫn nghe tiếp)
    if (ctxLoa.state === "suspended") await Promise.race([ctxLoa.resume(), new Promise(ok => setTimeout(ok, 500))]);
    if (ctxLoa.state !== "running") throw new Error("loa chưa được phép");
    // loa đang đọc thì tạm dừng nghe «hey nexa» — câu trả lời có chữ "NexaAI", không để trợ lý tự đánh thức mình
    guiWake({ type: "tam_dung" });
    const doc = r.body.getReader();
    let du = new Uint8Array(0);   // byte lẻ cuối gói mạng (mẫu 16-bit bị cắt đôi) giữ lại ghép với gói sau
    let lich = 0;                 // thời điểm đoạn kế tiếp bắt đầu phát
    let hut = 0;                  // số lần đoạn sau tới trễ hơn lúc đoạn trước phát xong (tiếng bị ngắt)
    for (;;) {
      const { value, done } = await doc.read();
      if (done || dangDoc !== d) break;
      const gop = new Uint8Array(du.length + value.length);
      gop.set(du); gop.set(value, du.length);
      const n = gop.length >> 1;
      du = gop.slice(n * 2);
      if (!n) continue;
      const mau = new Int16Array(gop.buffer, 0, n);
      const buf = ctxLoa.createBuffer(1, n, tanSo);
      const kenh = buf.getChannelData(0);
      for (let i = 0; i < n; i++) kenh[i] = mau[i] / 32768;
      const nguon = ctxLoa.createBufferSource();
      nguon.buffer = buf; nguon.connect(ctxLoa.destination);
      // tới kịp thì nối đuôi; đoạn đầu / tới trễ thì đệm lại DEM_TRUOC rồi phát (một lần ngắt dài hơn nhưng không ngắt lắt nhắt)
      if (lich < ctxLoa.currentTime + 0.01) { if (lich) hut++; lich = ctxLoa.currentTime + DEM_TRUOC; }
      nguon.start(lich); lich += buf.duration;
      d.nguon.push(nguon);
    }
    // chẩn đoán: xem ở Console (F12) hoặc gõ __nexaTieng — số đoạn, số lần hụt, tần số loa thật
    window.__nexaTieng = { doan: d.nguon.length, hut, tanSoLoa: ctxLoa.sampleRate, tanSoTts: tanSo };
    if (hut) console.warn("[nexa] tiếng bị hụt", window.__nexaTieng);
    if (dangDoc !== d) return;
    // Phát hết đoạn cuối thì nghe lại. HẸN GIỜ theo lịch phát chứ không gắn `onended` vào đoạn cuối: máy bận thì
    // đoạn cuối có thể phát xong TRƯỚC khi luồng báo hết (đo 2026-10-13: chỉ dư 0,2–0,8 s) -> sự kiện đã bắn mất,
    // dịch vụ nghe bị tạm dừng mãi (gọi «hey nexa» cũng không dậy).
    setTimeout(xong, Math.max(0, (lich - ctxLoa.currentTime) * 1000) + 60);
  } catch { xong(); /* chưa bật dịch vụ giọng nói / đã huỷ: chỉ hiện chữ */ }
}

/* ---- Wake-word «hey nexa» (PLAN.md Phase 6b): bật nút thì gửi âm thanh mic LIÊN TỤC (16 kHz int16, khung 80 ms) tới
   serve/giong_noi.py /wake. Server báo "thuc" khi nghe thấy, tự gom câu lệnh tới khi im, nhận dạng rồi gửi "lenh". ---- */
const wakeBtn = document.getElementById("heyNexa");
const KHUNG_WAKE = 1280;
const WORKLET_WAKE = `class ThuAm extends AudioWorkletProcessor {
  process(vao) { const c = vao[0] && vao[0][0]; if (c) this.port.postMessage(c.slice(0)); return true; }
} registerProcessor("thu-am", ThuAm);`;
let wake = null;   // { ws, ctx, luong, node }
let cheDoNoi = false;   // lượt hiện tại đến từ giọng nói qua «hey nexa» (gui(..., "wake_word")) -> đọc xong thì nghe tiếp
const CHAO = "Dạ, mình nghe đây.";
function guiWake(tin) { if (wake?.ws.readyState === WebSocket.OPEN) wake.ws.send(JSON.stringify(tin)); }
function batNgheTiep() {   // nghe câu lệnh tiếp theo ~10 s, KHÔNG cần gọi lại «hey nexa»; im lặng thì server gửi het_tiep
  guiWake({ type: "nghe_tiep" });
  wakeBtn.classList.add("nghe");
  setText(saidEl, "Đang nghe lệnh tiếp theo (khoảng 10 giây)… cứ nói, không cần gọi lại", true);
}
function veChoGoi() {
  cheDoNoi = false;
  wakeBtn.classList.remove("nghe");
  setText(saidEl, "Đang chờ «hey nexa»… (bấm micro vẫn dùng được)", true);
}
// Trình duyệt chỉ cho chạy âm thanh sau một lần tương tác với trang -> trang tự bật nghe lúc mở; nếu âm thanh còn
// bị khoá thì lần click / phím ĐẦU TIÊN ở bất kỳ đâu sẽ mở khoá (cả mic «hey nexa» lẫn loa).
let boQuaClickWake = false;
function moKhoaAmThanh() {
  const mo = ev => {
    document.removeEventListener("pointerdown", mo, true); document.removeEventListener("keydown", mo, true);
    if (ev.target?.closest?.("#heyNexa")) boQuaClickWake = true;   // click mở khoá không được tắt luôn «hey nexa»
    wake?.ctx.resume();
    moLoa().resume();
    if (wake) veChoGoi();
  };
  document.addEventListener("pointerdown", mo, true); document.addEventListener("keydown", mo, true);
}

async function batWake() {
  let luong;
  try {
    // âm thanh THÔ như lúc ghi dữ liệu train / đo wake-word; loa đọc thì đã tạm dừng nghe nên không cần khử vọng
    luong = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
  } catch { setText(saidEl, "Chưa cho phép dùng micro. Bật quyền micro cho trang này rồi thử lại.", true); return; }
  const ctx = new AudioContext({ sampleRate: 16000 });   // trình duyệt tự đổi tần số lấy mẫu của mic về 16 kHz
  await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET_WAKE], { type: "text/javascript" })));
  const node = new AudioWorkletNode(ctx, "thu-am");
  ctx.createMediaStreamSource(luong).connect(node);      // không nối ra loa
  const ws = new WebSocket(CONFIG.GIONG_NOI_URL.replace(/^http/, "ws") + "/wake");
  ws.binaryType = "arraybuffer";
  const dem = new Int16Array(KHUNG_WAKE); let n = 0;
  node.port.onmessage = ev => {
    if (ws.readyState !== WebSocket.OPEN) return;
    for (const x of ev.data) {
      dem[n++] = Math.max(-32768, Math.min(32767, Math.round(x * 32767)));
      if (n === KHUNG_WAKE) { ws.send(dem.buffer.slice(0)); n = 0; }
    }
  };
  ws.onmessage = ev => {
    const m = JSON.parse(ev.data);
    if (m.type === "thuc") {
      dungDoc(); wakeBtn.classList.add("nghe");
      setText(saidEl, "Mình nghe đây… nói lệnh rồi ngừng một chút", true);
    } else if (m.type === "chi_goi") {   // chỉ gọi «hey nexa», chưa nói lệnh -> chào bằng giọng rồi nghe lệnh
      cheDoNoi = true;
      guiWake({ type: "tam_dung" });   // đang chào thì không nghe (khỏi thu lại chính lời chào)
      setText(saidEl, "Hey Nexa"); setText(replyEl, CHAO); callEl.hidden = true;
      docTo(CHAO, true);
    } else if (m.type === "het_tiep" || m.type === "huy") {
      veChoGoi();
      // nói rõ là ĐÃ hết giờ nghe tiếp — người dùng thử 2026-10-13 tưởng trợ lý vẫn nghe nên nói lệnh trần
      if (m.type === "het_tiep") setText(saidEl, "Hết giờ nghe tiếp — gọi «hey nexa» rồi nói lệnh (hoặc bấm micro)", true);
    } else if (m.type === "lenh") {
      wakeBtn.classList.remove("nghe");
      if (m.text) gui(m.text, "wake_word"); else veChoGoi();
    }
  };
  ws.onerror = ws.onclose = () => {
    if (wake?.ws !== ws) return;
    tatWake(); setText(saidEl, "Chưa nối được dịch vụ nghe «hey nexa» (cổng 8001, cần model models/wake/hey_nexa.joblib).", true);
  };
  wake = { ws, ctx, luong, node };
  wakeBtn.setAttribute("aria-pressed", "true");
  if (ctx.state === "suspended") {   // mở trang mà chưa click: trình duyệt khoá âm thanh tới lần tương tác đầu
    setText(saidEl, "Bấm vào bất kỳ đâu trên trang để bắt đầu nghe «hey nexa»", true);
    moKhoaAmThanh();
  } else veChoGoi();
}
function tatWake() {
  if (!wake) return;
  const w = wake; wake = null;
  w.node.port.onmessage = null; w.ws.onclose = w.ws.onerror = null; w.ws.close();
  w.luong.getTracks().forEach(t => t.stop()); w.ctx.close();
  wakeBtn.setAttribute("aria-pressed", "false"); wakeBtn.classList.remove("nghe");
}
wakeBtn.addEventListener("click", () => {
  if (boQuaClickWake) { boQuaClickWake = false; return; }   // click này chỉ để mở khoá âm thanh
  if (wake) tatWake(); else batWake();
});
// mở trang là tự nghe «hey nexa» (người dùng 2026-10-12); nút vẫn tắt / bật lại được
batWake();

/* ---- Giọng nói vào: bấm micro để nói, bấm lần nữa để gửi (push-to-talk) ---- */
const micBtn = document.getElementById("mic");
const GHI_TOI_DA_MS = 10000;   // lệnh trên xe chỉ vài giây; quên bấm dừng thì tự dừng
let ghiAm = null;
function setMic(on) {
  micBtn.setAttribute("aria-pressed", String(on));
  micBtn.setAttribute("aria-label", on ? "Dừng nghe và gửi" : "Nói lệnh");
}
micBtn.disabled = false;
micBtn.title = "Bấm để nói, bấm lần nữa để gửi";
micBtn.setAttribute("aria-label", "Nói lệnh");
micBtn.addEventListener("click", async () => {
  if (ghiAm) { ghiAm.stop(); return; }
  let luong;
  // âm thanh THÔ — tắt khử ồn / khử vọng / tự chỉnh âm lượng của trình duyệt: đúng loại âm thanh đã đo STT ở bộ B / C / D
  // (ghi thô). Soi log 2026-10-08: khử ồn làm mất năng lượng phụ âm (>4 kHz) -> "nóng quá" nghe thành "lóng quá".
  try {
    luong = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
  }
  catch { setText(saidEl, "Chưa cho phép dùng micro. Bật quyền micro cho trang này rồi thử lại.", true); return; }
  dungDoc();
  guiWake({ type: "tam_dung" });   // đang bấm mic thì «hey nexa» không nghe chồng
  const manh = [];
  const rec = new MediaRecorder(luong);
  ghiAm = rec;
  const hetGio = setTimeout(() => rec.state === "recording" && rec.stop(), GHI_TOI_DA_MS);
  rec.ondataavailable = ev => manh.push(ev.data);
  rec.onstop = async () => {
    clearTimeout(hetGio); luong.getTracks().forEach(t => t.stop()); ghiAm = null; setMic(false);
    guiWake({ type: "tiep_tuc" });
    setText(saidEl, "Đang nhận dạng…", true);
    try {
      const am = new Blob(manh, { type: rec.mimeType || "audio/webm" });
      const r = await fetch(`${CONFIG.GIONG_NOI_URL}/stt`, { method: "POST", headers: { "Content-Type": am.type }, body: am });
      if (!r.ok) throw new Error(String(r.status));
      const { text } = await r.json();
      if (text) gui(text); else setText(saidEl, "Không nghe rõ, bạn nói lại nhé.", true);
    } catch {
      setText(saidEl, "Chưa nối được dịch vụ giọng nói (cổng 8001). Bạn gõ lệnh ở dưới nhé.", true);
    }
  };
  rec.start(); setMic(true); setText(saidEl, "Đang nghe… bấm micro lần nữa để gửi", true);
});

const sock = new NexaSocket(onEvent, st => { ketNoi = st; renderChips(); });

// nguon: "nut" = gõ / bấm mic (tài xế) · "wake_word" = gọi «hey nexa» — backend dùng để áp luật an toàn chặt hơn
function gui(text, nguon = "nut") {
  const c = text.trim(); if (!c) return;
  cheDoNoi = nguon === "wake_word";   // gõ chữ / bấm mic / bấm nút trả lời -> không tự nghe tiếp
  setText(saidEl, c); hienHoiLai(null); clearCallouts(); callEl.hidden = true;
  if (!sock.send(c, "vi", nguon)) { setText(replyEl, "Chưa kết nối được backend — kiểm tra serve.app đang chạy ở cổng 8000."); return; }
  setText(replyEl, "Đang xử lý…", true);
}

document.querySelectorAll("[data-time]").forEach(b => b.addEventListener("click", () => applyTime(b.dataset.time === "night")));
// Đỗ xe / Đang chạy, Khoá trẻ em: gửi lên backend; nút sáng theo trạng thái backend trả về (dongBo)
document.querySelectorAll("[data-mode]").forEach(b => b.addEventListener("click", () => sock.setVehicle({ dang_chay: b.dataset.mode === "driving" })));
childBtn.addEventListener("click", () => {
  const bat = !vehicle.child_lock;
  if (!sock.setVehicle({ khoa_tre_em: bat })) return;
  focus("doorLeft");
  const t = bat ? "Khoá trẻ em: bật" : "Khoá trẻ em: tắt";
  showCallouts([{ at: ANCHOR.rear_left, text: t }, { at: ANCHOR.rear_right, text: t }], 5000);
});
document.getElementById("chat").addEventListener("submit", e => {
  e.preventDefault(); const t = input.value; input.value = ""; gui(t);
});

sock.connect();
