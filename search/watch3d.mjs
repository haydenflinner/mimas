// mimo solver watch — H³ state-space view. Same event stream as
// watch.mjs (FUZZ_WATCH JSONL tail, wasm worker, replay buffer), but
// the archive is drawn as what it is: a tree growing through
// hyperbolic 3-space, not a 2D position map. Vertices carry
// hyperboloid coordinates (h3.mjs); a single isometry uniform does all
// navigation — clicking a node recenters the ball on it (Möbius
// translate), fly mode puts the camera inside the space (Klein
// projection, geodesics straight), fog is geodesic distance — what you
// can see from a state is its k-step reachable set.
import * as THREE from 'three';
import * as H from './h3.mjs';

const $ = (id) => document.getElementById(id);
const cv = $('cv');

// ---- palette (shared with watch.mjs) ---------------------------------------
const KIND_COLOR = { seed: '#888', beam: '#6ea8ff', steer: '#5fd4d0',
  seek: '#7ed67e', walk: '#b48ef2', splice: '#f2a65e', bank: '#9aa4b8',
  ge: '#e8d44d', expand: '#6ea8ff',
  Cov: '#6ea8ff', Bucket: '#5fd4d0', Max: '#ffd75e', Cell: '#7ed67e',
  State: '#b48ef2', Fault: '#ff6f6f', Edge: '#f2a65e', Dist: '#f2a65e',
  absorb: '#555c6e' };
const VIA_COLOR = { elite: '#ff9ad5', detour: '#ffb75e', goal: '#7ee8ff',
  cell: '#7ed67e', max: '#ffd75e' };
const BG = new THREE.Color('#14161c');

// ---- state ------------------------------------------------------------------
let worker = null, tailTimer = null;
const log = [];
let pos = 0, playing = false, doneInfo = null;
const commitTimes = [];

const st = freshState();
function freshState() {
  return { nodes: [], nodeN: 0, phase: '—', pick: null, lastPickVia: null,
    goal: null, findList: [], covered: 0, maxScore: 0, cells: 0,
    actNames: [], acts: [], pol: [], elites: new Set() };
}
// each node: {i,parent,px,py,score,got,dead,tlen,kind,leaf,pos,fwd}

const KIND = (n) => n.kind || (n.kinds && n.kinds[0]) || '';

// ---- geometry buffers --------------------------------------------------------
const SEG = 6;                    // geodesic subdivisions per edge
const EV = SEG - 1;               // line segs per edge
let capN = 1 << 15;               // 32768 nodes to start
let hposA, colrA, miscA, eposA, ecolA; // f32 GPU backing arrays
let qposA, qcolA, quvA, qpinA, qcorA, qidxA;  // per-node thumbnail quad (4 verts)
// level state — declared here because bindBufs() runs at module init
let levelBBox = null, levelCv = null, levelTex = null, builtRows = null;
let thumbMode = 'near';                    // near | all | off
const QCORNER = [[-1, -1], [1, -1], [-1, 1], [1, 1]];
function allocBufs(n) {
  capN = n;
  hposA = new Float32Array(capN * 4);
  colrA = new Float32Array(capN * 3);
  miscA = new Float32Array(capN * 4); // size, dead, leaf, elite
  eposA = new Float32Array(capN * EV * 2 * 4);
  ecolA = new Float32Array(capN * EV * 2 * 3);
  qposA = new Float32Array(capN * 4 * 4);
  qcolA = new Float32Array(capN * 4 * 3);
  quvA = new Float32Array(capN * 4 * 4);   // level uvMin.xy, uvMax.xy per corner
  qpinA = new Float32Array(capN * 4);      // pin flag per quad (same on all 4 verts)
  qcorA = new Float32Array(capN * 4 * 2);
  qidxA = new Uint32Array(capN * 6);
  for (let i = 0; i < n; i++) {
    const b = i * 6, v = i * 4;
    qidxA[b] = v; qidxA[b + 1] = v + 1; qidxA[b + 2] = v + 2;
    qidxA[b + 3] = v + 2; qidxA[b + 4] = v + 1; qidxA[b + 5] = v + 3;
    for (let k = 0; k < 4; k++) qcorA.set(QCORNER[k], (v + k) * 2);
  }
}
allocBufs(capN);

function growBufs(need) {
  if (need <= capN) return;
  const old = { hposA, colrA, miscA, eposA, ecolA, qposA, qcolA, quvA, qpinA, qcorA, qidxA };
  allocBufs(Math.max(capN * 2, need));
  hposA.set(old.hposA); colrA.set(old.colrA); miscA.set(old.miscA);
  eposA.set(old.eposA); ecolA.set(old.ecolA);
  qposA.set(old.qposA); qcolA.set(old.qcolA); quvA.set(old.quvA);
  qpinA.set(old.qpinA); qcorA.set(old.qcorA); qidxA.set(old.qidxA);
  for (const i of pinnedSet()) writePin(i, 1);
  bindBufs();
}

const colCache = new Map();
const col3 = (hex) => {
  let c = colCache.get(hex);
  if (!c) { c = new THREE.Color(hex); colCache.set(hex, c); }
  return c;
};

// node coloring: 'px' paints the level-progress gradient (spawn → goal
// reads as a hue sweep across the tree), 'kind' the solver's event
// kinds, 'depth' the tape length. px needs the running extent.
let colBy = 'px';
const pxRange = { lo: Infinity, hi: -Infinity };
const depthMax = () => Math.max(1, ...depthA);
const pxFrac = (px) => {
  const w = pxRange.hi - pxRange.lo;
  return w < 1e-9 ? 0.5 : (px - pxRange.lo) / w;
};
function nodeColor(n) {
  if (n.hl) return col3('#ffe9a8');
  if (n.dead) return col3('#5a3a3a');
  if (colBy === 'kind') return col3(KIND_COLOR[KIND(n)] || '#9aa4b8');
  const c = new THREE.Color();
  if (colBy === 'depth') return c.setHSL(0.62 - 0.5 * Math.min(1, (depthA[n.i] || 0) / depthMax()), 0.8, 0.55);
  if (typeof n.px !== 'number') return c.setHSL(0.6, 0.25, 0.5);
  return c.setHSL(0.66 - 0.66 * pxFrac(n.px), 0.85, 0.55); // px: blue spawn → red frontier
}
let pxDirty = false, lastRecolor = 0;
function recolorAll() {
  for (const n of st.nodes) {
    if (!n || !n.pos) continue;
    const c = nodeColor(n);
    colrA[n.i * 3] = c.r; colrA[n.i * 3 + 1] = c.g; colrA[n.i * 3 + 2] = c.b;
  }
  markDirty(0); markDirty(Math.max(0, st.nodeN - 1));
  bufDirty = true;
}

// Insert node i's point + parent edge into the f32 arrays (layout must
// already have filled .pos/.fwd). Called on insert and full rebuild.
function writeNode(n) {
  const i = n.i;
  hposA.set(n.pos, i * 4);
  const c = nodeColor(n);
  colrA[i * 3] = c.r; colrA[i * 3 + 1] = c.g; colrA[i * 3 + 2] = c.b;
  const s = (2.1 + 1.7 * Math.min(1, (n.score || 0) / 40)) * (n.hl ? 1.5 : 1);
  miscA[i * 4] = s;
  miscA[i * 4 + 1] = n.dead ? 1 : 0;
  miscA[i * 4 + 2] = n.leaf ? 1 : 0;
  miscA[i * 4 + 3] = n.elite ? 1 : 0;
  const p = n.parent;
  if (p === undefined || p === null || p < 0 || !st.nodes[p]) {
    // root — fill its edge slots with itself so no zero-verts point at origin
    for (let k = 0; k < EV; k++) eposA.set(n.pos, (i * EV + k) * 8), eposA.set(n.pos, (i * EV + k) * 8 + 4);
    return;
  }
  {
    const P = st.nodes[p].pos;
    for (let k = 0; k < EV; k++) {
      const a = H.geoAB(P, n.pos, k / SEG);
      const b = H.geoAB(P, n.pos, (k + 1) / SEG);
      eposA.set(a, (i * EV + k) * 8);
      eposA.set(b, (i * EV + k) * 8 + 4);
      const ec = n.hl ? col3('#c8a848') : n.dead ? col3('#463030') : col3('#5478b0');
      ecolA[i * EV * 6 + k * 6] = ec.r; ecolA[i * EV * 6 + k * 6 + 1] = ec.g; ecolA[i * EV * 6 + k * 6 + 2] = ec.b;
      ecolA[i * EV * 6 + k * 6 + 3] = ec.r; ecolA[i * EV * 6 + k * 6 + 4] = ec.g; ecolA[i * EV * 6 + k * 6 + 5] = ec.b;
    }
  }
  markDirty(i);
  writeQuad(n);
}

// Thumbnail quad: hpos + color + a uv window into the level texture. The
// window is a fixed world-space rectangle centered on the node's (px,py) —
// the thumb reads as "the game screen at this state" (terrain around the
// player), not a whole-level squint.
const THUMB_W = 280, THUMB_H = 190;   // world units around the player
function writeQuad(n) {
  const i = n.i;
  for (let k = 0; k < 4; k++) {
    qposA.set(n.pos, (i * 4 + k) * 4);
    const c = nodeColor(n);
    qcolA[(i * 4 + k) * 3] = c.r; qcolA[(i * 4 + k) * 3 + 1] = c.g; qcolA[(i * 4 + k) * 3 + 2] = c.b;
  }
  if (levelBBox) {
    const { x0, y0, bw, bh } = levelBBox;
    const u0 = (n.px - THUMB_W / 2 - x0) / bw, u1 = (n.px + THUMB_W / 2 - x0) / bw;
    const v1 = 1 - (n.py - THUMB_H / 2 - y0) / bh, v0 = 1 - (n.py + THUMB_H / 2 - y0) / bh;
    for (let k = 0; k < 4; k++) quvA.set([u0, v0, u1, v1], (i * 4 + k) * 4);
    quadDirty = true;
  }
}

function writePin(i, v) {
  for (let k = 0; k < 4; k++) qpinA[i * 4 + k] = v;
  pinDirty = true;
}
function pinnedSet() {
  const s = new Set();
  if (sel) s.add(sel.i);
  if (st.pick) s.add(st.pick.node);
  if (st.goal && st.goal.node !== undefined) s.add(st.goal.node);
  if (latestI !== null) s.add(latestI);
  return s;
}
let quadDirty = false, pinDirty = false;

// ranged upload tracking — a fuzz flood inserts hundreds of nodes a
// second; whole-buffer needsUpdate re-uploads ~2 MB/frame. Track the
// touched node-index span and addUpdateRange only that.
let dirtyLo = Infinity, dirtyHi = 0;
function markDirty(i) { dirtyLo = Math.min(dirtyLo, i); dirtyHi = Math.max(dirtyHi, i + 1); }
function flushDirty() {
  if (dirtyHi <= dirtyLo) { bufDirty = false; return; }
  const put = (attr, per) => {
    if (attr.addUpdateRange) attr.addUpdateRange(dirtyLo * per, (dirtyHi - dirtyLo) * per);
    attr.needsUpdate = true;
  };
  put(nodeGeo.attributes.hpos, 4);
  put(nodeGeo.attributes.position, 4); // same array, own attribute object
  put(nodeGeo.attributes.colr, 3);
  put(nodeGeo.attributes.misc, 4);
  put(edgeGeo.attributes.hpos, EV * 2 * 4);
  put(edgeGeo.attributes.position, EV * 2 * 4);
  put(edgeGeo.attributes.colr, EV * 2 * 3);
  if (quadGeo) {
    put(quadGeo.attributes.hpos, 4 * 4);
    put(quadGeo.attributes.colr, 4 * 3);
    quadGeo.setDrawRange(0, st.nodeN * 6);
  }
  nodeGeo.setDrawRange(0, st.nodeN);
  edgeGeo.setDrawRange(0, st.nodeN * EV * 2);
  dirtyLo = Infinity; dirtyHi = 0; bufDirty = false;
}

// Incremental layout — mirrors h3.layout() one node at a time so live
// inserts and scrub rebuilds produce identical geometry.
const kids = new Map(), depthA = [];
function placeNode(i) {
  const n = st.nodes[i];
  const p = n.parent;
  if (p === undefined || p === null || p < 0 || !st.nodes[p]) {
    const k = roots++;
    if (k === 0) { n.pos = [1, 0, 0, 0]; n.fwd = [0, 1, 0, 0]; }
    else {
      const d = H.fibDir(k - 1);
      const v = [0, d[0], d[1], d[2]];
      n.pos = H.geo([1, 0, 0, 0], v, 1.9);
      n.fwd = H.geoVel([1, 0, 0, 0], v, 1.9);
    }
    n.leaf = true; depthA[i] = 0; st.nodeN++;
    return;
  }
  depthA[i] = depthA[p] + 1;
  const par = st.nodes[p];
  par.leaf = false;
  writeNode(par); // leaf flag fell off — refresh the parent's point
  const k = kids.get(p) ?? 0;
  kids.set(p, k + 1);
  const P = par.pos, F = par.fwd;
  let v;
  if (k === 0) v = F;
  else {
    // same cone-spread as h3.layout (kept in sync — ortho is local to
    // h3.mjs so recompute the ⊥ basis inline here)
    const refs = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
    const out = [];
    for (let j = 0; j < 3 && out.length < 2; j++) {
      let t = H.projectT(P, refs[(i + j) % 3]);
      t = s2m(t, F, -H.mdot(t, F));
      for (const o of out) t = s2m(t, o, -H.mdot(t, o));
      if (H.mdot(t, t) > 1e-8) out.push(H.normT(t));
    }
    while (out.length < 2) out.push([0, 1, 0, 0]);
    const [u, w] = out;
    const phi = k * 2.39996322972865332;
    const th = Math.min(2.05, 1.15 * Math.sqrt(k / 5));
    const ct = Math.cos(th), stt = Math.sin(th);
    const cp = Math.cos(phi), sp = Math.sin(phi);
    v = [
      ct * F[0] + stt * (cp * u[0] + sp * w[0]),
      ct * F[1] + stt * (cp * u[1] + sp * w[1]),
      ct * F[2] + stt * (cp * u[2] + sp * w[2]),
      ct * F[3] + stt * (cp * u[3] + sp * w[3]),
    ];
  }
  const d = 0.72 * Math.pow(0.955, depthA[i]);
  n.pos = H.onSheet(H.geo(P, v, d));
  n.fwd = H.normT(H.projectT(n.pos, H.geoVel(P, v, d)));
  n.leaf = true;
  st.nodeN++;
}
let roots = 0;
const s2m = (a, b, s) => [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2], a[3] + s * b[3]];

// ---- three.js scene -----------------------------------------------------------
const renderer = new THREE.WebGLRenderer({ canvas: cv, antialias: true });
renderer.setClearColor(BG);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, 1, 0.002, 8);

const VERT = /* glsl */`
attribute vec4 hpos;
attribute vec3 colr;
attribute vec4 misc;
uniform mat4 uIso;
uniform mat4 uVP;
uniform float uPoin;
uniform float uPx;
varying vec3 vC;
varying float vA;
void main() {
  vec4 q = uIso * hpos;
  float w = max(q.x, 1.0);
  float d = acosh(clamp(w, 1.0, 1e12));
  vec3 b = q.yzw / (w + uPoin);
  vec4 clip = uVP * vec4(b, 1.0);
  gl_Position = clip;
  // true perspective size: clip.w is the eye-space depth (-b.z in fly)
  float s = misc.x * uPx * (1.0 + 0.55 * misc.z);
  gl_PointSize = clamp(s / max(clip.w, 0.05), 1.4, 60.0);
  vec3 c = colr;
  c = mix(c, vec3(1.0, 0.84, 0.37), misc.w);      // elite → gold
  c *= 1.0 - 0.62 * misc.y;                       // dead → dim
  vC = c;
  vA = exp(-d * mix(0.14, 0.10, 1.0 - uPoin));
}`;
const FRAG_P = /* glsl */`
precision mediump float;
varying vec3 vC;
varying float vA;
uniform vec3 uBg;
void main() {
  vec2 p = gl_PointCoord * 2.0 - 1.0;
  float r = dot(p, p);
  if (r > 1.0) discard;
  float a = smoothstep(1.0, 0.5, r) * vA;
  gl_FragColor = vec4(mix(uBg, vC, 0.15 + 0.85 * vA), a);
}`;
const VERT_L = /* glsl */`
attribute vec4 hpos;
attribute vec3 colr;
uniform mat4 uIso;
uniform mat4 uVP;
uniform float uPoin;
uniform float uFogK;
varying vec3 vC;
varying float vA;
void main() {
  vec4 q = uIso * hpos;
  float w = max(q.x, 1.0);
  float d = acosh(clamp(w, 1.0, 1e12));
  vec3 b = q.yzw / (w + uPoin);
  gl_Position = uVP * vec4(b, 1.0);
  vC = colr;
  vA = exp(-d * uFogK);
}`;
const FRAG_L = /* glsl */`
precision mediump float;
varying vec3 vC;
varying float vA;
uniform vec3 uBg;
void main() {
  gl_FragColor = vec4(mix(uBg, vC, 0.1 + 0.9 * vA), vA * 0.85);
}`;

// Node thumbnail: a screen-billboarded quad showing the level around the
// node's (px,py) — "the game screen in this state". Screen-space offset
// keeps a constant pixel size; alpha fades in for nodes within uThumbD of
// the view center (camera in fly, focus in ball), and pins always show.
const VERT_Q = /* glsl */`
attribute vec4 hpos;
attribute vec2 corner;
attribute vec4 uvr;
attribute vec3 colr;
attribute float pin;
uniform mat4 uIso;
uniform mat4 uVP;
uniform float uPoin;
uniform vec2 uView;
uniform vec2 uThumbPx;
uniform float uThumbD;
varying vec2 vUv;
varying vec2 vLuv;
varying vec3 vC;
varying float vA;
varying float vPin;
void main() {
  vec4 q = uIso * hpos;
  float w = max(q.x, 1.0);
  float d = acosh(clamp(w, 1.0, 1e12));
  vec3 b = q.yzw / (w + uPoin);
  vec4 clip = uVP * vec4(b, 1.0);
  vUv = corner * 0.5 + 0.5;
  vLuv = mix(uvr.xy, uvr.zw, vUv);
  vC = colr;
  vPin = pin;
  if (clip.w <= 0.001) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); vA = 0.0; return; }
  vec2 px = uThumbPx * (1.0 + 0.4 * pin);
  vec2 off = (corner * 0.5 + vec2(0.62, -0.58)) * px / uView * 2.0 * clip.w;
  gl_Position = clip + vec4(off, 0.0, 0.0);
  float near_ = 1.0 - smoothstep(uThumbD * 0.5, uThumbD, d);
  float close = smoothstep(0.12, 0.6, d);  // fade when you're inside the node
  vA = max(pin, near_ * close) * 0.95;
}`;
const FRAG_Q = /* glsl */`
precision mediump float;
uniform sampler2D uLevel;
varying vec2 vUv;
varying vec2 vLuv;
varying vec3 vC;
varying float vA;
varying float vPin;
void main() {
  if (vA < 0.02) discard;
  vec3 c = texture2D(uLevel, vLuv).rgb;
  // player marker — the state sits at the window's center
  float m = distance(vUv, vec2(0.5));
  c = mix(vC, c, smoothstep(0.055, 0.09, m));
  c = mix(vec3(1.0), c, smoothstep(0.02, 0.05, m));
  // border in the node's own color
  float e = max(abs(vUv.x - 0.5), abs(vUv.y - 0.5));
  c = mix(c, vC, smoothstep(0.42, 0.49, e));
  gl_FragColor = vec4(c, vA * (0.8 + 0.2 * vPin));
}`;

// ShaderMaterial clones its `uniforms` dict (values too) — so the same
// uniform *objects* get installed on every material after creation and
// frame() writes through these.
const U = {
  uIso: new THREE.Matrix4(),
  uVP: new THREE.Matrix4(),
  uPoin: 1,
  uPx: 8,
  uBg: BG,
  uView: new THREE.Vector2(1, 1),
  uThumbPx: new THREE.Vector2(72, 47),
};
const mats = [];
const mkMat = (v, f) => {
  const m = new THREE.ShaderMaterial({
    vertexShader: v, fragmentShader: f,
    uniforms: {
      uIso: { value: U.uIso }, uVP: { value: U.uVP }, uPoin: { value: U.uPoin },
      uPx: { value: U.uPx }, uBg: { value: U.uBg }, uFogK: { value: 0.12 },
      uView: { value: U.uView }, uThumbPx: { value: U.uThumbPx },
      uThumbD: { value: 2.6 }, uLevel: { value: null },
    },
    transparent: true, depthWrite: false,
  });
  // overwrite the clones with the shared objects
  m.uniforms.uIso.value = U.uIso; m.uniforms.uVP.value = U.uVP;
  m.uniforms.uBg.value = U.uBg;
  m.uniforms.uView.value = U.uView; m.uniforms.uThumbPx.value = U.uThumbPx;
  mats.push(m);
  return m;
};
const nodeMat = mkMat(VERT, FRAG_P);
const edgeMat = mkMat(VERT_L, FRAG_L);
const quadMat = mkMat(VERT_Q, FRAG_Q);

let nodeGeo, edgeGeo, quadGeo, points, lines, quads;
function bindBufs() {
  if (points) {
    scene.remove(points); scene.remove(lines); scene.remove(quads);
    nodeGeo.dispose(); edgeGeo.dispose(); quadGeo.dispose();
  }
  nodeGeo = new THREE.BufferGeometry();
  nodeGeo.setAttribute('position', new THREE.BufferAttribute(hposA, 4)); // count anchor
  nodeGeo.setAttribute('hpos', new THREE.BufferAttribute(hposA, 4));
  nodeGeo.setAttribute('colr', new THREE.BufferAttribute(colrA, 3));
  nodeGeo.setAttribute('misc', new THREE.BufferAttribute(miscA, 4));
  nodeGeo.setDrawRange(0, st.nodeN);
  edgeGeo = new THREE.BufferGeometry();
  edgeGeo.setAttribute('position', new THREE.BufferAttribute(eposA, 4));
  edgeGeo.setAttribute('hpos', new THREE.BufferAttribute(eposA, 4));
  edgeGeo.setAttribute('colr', new THREE.BufferAttribute(ecolA, 3));
  edgeGeo.setDrawRange(0, 0);
  quadGeo = new THREE.BufferGeometry();
  quadGeo.setAttribute('position', new THREE.BufferAttribute(qposA, 4));
  quadGeo.setAttribute('hpos', new THREE.BufferAttribute(qposA, 4));
  quadGeo.setAttribute('corner', new THREE.BufferAttribute(qcorA, 2));
  quadGeo.setAttribute('uvr', new THREE.BufferAttribute(quvA, 4));
  quadGeo.setAttribute('colr', new THREE.BufferAttribute(qcolA, 3));
  quadGeo.setAttribute('pin', new THREE.BufferAttribute(qpinA, 1));
  quadGeo.setIndex(new THREE.BufferAttribute(qidxA, 1));
  quadGeo.setDrawRange(0, st.nodeN * 6);
  points = new THREE.Points(nodeGeo, nodeMat);
  lines = new THREE.LineSegments(edgeGeo, edgeMat);
  quads = new THREE.Mesh(quadGeo, quadMat);
  points.frustumCulled = lines.frustumCulled = quads.frustumCulled = false;
  points.renderOrder = 2; lines.renderOrder = 1; quads.renderOrder = 4;
  quads.visible = !!levelBBox;
  scene.add(lines); scene.add(points); scene.add(quads);
}
bindBufs();

// Horizon cage — a lat/long wireframe at depth ~24, the "sphere at
// infinity" feuclide draws. Points are far hyperboloid points, so the
// same isometry moves it correctly under translation (parallax).
{
  const R = 24, pts = [];
  const C = Math.cosh(R), S = Math.sinh(R);
  const push = (d) => pts.push(C, S * d[0], S * d[1], S * d[2]);
  const dir = (th, ph) => [Math.cos(th) * Math.cos(ph), Math.cos(th) * Math.sin(ph), Math.sin(th)];
  const N = 40;
  const hp = [];
  for (let li = -3; li <= 3; li++) {
    const ph = (li / 4) * (Math.PI / 2 - 0.08);
    for (let k = 0; k < N; k++) {
      push(dir((k / N) * 2 * Math.PI, ph));
      push(dir(((k + 1) / N) * 2 * Math.PI, ph));
    }
  }
  for (let lo = 0; lo < 10; lo++) {
    const th = (lo / 10) * 2 * Math.PI;
    for (let k = 0; k <= 20; k++) {
      const ph0 = -Math.PI / 2 + (k / 20) * Math.PI, ph1 = -Math.PI / 2 + ((k + 1) / 20) * Math.PI;
      if (k === 20) break;
      push(dir(th, ph0)); push(dir(th, ph1));
    }
  }
  const arr = new Float32Array(pts);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(arr, 4));
  g.setAttribute('hpos', new THREE.BufferAttribute(arr, 4));
  const cols = new Float32Array(arr.length / 4 * 3);
  for (let i = 0; i < cols.length; i += 3) { cols[i] = 0.16; cols[i + 1] = 0.19; cols[i + 2] = 0.26; }
  g.setAttribute('colr', new THREE.BufferAttribute(cols, 3));
  const cageMat = mkMat(VERT_L, FRAG_L);
  cageMat.uniforms.uFogK.value = 0; // the horizon must not fog out
  const cage = new THREE.LineSegments(g, cageMat);
  cage.frustumCulled = false;
  scene.add(cage);
}

// Depth ruler — rings at fixed geodesic radii around the root, in the
// p1/p2 plane the minimap plots. A scale reference inside the space:
// how many exploration-hops away a state sits.
{
  const pts = [];
  for (const R of [1, 2, 3, 5, 8, 12]) {
    const C = Math.cosh(R), S = Math.sinh(R), N = 72;
    for (let k = 0; k < N; k++) {
      const a = (k / N) * 2 * Math.PI, b = ((k + 1) / N) * 2 * Math.PI;
      pts.push(C, S * Math.cos(a), S * Math.sin(a), 0,
               C, S * Math.cos(b), S * Math.sin(b), 0);
    }
  }
  const arr = new Float32Array(pts);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(arr, 4));
  g.setAttribute('hpos', new THREE.BufferAttribute(arr, 4));
  const cols = new Float32Array(arr.length / 4 * 3);
  for (let i = 0; i < cols.length; i += 3) { cols[i] = 0.22; cols[i + 1] = 0.27; cols[i + 2] = 0.40; }
  g.setAttribute('colr', new THREE.BufferAttribute(cols, 3));
  const ringMat = mkMat(VERT_L, FRAG_L);
  ringMat.uniforms.uFogK.value = 0.045;
  const rings = new THREE.LineSegments(g, ringMat);
  rings.frustumCulled = false;
  scene.add(rings);
}

// ---- level texture: the game's own CSV geometry → one shared texture ---------
// Node thumbnails and the level inset both draw from this canvas.
const ROW_COLOR = { ground: '#3d4351', plat: '#4a5163', lava: '#b04040',
  goomba: '#b07040', coin: '#d8c052', flag: '#4fae62', spawn: '#4fae62',
  goal: '#4fae62' };
const thumbD = () => thumbMode === 'all' ? 1e9 : (mode === 'fly' ? 7.5 : 2.8);
function applyThumb() {
  quadMat.uniforms.uThumbD.value = thumbD();
  if (quads) quads.visible = thumbMode !== 'off' && !!levelBBox;
}
function buildLevel() {
  const rows = meta && meta.rows;
  if (!rows || rows === builtRows) return;
  builtRows = rows;
  let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
  for (const r of rows) {
    x0 = Math.min(x0, +r[2]); y0 = Math.min(y0, +r[3]);
    x1 = Math.max(x1, +r[2] + +r[4]); y1 = Math.max(y1, +r[3] + +r[5]);
  }
  const pad = 30;
  levelBBox = { x0: x0 - pad, y0: y0 - pad, bw: x1 - x0 + 2 * pad, bh: y1 - y0 + 2 * pad };
  const W = 1536, Hh = Math.max(2, Math.round(W * levelBBox.bh / levelBBox.bw));
  levelCv = document.createElement('canvas');
  levelCv.width = W; levelCv.height = Hh;
  const g = levelCv.getContext('2d');
  g.fillStyle = '#191c26'; g.fillRect(0, 0, W, Hh);
  const sx = W / levelBBox.bw, sy = Hh / levelBBox.bh;
  for (const r of rows) {
    const col = ROW_COLOR[r[1]];
    if (!col) continue;
    g.fillStyle = col;
    if (r[1] === 'coin' || r[1] === 'goomba') {
      g.beginPath();
      g.arc((+r[2] + +r[4] / 2 - levelBBox.x0) * sx, (+r[3] + +r[5] / 2 - levelBBox.y0) * sy, 3.5, 0, 7);
      g.fill();
    } else {
      g.fillRect((+r[2] - levelBBox.x0) * sx, (+r[3] - levelBBox.y0) * sy,
        +r[4] * sx, Math.max(2, +r[5] * sy));
    }
  }
  if (levelTex) levelTex.dispose();
  levelTex = new THREE.CanvasTexture(levelCv);
  levelTex.minFilter = THREE.LinearFilter;
  levelTex.magFilter = THREE.LinearFilter;
  quadMat.uniforms.uLevel.value = levelTex;
  // uv windows were written as zeros before meta arrived — fill them now
  for (const n of st.nodes) if (n && n.pos) writeQuad(n);
  quadDirty = true;
  applyThumb();
  fitLvl();
}

// swarm markers — commit rollout endpoints riding geodesics
const SWARM_MAX = 160;
const swarmA = new Float32Array(SWARM_MAX * 4);
const swarmC = new Float32Array(SWARM_MAX * 3);
const swarmM = new Float32Array(SWARM_MAX * 4);
const swarmGeo = new THREE.BufferGeometry();
swarmGeo.setAttribute('position', new THREE.BufferAttribute(swarmA, 4));
swarmGeo.setAttribute('hpos', new THREE.BufferAttribute(swarmA, 4));
swarmGeo.setAttribute('colr', new THREE.BufferAttribute(swarmC, 3));
swarmGeo.setAttribute('misc', new THREE.BufferAttribute(swarmM, 4));
const swarm = new THREE.Points(swarmGeo, mkMat(VERT, FRAG_P));
swarm.frustumCulled = false; swarm.renderOrder = 3;
scene.add(swarm);
const anims = []; // {a,b,t0,via}
const MARK_MS = 900, FADE_MS = 1400;

// pick / goal / selection beacons — plain meshes, positioned in ball
// space each frame from the current isometry.
const beacon = (hex, r = 0.02) => {
  const m = new THREE.Mesh(
    new THREE.SphereGeometry(r, 16, 12),
    new THREE.MeshBasicMaterial({ color: hex, transparent: true, opacity: 0.9, depthTest: false }));
  m.visible = false; m.renderOrder = 5; scene.add(m); return m;
};
const selBeacon = beacon('#ffffff', 0.022);
const pickBeacon = beacon('#9ae0ff', 0.03);
const goalBeacon = beacon('#7ee8ff', 0.028);

// ---- isometry / camera --------------------------------------------------------
let mode = 'ball';                    // 'ball' (Poincaré overview) | 'fly'
let uIsoNow = H.I4();                 // what the shader applied last frame
let focus = H.I4();                   // ball-mode recenter isometry
let focusAnim = null;                 // {T,start}
let cam = H.I4();                     // fly camera pose ∈ O⁺(3,1)
let camInv = H.I4();
let flyTo = null;                     // node index we're flying to
let flyAimed = true;                  // false until fly entry aim ran on a non-empty tree
let follow = true;                    // chase the commit frontier while replaying
let latestI = null;                   // most recent commit's landed node
let orbit = { th: 0.7, ph: 0.5, r: 2.6 };
const camState = { speed: 0.55 };

const isoUniformM4 = () => {
  const M = U.uIso;
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++)
    M.elements[j * 4 + i] = uIsoNow[i * 4 + j]; // our row-major → three's column-major
  return M;
};

/// 4×4 rotation aiming the local −z view axis at local direction d.
function aimRot(dx, dy, dz) {
  const n = Math.hypot(dx, dy, dz) || 1;
  const d = [dx / n, dy / n, dz / n];
  // axis = (−z) × d = (dy, −dx, 0), angle = acos((−z)·d)
  const ax = [d[1], -d[0], 0]; // (0,0,-1)×d
  const an = Math.hypot(ax[0], ax[1]);
  if (an < 1e-6) return d[2] < 0 ? H.I4() : H.rot(1, 0, 0, Math.PI);
  return H.rot(ax[0] / an, ax[1] / an, 0, Math.acos(Math.max(-1, Math.min(1, -d[2]))));
}

/// Where a fresh fly camera goes: the node nearest the tree's ball
/// centroid (inside the canopy, not at the edge), aimed at the bulk.
function treeEntryCam() {
  let x = 0, y = 0, z = 0, c = 0;
  for (const n of st.nodes) {
    if (!n || !n.pos) continue;
    const w = n.pos[0] + 1;
    x += n.pos[1] / w; y += n.pos[2] / w; z += n.pos[3] / w; c++;
  }
  if (!c) return H.I4();
  const m = [x / c, y / c, z / c];
  let mid = null, bd = 1e9;
  for (const n of st.nodes) {
    if (!n || !n.pos) continue;
    const w = n.pos[0] + 1;
    const d = (n.pos[1] / w - m[0]) ** 2 + (n.pos[2] / w - m[1]) ** 2 + (n.pos[3] / w - m[2]) ** 2;
    if (d < bd) { bd = d; mid = n; }
  }
  // lift the ball centroid to the sheet
  const m2 = Math.min(0.9999, m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
  const lift = [(1 + m2) / (1 - m2), 2 * m[0] / (1 - m2), 2 * m[1] / (1 - m2), 2 * m[2] / (1 - m2)];
  // candidate entry poses — score each by how much of the tree lands in
  // the 82° fov, take the best. From inside, the centroid direction can
  // sit in an empty gap between the trunk and the canopy.
  const cands = [];
  const at = (P, lookAt) => {
    const T = H.translateTo(P);
    const lv = H.mapply(H.minv(T), H.toward(P, lookAt));
    return H.mmul(T, aimRot(lv[1], lv[2], lv[3]));
  };
  cands.push(aimRot(lift[1], lift[2], lift[3]));                   // at root, out at the canopy
  const dn = Math.hypot(lift[1], lift[2], lift[3]) || 1;
  const back = H.geo([1, 0, 0, 0], [0, -lift[1] / dn, -lift[2] / dn, -lift[3] / dn], 1.2);
  cands.push(at(H.onSheet(back), lift));                          // pulled back, tree ahead
  cands.push(at(mid.pos, lift));                                  // interior node → centroid
  // aim along a mid-tree node's growth axis — subtree opens ahead
  const fl = H.mapply(H.minv(H.translateTo(mid.pos)), mid.fwd || [0, 1, 0, 0]);
  cands.push(H.mmul(H.translateTo(mid.pos), aimRot(fl[1], fl[2], fl[3])));
  // score = Σ over nodes in the fov cone of 1/(1+d²) — favors views with
  // the tree close and centered, not Klein-compressed at the horizon
  let best = cands[0], bs = -1;
  for (const cnd of cands) {
    const ci = H.minv(cnd);
    let s = 0;
    for (const n of st.nodes) {
      if (!n || !n.pos) continue;
      const q = H.mapply(ci, n.pos);
      const d = Math.acosh(Math.max(1, Math.min(1e12, q[0])));
      const bn = Math.hypot(q[1], q[2], q[3]) || 1;
      if (q[3] < 0 && Math.acos(Math.max(-1, Math.min(1, -q[3] / bn))) < 0.72)
        s += 1 / (1 + d * d);
    }
    if (s > bs) { bs = s; best = cnd; }
  }
  return best;
}

function setMode(m) {
  mode = m;
  $('mode').textContent = m === 'fly'
    ? 'mode fly — wasd/qe move · drag look · wheel speed · g go · c follow · t thumbs · esc back'
    : 'mode ball — drag orbit · wheel zoom · click recenter · dbl-click flies in · f flies · t thumbs';
  for (const mm of mats) mm.uniforms.uPoin.value = m === 'fly' ? 0 : 1;
  if (m === 'fly') {
    camera.fov = 82; camera.near = 0.01; camera.far = 30;
    camera.position.set(0, 0, 0); camera.rotation.set(0, 0, 0); camera.updateMatrixWorld();
    const selN = sel && st.nodes[sel.i];
    if (selN) {
      // arrive at the node looking along its growth axis — the subtree
      // opens up ahead
      const T = H.translateTo(selN.pos);
      const fl = H.mapply(H.minv(T), selN.fwd);
      cam = H.mmul(T, aimRot(fl[1], fl[2], fl[3]));
    } else {
      cam = treeEntryCam();
    }
    // if the tree was empty at entry, aim again once nodes arrive
    flyAimed = !!(selN || st.nodeN);
    camInv = H.minv(cam);
  } else {
    camera.fov = 55; camera.near = 0.01; camera.far = 30;
    focus = H.minv(cam); // come back out looking at where we flew
  }
  camera.updateProjectionMatrix();
  applyThumb();
}

// frame-to-frame camera update (fly mode)
const keys = {};
addEventListener('keydown', (e) => {
  if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT')) return;
  keys[e.key.toLowerCase()] = true;
  if (e.key === 'g' && sel) flyTo = sel.i;
  if (e.key === 'c' && mode === 'fly') follow = !follow;
  if (e.key === 't') { thumbMode = { near: 'all', all: 'off', off: 'near' }[thumbMode]; applyThumb(); }
  if (e.key === 'f' && mode === 'ball') setMode('fly');
  if (e.key === 'Escape' && mode === 'fly') setMode('ball');
  if (e.key === ' ') { e.preventDefault(); setPlaying(!playing); }
});
addEventListener('keyup', (e) => (keys[e.key.toLowerCase()] = false));

function flyCam(dt) {
  if (flyTo !== null && st.nodes[flyTo]) {
    const tgt = st.nodes[flyTo].pos;
    const camPos = [cam[0], cam[4], cam[8], cam[12]];
    const d = H.dist(camPos, tgt);
    const u = H.toward(camPos, tgt);
    const lu = H.mapply(camInv, u);
    const lv = [lu[1], lu[2], lu[3]];
    const ln = Math.hypot(...lv) || 1;
    const step = Math.min(d - 0.32, camState.speed * 1.7 * dt);
    if (step > 0.002) cam = H.mmul(cam, H.boost(lv[0] / ln, lv[1] / ln, lv[2] / ln, step));
    // steer the view toward the target: rotate local −z to lu's bearing
    const az = Math.atan2(-lv[0], -lv[2]);
    const el = Math.atan2(lv[1], Math.hypot(lv[0], lv[2]));
    const rr = Math.min(1, 4 * dt);
    cam = H.mmul(cam, H.rot(0, 1, 0, az * rr));
    cam = H.mmul(cam, H.rot(1, 0, 0, el * rr));
    if (d < 0.45) flyTo = null;
  } else {
    flyTo = null;
    const sp = camState.speed * dt;
    const manual = keys.w || keys.s || keys.a || keys.d || keys.q || keys.e ||
      keys.arrowup || keys.arrowdown || keys.arrowleft || keys.arrowright;
    if (manual) follow = false;          // hands on the stick → stop chasing
    if (follow && playing && latestI !== null && st.nodes[latestI] && st.nodes[latestI].pos) {
      // soft chase-cam: hover ~0.8 off the newest commit, ease-aimed at it.
      // The frontier keeps moving, so the search reads as "happening" around you.
      const tgt = st.nodes[latestI].pos;
      const camPos = [cam[0], cam[4], cam[8], cam[12]];
      const d = H.dist(camPos, tgt);
      const u = H.toward(camPos, tgt);
      const lu = H.mapply(camInv, u);
      const lv = [lu[1], lu[2], lu[3]];
      const ln = Math.hypot(...lv) || 1;
      const az = Math.atan2(-lv[0], -lv[2]);
      const el = Math.atan2(lv[1], Math.hypot(lv[0], lv[2]));
      const rr = Math.min(1, 2.2 * dt);
      cam = H.mmul(cam, H.rot(0, 1, 0, az * rr));
      cam = H.mmul(cam, H.rot(1, 0, 0, el * rr));
      if (d > 1.1) cam = H.mmul(cam, H.boost(lv[0] / ln, lv[1] / ln, lv[2] / ln, Math.min(d - 1.1, sp * 1.3)));
      else if (d < 0.5) cam = H.mmul(cam, H.boost(-lv[0] / ln, -lv[1] / ln, -lv[2] / ln, Math.min(0.5 - d, sp * 0.6)));
    }
    const mv = (dx, dy, dz, r) => (cam = H.mmul(cam, H.boost(dx, dy, dz, r)));
    if (keys.w || keys.arrowup) mv(0, 0, -1, sp);
    if (keys.s || keys.arrowdown) mv(0, 0, 1, sp);
    if (keys.a || keys.arrowleft) mv(-1, 0, 0, sp);
    if (keys.d || keys.arrowright) mv(1, 0, 0, sp);
    if (keys.q) mv(0, 1, 0, sp);
    if (keys.e) mv(0, -1, 0, sp);
  }
  camInv = H.minv(cam);
}

// pointer: drag orbits (ball) / looks (fly); click picks a node
let drag = null;
cv.addEventListener('pointerdown', (e) => {
  drag = { x: e.clientX, y: e.clientY, moved: 0 };
  cv.setPointerCapture(e.pointerId);
});
cv.addEventListener('pointermove', (e) => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag.x = e.clientX; drag.y = e.clientY;
  drag.moved += Math.abs(dx) + Math.abs(dy);
  if (mode === 'ball') {
    orbit.th -= dx * 0.006; orbit.ph = Math.max(-1.4, Math.min(1.4, orbit.ph - dy * 0.006));
  } else {
    cam = H.mmul(cam, H.rot(0, 1, 0, -dx * 0.004));
    cam = H.mmul(cam, H.rot(1, 0, 0, -dy * 0.004));
    flyTo = null;
    follow = false;
  }
});
cv.addEventListener('pointerup', (e) => {
  if (drag && drag.moved < 6) pickAt(e.clientX, e.clientY);
  drag = null;
});
cv.addEventListener('dblclick', (e) => {
  // jump inside, at the node under the cursor
  pickAt(e.clientX, e.clientY);
  if (sel) setMode('fly');
});
cv.addEventListener('wheel', (e) => {
  e.preventDefault();
  if (mode === 'ball') orbit.r = Math.max(1.04, Math.min(9, orbit.r * (1 + Math.sign(e.deltaY) * 0.1)));
  else camState.speed = Math.max(0.06, Math.min(3, camState.speed * (1 - Math.sign(e.deltaY) * 0.12)));
}, { passive: false });

// ---- picking (JS mirror of the shader transform) -------------------------------
const uPoinVal = () => nodeMat.uniforms.uPoin.value;
const tmpV = new THREE.Vector4();
function projectNode(n) {
  const q = H.mapply(uIsoNow, n.pos);
  const w = Math.max(q[0], 1);
  const den = w + uPoinVal();
  tmpV.set(q[1] / den, q[2] / den, q[3] / den, 1).applyMatrix4(vp);
  if (tmpV.w <= 0) return null;
  const nx = tmpV.x / tmpV.w, ny = tmpV.y / tmpV.w;
  if (nx < -1.1 || nx > 1.1 || ny < -1.1 || ny > 1.1) return null;
  return { x: (nx * 0.5 + 0.5) * cv.clientWidth, y: (1 - (ny * 0.5 + 0.5)) * cv.clientHeight, d: Math.acosh(Math.min(w, 1e12)) };
}
let sel = null;
let hlPath = [];
let prevPins = new Set();
function updateHl() {
  for (const i of hlPath) { const n = st.nodes[i]; if (n) { n.hl = false; writeNode(n); } }
  hlPath = [];
  for (let c = sel ? sel.i : -1; c !== undefined && c !== null && c >= 0 && st.nodes[c]; c = st.nodes[c].parent) {
    st.nodes[c].hl = true; writeNode(st.nodes[c]); hlPath.push(c);
  }
  bufDirty = true;
}
function pickAt(mx, my) {
  const r = cv.getBoundingClientRect();
  let best = null, bd = 18 * 18;
  for (const n of st.nodes) {
    if (!n || !n.pos) continue;
    const p = projectNode(n);
    if (!p) continue;
    const dd = (p.x - (mx - r.left)) ** 2 + (p.y - (my - r.top)) ** 2;
    if (dd < bd) { bd = dd; best = n; }
  }
  sel = best;
  updateHl();
  if (best && mode === 'ball') {
    // recenter: animate a Möbius translate taking this node to origin —
    // partial boost along −(focus·pos)'s spatial direction, eased in.
    const q = H.onSheet(H.mapply(focus, best.pos));
    const wv = Math.hypot(q[1], q[2], q[3]);
    if (wv > 1e-6)
      focusAnim = { dir: [-q[1] / wv, -q[2] / wv, -q[3] / wv],
        d: Math.acosh(Math.max(1, q[0])), t0: performance.now(), base: focus };
  }
  showNode();
}

// ---- event application ----------------------------------------------------------
function apply(ev) {
  let { t, d } = ev;
  if (d === undefined) { d = { ...ev }; delete d.t; }
  if (t === 'phase') { st.phase = d.name + (d.goal !== undefined ? ' → ' + decode(d.goal) : ''); return; }
  if (t === 'actions') { st.actNames = d.list || []; return; }
  if (t === 'grid') { st.acts = d.acts || []; st.pol = d.pol || []; st.cells = d.cells ?? st.cells; return; }
  if (t === 'micro') return;
  if (t === 'meta') { meta = Object.assign(meta || {}, d); buildLevel(); return; }
  if (t === 'pick') { st.pick = d; st.lastPickVia = d.via; st.goal = null; st.phase = st.phase === '—' ? 'fuzz' : st.phase; return; }
  if (t === 'goal') { st.goal = d; return; }
  if (t === 'route' || t === 'sims') return;
  if (t === 'max') { const n = st.nodes[d.node]; if (n) { n.elite = true; st.elites.add(d.node); writeNode(n); } return; }
  if (t === 'node') {
    growBufs(d.i + 1);
    if (typeof d.px === 'number') {
      if (d.px < pxRange.lo || d.px > pxRange.hi) pxDirty = true;
      if (d.px < pxRange.lo) pxRange.lo = d.px;
      if (d.px > pxRange.hi) pxRange.hi = d.px;
    }
    const n = st.nodes[d.i] = { ...d, leaf: true };
    placeNode(d.i);
    writeNode(n);
    st.maxScore = Math.max(st.maxScore, d.score || 0);
    bufDirty = true;
    return;
  }
  if (t === 'commit') {
    const landed = d.node ?? d.landed;
    const parent = d.parent ?? (landed !== undefined && st.nodes[landed] ? st.nodes[landed].parent : undefined);
    if (anims.length >= SWARM_MAX) anims.shift();
    if (parent !== undefined && st.nodes[parent])
      anims.push({ a: parent, b: landed, t0: performance.now(), via: ev._via ?? st.lastPickVia });
    if (landed !== undefined && landed !== null) latestI = landed;
    commitTimes.push(performance.now());
    if (d.cells !== undefined) st.cells = d.cells;
    return;
  }
  if (t === 'find') { st.findList.push(d.kind ? `${d.kind} ${d.item}` : d.item); st.covered++; return; }
  if (t === 'done') { doneInfo = d; st.phase = 'done'; return; }
  if (t === 'error') { $('err').textContent = d; return; }
}
let meta = null, bufDirty = false;

function replayTo(n) {
  // rebuild event-derived state, then relayout + rewrite the buffers —
  // layout is order-deterministic so a scrub lands on the same shape
  Object.assign(st, freshState());
  kids.clear(); depthA.length = 0; roots = 0;
  pxRange.lo = Infinity; pxRange.hi = -Infinity;
  hposA.fill(0); eposA.fill(0);
  doneInfo = null;
  anims.length = 0;
  hlPath = [];
  for (let i = 0; i < n; i++) apply(log[i]);
  pos = n;
  nodeGeo.setDrawRange(0, st.nodeN);
  edgeGeo.setDrawRange(0, st.nodeN * EV * 2);
  quadGeo.setDrawRange(0, st.nodeN * 6);
  swarmGeo.setDrawRange(0, 0);
  bufDirty = true;
  drawHud();
}

function decode(item) {
  if (typeof item !== 'string' || !meta || !meta.points) return String(item);
  const [k, rest] = [item[0], item.slice(2)];
  if (k === 'p' && meta) {
    const p = meta.points[+rest];
    return p ? 'point line ' + p.line : item;
  }
  if (k === 'd' && meta) {
    const [di, w] = rest.split(':');
    const d = meta.decs[+di];
    return d ? 'line ' + d.line + ' ' + w + ' «' + d.text + '»' : item;
  }
  return item;
}
const escapeHtml = (s) => s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

// ---- HUD ------------------------------------------------------------------------
function showNode() {
  const el = $('node');
  if (!sel) { el.innerHTML = 'click a dot'; return; }
  const n = sel;
  el.innerHTML =
    `#<b>${n.i}</b> ← ${n.parent ?? '?'} · ${KIND(n) || '—'} · pos <b>${(n.px || 0).toFixed(0)},${(n.py || 0).toFixed(0)}</b>` +
    ` · score <b>${n.score ?? '?'}</b> · coins <b>${n.got ?? '?'}</b> · tape <b>${n.tlen ?? '?'}</b>` +
    `${n.dead ? ' · <b style="color:#ff8f8f">dead</b>' : ''}${n.elite ? ' · <b style="color:#ffd75e">elite</b>' : ''}` +
    ` · depth <b>${depthA[n.i] ?? '?'}</b>`;
}

function drawHud() {
  const cnow = performance.now();
  while (commitTimes.length && commitTimes[0] < cnow - 2000) commitTimes.shift();
  $('status').innerHTML =
    `phase <b>${st.phase}</b> · archive <b>${st.nodeN}</b>` +
    ` · expanded <b>${doneInfo ? doneInfo.expanded : '…'}</b>` +
    ` · finds <b>${st.findList.length}</b>` +
    ` · roll/s <b>${(commitTimes.length / 2).toFixed(1)}</b>` +
    (st.cells ? ` · cells <b>${st.cells}</b>` : '') +
    ` · maxScore <b>${Math.max(st.maxScore, doneInfo?.maxScore || 0)}</b>` +
    (mode === 'fly' ? ` · follow <b>${follow ? 'on' : 'off'}</b> · spd <b>${camState.speed.toFixed(2)}</b>` : '') +
    (doneInfo ? ` · <b>${doneInfo.ms}ms</b>` + (doneInfo.found ? ' · <b style="color:#b7e59a">SOLVED</b>' : '') : '');
  const gd = $('decision');
  if (st.pick) {
    const p = st.nodes[st.pick.node];
    const extra = Object.entries(st.pick)
      .filter(([k]) => !['t', 'node', 'via'].includes(k))
      .map(([k, v]) => `${k} <b>${v}</b>`).join(' · ');
    gd.innerHTML = `pick <b>${st.pick.node}</b> via <b>${st.pick.via}</b>` +
      (extra ? ` · ${extra}` : '') +
      (p ? ` · pos (${(p.px || 0).toFixed(0)},${(p.py || 0).toFixed(0)}) score ${p.score ?? '?'}` : '');
  } else if (st.goal) {
    const gtxt = st.goal.goal ?? st.goal.id ?? st.goal.d;
    if (gtxt !== undefined) gd.innerHTML = `goal <b>${decode(String(gtxt))}</b>`;
  }
  $('evpos').textContent = pos + '/' + log.length;
  if (st.acts && st.acts.length) {
    const mw = Math.max(...st.acts.map((a) => a[1]), 1);
    $('cands').innerHTML =
      '<div style="opacity:.6;margin-bottom:4px">learned action win-rates</div>' +
      st.acts.map(([tries, wins], i) => {
        const rate = tries ? wins / tries : 0;
        const w = Math.round((90 * wins) / mw);
        return `<div class="cand"><b>${escapeHtml(st.actNames[i] ?? i)}</b> ${wins}/${tries} (${(rate * 100).toFixed(0)}%) ` +
          `<span style="display:inline-block;height:8px;width:${w}px;background:#7ed67e;vertical-align:middle"></span></div>`;
      }).join('');
  }
  $('findlist').innerHTML = st.findList.slice(-40).reverse()
    .map((it) => `<div class="find">${escapeHtml(decode(it))} <span class="ln">${it}</span></div>`).join('');
}

// ---- beacons / swarm per-frame update ---------------------------------------------
const beaconAt = (mesh, nodeIdx) => {
  const n = st.nodes[nodeIdx];
  if (!n || !n.pos) { mesh.visible = false; return; }
  const q = H.mapply(uIsoNow, n.pos);
  const den = q[0] + uPoinVal();
  mesh.position.set(q[1] / den, q[2] / den, q[3] / den);
  mesh.visible = true;
};

function frame(dt) {
  // mode cameras
  if (mode === 'fly') {
    flyCam(dt);
    uIsoNow = camInv;
    camera.position.set(0, 0, 0); camera.rotation.set(0, 0, 0);
    camera.updateMatrixWorld();
  } else {
    if (focusAnim) {
      const a = focusAnim;
      const t = Math.min(1, (performance.now() - a.t0) / 650);
      const e = t * t * (3 - 2 * t);
      focus = H.mmul(H.boost(a.dir[0], a.dir[1], a.dir[2], a.d * e), a.base);
      if (t >= 1) focusAnim = null;
    }
    const { th, ph, r } = orbit;
    camera.position.set(r * Math.cos(ph) * Math.cos(th), r * Math.sin(ph), r * Math.cos(ph) * Math.sin(th));
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    uIsoNow = focus;
  }
  isoUniformM4();
  vp.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  U.uVP.copy(vp);

  // resize
  const w = cv.clientWidth, h = cv.clientHeight, dpr = devicePixelRatio;
  if (cv.width !== (w * dpr) | 0 || cv.height !== (h * dpr) | 0) {
    renderer.setSize(w, h, false);
    renderer.setPixelRatio(dpr);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  U.uView.set(cv.width, cv.height);
  U.uThumbPx.set(72 * dpr, 47 * dpr);

  // thumbnail pins — sel/pick/goal/frontier always show their level card
  if (quadGeo) {
    const pins = pinnedSet();
    for (const i of prevPins) if (!pins.has(i) && st.nodes[i]) writePin(i, 0);
    for (const i of pins) if (!prevPins.has(i) && st.nodes[i]) writePin(i, 1);
    prevPins = pins;
    if (pinDirty) { quadGeo.attributes.pin.needsUpdate = true; pinDirty = false; }
    if (quadDirty) { quadGeo.attributes.uvr.needsUpdate = true; quadDirty = false; }
  }

  // beacons
  if (sel && st.nodes[sel.i]) beaconAt(selBeacon, sel.i); else selBeacon.visible = false;
  if (st.pick && st.nodes[st.pick.node]) beaconAt(pickBeacon, st.pick.node); else pickBeacon.visible = false;
  if (st.goal && st.nodes[st.goal.node]) beaconAt(goalBeacon, st.goal.node); else goalBeacon.visible = false;

  // swarm — markers ride the geodesic parent→landed for MARK_MS then fade
  let sn = 0;
  const now = performance.now();
  for (let i = anims.length - 1; i >= 0; i--) {
    const a = anims[i];
    const t = now - a.t0;
    if (t > FADE_MS) { anims.splice(i, 1); continue; }
    const A = st.nodes[a.a], B = a.b !== undefined ? st.nodes[a.b] : null;
    if (!A || !A.pos) continue;
    const u = B && B.pos ? Math.min(1, t / MARK_MS) : 1;
    const p = B && B.pos ? H.geoAB(A.pos, B.pos, u * u * (3 - 2 * u)) : A.pos;
    swarmA.set(p, sn * 4);
    const c = col3(VIA_COLOR[a.via] || '#9ae0ff');
    swarmC[sn * 3] = c.r; swarmC[sn * 3 + 1] = c.g; swarmC[sn * 3 + 2] = c.b;
    swarmM[sn * 4] = 7 * (1 - t / FADE_MS) + 2;
    swarmM[sn * 4 + 1] = 0; swarmM[sn * 4 + 2] = 0; swarmM[sn * 4 + 3] = 0;
    sn++;
  }
  swarmGeo.setDrawRange(0, sn);
  swarmGeo.attributes.hpos.needsUpdate = true;
  swarmGeo.attributes.position.needsUpdate = true;
  swarmGeo.attributes.colr.needsUpdate = true;
  swarmGeo.attributes.misc.needsUpdate = true;

  if (bufDirty) flushDirty();
  const tnow = performance.now();
  if (pxDirty && tnow - lastRecolor > 800) {
    pxDirty = false; lastRecolor = tnow; recolorAll();
    flushDirty();
  }
  renderer.render(scene, camera);
}
// ---- minimap: Klein-disk top-down — a fixed global frame H³ otherwise lacks
const mapCv = $('map'), mg = mapCv.getContext('2d');
let lastMap = 0;
function drawMap() {
  const S = mapCv.width, R = S / 2, rr = R - 4;
  mg.clearRect(0, 0, S, S);
  mg.fillStyle = 'rgba(13,15,20,.6)';
  mg.beginPath(); mg.arc(R, R, R - 1, 0, 7); mg.fill();
  mg.strokeStyle = '#3a4054'; mg.stroke();
  // distance contours — Klein radius tanh(ρ) is true geodesic distance
  // from the root, so these are labeled "hops from seed"
  mg.font = '9px ui-monospace,monospace';
  for (const rho of [1, 2, 3, 5, 8]) {
    const r = Math.tanh(rho) * rr;
    mg.strokeStyle = 'rgba(90,105,150,.4)';
    mg.beginPath(); mg.arc(R, R, r, 0, 7); mg.stroke();
    mg.fillStyle = 'rgba(140,152,185,.9)';
    mg.fillText(String(rho), R + 2, R - r + 3);
  }
  mg.fillStyle = 'rgba(140,152,185,.9)';
  mg.fillText('root', R + 4, R - 3);
  mg.fillText('klein · ring = hops from seed', 8, S - 6);
  const toXY = (p) => { const w = p[0] || 1; return [R + (p[1] / w) * rr, R - (p[2] / w) * rr]; };
  for (const n of st.nodes) {
    if (!n || !n.pos) continue;
    const [x, y] = toXY(n.pos);
    if (x < 2 || x > S - 2 || y < 2 || y > S - 2) continue;
    const c = nodeColor(n);
    mg.fillStyle = `rgb(${c.r * 255 | 0},${c.g * 255 | 0},${c.b * 255 | 0})`;
    mg.fillRect(x - 1, y - 1, 2.4, 2.4);
  }
  const ring = (i, col, r = 4) => {
    const n = st.nodes[i]; if (!n || !n.pos) return;
    const [x, y] = toXY(n.pos);
    mg.strokeStyle = col; mg.beginPath(); mg.arc(x, y, r, 0, 7); mg.stroke();
  };
  if (latestI !== null) ring(latestI, '#ff9ad5', 3);   // the frontier
  if (sel) ring(sel.i, '#fff');
  if (st.pick) ring(st.pick.node, '#9ae0ff', 5);
  if (st.goal) ring(st.goal.node, '#7ee8ff', 6);
  if (mode === 'fly') {
    // camera: col0 is position on the sheet; col3 is the local +z axis
    const cp = [cam[0], cam[4], cam[8], cam[12]];
    const [x, y] = toXY(cp);
    const fx = -cam[7], fy = -cam[11], fn = Math.hypot(fx, fy) || 1;
    mg.strokeStyle = '#e6e9f0';
    mg.beginPath(); mg.arc(x, y, 3, 0, 7); mg.stroke();
    mg.beginPath(); mg.moveTo(x, y); mg.lineTo(x + (fx / fn) * 13, y - (fy / fn) * 13); mg.stroke();
  } else {
    // where the ball is currently centered (focus⁻¹ · origin)
    const fc = H.minv(focus);
    const [x, y] = toXY([fc[0], fc[4], fc[8], fc[12]]);
    mg.strokeStyle = '#8b93a5';
    mg.beginPath(); mg.moveTo(x - 4, y); mg.lineTo(x + 4, y); mg.moveTo(x, y - 4); mg.lineTo(x, y + 4); mg.stroke();
  }
}

// ---- level inset: every node plotted at its true game position --------------
// The tree's *other* axis: structure is radial in H³, but (px,py) is where
// the state actually is in the game. This is the same data watch.mjs draws.
const lvlCv = $('lvl'), lg = lvlCv.getContext('2d');
let lvlW = 0;
function fitLvl() {
  if (!levelBBox) return;
  // fit the space left of the minimap; levels can be very wide (side-
  // scrollers) so height collapses for them — px ticks carry the info
  const stageW = cv.clientWidth || 800;
  const W = Math.max(160, Math.min(380, stageW - 196));
  lvlCv.width = W;
  lvlCv.height = Math.max(30, Math.min(120, Math.round(W * levelBBox.bh / levelBBox.bw))) + 16;
  lvlW = stageW;
}
function drawLvl() {
  if (!levelCv || !levelBBox) { lvlCv.style.display = 'none'; return; }
  lvlCv.style.display = 'block';
  if (!lvlCv.width || lvlCv.width < 8 || Math.abs(cv.clientWidth - lvlW) > 60) fitLvl();
  const W = lvlCv.width, Hh = lvlCv.height - 16;
  lg.clearRect(0, 0, W, lvlCv.height);
  lg.drawImage(levelCv, 0, 0, W, Hh);
  const X = (px) => (px - levelBBox.x0) / levelBBox.bw * W;
  const Y = (py) => (py - levelBBox.y0) / levelBBox.bh * Hh;
  // selected node's ancestry in game space — the gold path, un-hyperbolic
  if (hlPath.length > 1) {
    lg.strokeStyle = 'rgba(255,220,120,.85)';
    lg.lineWidth = 1.5;
    lg.beginPath();
    let first = true;
    for (const i of hlPath) {
      const n = st.nodes[i];
      if (!n || n.px === undefined) continue;
      if (first) { lg.moveTo(X(n.px), Y(n.py)); first = false; }
      else lg.lineTo(X(n.px), Y(n.py));
    }
    lg.stroke();
    lg.lineWidth = 1;
  }
  for (const n of st.nodes) {
    if (!n || n.px === undefined) continue;
    const c = nodeColor(n);
    lg.fillStyle = `rgba(${c.r * 255 | 0},${c.g * 255 | 0},${c.b * 255 | 0},.75)`;
    lg.fillRect(X(n.px) - 1, Y(n.py) - 1, 2.4, 2.4);
  }
  const ring = (i, col, r = 4) => {
    const n = st.nodes[i];
    if (!n || n.px === undefined) return;
    lg.strokeStyle = col;
    lg.beginPath(); lg.arc(X(n.px), Y(n.py), r, 0, 7); lg.stroke();
  };
  if (latestI !== null) ring(latestI, '#ff9ad5', 3);
  if (sel) ring(sel.i, '#fff');
  if (st.pick) ring(st.pick.node, '#9ae0ff', 5);
  if (st.goal) ring(st.goal.node, '#7ee8ff', 6);
  // px axis ticks — progress through the level is the dimension that matters
  lg.strokeStyle = 'rgba(140,152,185,.35)';
  lg.fillStyle = 'rgba(140,152,185,.95)';
  lg.font = '9px ui-monospace,monospace';
  for (const f of [0, .25, .5, .75, 1]) {
    const x = X(levelBBox.x0 + f * levelBBox.bw);
    lg.beginPath(); lg.moveTo(x, Hh); lg.lineTo(x, Hh + 3); lg.stroke();
    if (f < 1) lg.fillText(Math.round(levelBBox.x0 + f * levelBBox.bw), x + 2, lvlCv.height - 4);
  }
  lg.fillText('px →', Math.max(2, X(levelBBox.x0 + levelBBox.bw) - 24), lvlCv.height - 4);
}
// ---- legend: what the encodings mean -------------------------------------------
let legendKey = '';
function drawLegend() {
  const kinds = [...new Set(st.nodes.filter(n => n).map(n => KIND(n)))].filter(Boolean);
  const key = colBy + '|' + kinds.join() + '|' + !!levelBBox + '|' + thumbMode;
  if (key === legendKey) return;
  legendKey = key;
  const sw = (bg) => `<span style="display:inline-block;width:22px;height:8px;border-radius:2px;background:${bg};vertical-align:middle;margin-right:5px"></span>`;
  const rows = [];
  if (colBy === 'px')
    rows.push(sw('linear-gradient(90deg,#6ea8ff,#8f6fb0,#e04c4c)') + '<b>px</b> — player x · spawn → flag');
  else if (colBy === 'kind')
    rows.push(kinds.map(k =>
      `<span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${KIND_COLOR[k] || '#9aa4b8'};vertical-align:middle;margin-right:3px"></span>${k}`).join(' '));
  else
    rows.push(sw('linear-gradient(90deg,#6fa0e0,#e04c4c)') + '<b>depth</b> — tape length');
  rows.push('<span style="color:#ffe9a8">●</span> ancestry · <span style="color:#ff9ad5">○</span> newest · <span style="color:#9ae0ff">○</span> pick · <span style="color:#7ee8ff">○</span> goal · dim = dead');
  if (levelBBox) rows.push(`cards show level @ (px,py) · <b>t</b>: ${thumbMode}`);
  $('legend').innerHTML = rows.join('<br>');
}

const vp = new THREE.Matrix4();
let lastT = performance.now();
function loop() {
  const now = performance.now();
  const dt = Math.min(0.1, (now - lastT) / 1000);
  lastT = now;
  if (mode === 'fly' && !flyAimed && st.nodeN && pos >= log.length) {
    // wait until the replay caught up so the entry aim sees the whole tree
    cam = treeEntryCam(); camInv = H.minv(cam); flyAimed = true;
  }
  frame(dt);
  if (now - lastMap > 100) { lastMap = now; drawMap(); drawLvl(); drawLegend(); }
  requestAnimationFrame(loop);
}

// ---- replay pump -------------------------------------------------------------------
function pump() {
  if (!playing) return;
  const stepv = [1, 2, 4, 8, 16, 32, 64, 128][+$('speed').value - 1] || 4;
  const from = pos;
  const n = Math.min(pos + stepv, log.length);
  // incremental apply — apply() writes buffers as it goes
  for (let i = from; i < n; i++) {
    if (log[i].t === 'commit') log[i]._via = st.lastPickVia;
    apply(log[i]);
  }
  pos = n;
  const hnow = performance.now();
  if (hnow - lastHud > 150) { lastHud = hnow; drawHud(); }
  if (pos >= log.length && doneInfo) setPlaying(false);
  requestAnimationFrame(pump);
}
let lastHud = 0;
function setPlaying(v) {
  playing = v;
  $('play').textContent = playing ? 'pause' : 'play';
  if (playing) requestAnimationFrame(pump);
}
$('play').onclick = () => setPlaying(!playing);
$('scrub').oninput = () => { setPlaying(false); commitTimes.length = 0; replayTo(+$('scrub').value); };

// ---- ingest: worker + tail + file + demo ----------------------------------------------
function resetLog() {
  if (worker) { worker.terminate(); worker = null; }
  if (tailTimer) { clearInterval(tailTimer); tailTimer = null; }
  log.length = 0; pos = 0; doneInfo = null; meta = null;
  levelBBox = null; levelCv = null; builtRows = null;
  if (levelTex) { levelTex.dispose(); levelTex = null; }
  quadMat.uniforms.uLevel.value = null;
  prevPins = new Set(); pinDirty = false; quadDirty = false;
  qpinA.fill(0);
  if (quads) quads.visible = false;
  Object.assign(st, freshState());
  kids.clear(); depthA.length = 0; roots = 0;
  pxRange.lo = Infinity; pxRange.hi = -Infinity;
  sel = null; anims.length = 0; commitTimes.length = 0;
  hlPath = []; latestI = null; follow = true;
  focus = H.I4(); focusAnim = null; cam = H.I4(); camInv = H.I4(); flyTo = null;
  flyAimed = mode !== 'fly'; pxDirty = false;
  hposA.fill(0); eposA.fill(0);
  nodeGeo.setDrawRange(0, 0); edgeGeo.setDrawRange(0, 0);
  $('err').textContent = ''; $('findlist').innerHTML = '';
  $('scrub').max = 0;
}
const pushEv = (ev) => { log.push(ev); $('scrub').max = log.length; };

$('run').onclick = () => {
  resetLog();
  worker = new Worker('./worker.mjs', { type: 'module' });
  worker.onmessage = (e) => {
    pushEv(e.data);
    if (e.data.t === 'error' || e.data.t === 'done') setPlaying(pos >= log.length ? false : playing);
  };
  worker.postMessage({ cmd: 'run', base: './mimo', budget: +$('budget').value || 6000,
    strategy: $('strategy').value, depth: +$('gedepth').value || 500 });
  $('play').disabled = false;
  setPlaying(true);
};

async function tail(path) {
  resetLog();
  let off = -1;
  const poll = async () => {
    try {
      const r = await fetch(`/log?path=${encodeURIComponent(path)}&after=${off}`);
      const j = await r.json();
      off = j.next;
      for (const l of j.lines) { try { pushEv(JSON.parse(l)); } catch {} }
    } catch {}
  };
  await poll();
  tailTimer = setInterval(poll, 400);
  $('play').disabled = false;
  setPlaying(true);
}
$('tail').onclick = () => tail($('logpath').value || '/tmp/mimo-fuzz.jsonl');

cv.addEventListener('dragover', (e) => e.preventDefault());
addEventListener('drop', async (e) => {
  e.preventDefault();
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (!f) return;
  resetLog();
  for (const l of (await f.text()).split('\n')) {
    if (!l.trim()) continue;
    try { pushEv(JSON.parse(l)); } catch {}
  }
  $('play').disabled = false;
  setPlaying(true);
});

// ?demo — synthetic tree so the view works with no solver running:
// a random-branching archive with a champion spine, commits, picks.
function demo() {
  resetLog();
  let N = 2600;
  pushEv({ t: 'actions', d: { list: ['none', '↑', '←', '→', '↑←', '↑→'] } });
  // synthetic level so thumbnails + the inset work without a real run —
  // a row per ground/plat/hazard/coin/flag, same CSV shape the driver emits
  const rows = [['g0', 'ground', 0, 560, 3800, 90]];
  for (let k = 0; k < 11; k++) {
    rows.push(['p' + k, 'plat', 240 + k * 320, 340 - (k % 3) * 46, 170, 18]);
    if (k % 3 === 1) rows.push(['l' + k, 'lava', 330 + k * 320, 548, 110, 14]);
    if (k % 2 === 0) rows.push(['c' + k, 'coin', 300 + k * 320, 300 - (k % 3) * 46, 14, 14]);
  }
  rows.push(['flag', 'flag', 3600, 460, 26, 100]);
  pushEv({ t: 'meta', d: { rows } });
  const px = [320], py = [300];
  for (let i = 0; i < N; i++) {
    const parent = i === 0 ? -1 : Math.max(0, (Math.random() < 0.7 ? i - 1 - (Math.random() * 4 | 0) : (Math.random() * i) | 0));
    const ppx = parent >= 0 ? px[parent] : 320, ppy = parent >= 0 ? py[parent] : 300;
    px[i] = ppx + (Math.random() * 60 - 20); py[i] = ppy + (Math.random() * 30 - 15);
    const kinds = [['beam'], ['steer'], ['seek'], ['ge'], ['walk']][Math.random() * 5 | 0];
    pushEv({ t: 'node', d: { i, parent, px: px[i], py: py[i], score: Math.max(0, (px[i] - 300) / 40) | 0,
      got: 0, dead: Math.random() < 0.08, tlen: i % 200, kind: kinds[0] } });
    if (i && Math.random() < 0.6)
      pushEv({ t: 'commit', d: { parent, node: i, trail: [[ppx, ppy], [px[i], py[i]]] } });
    if (Math.random() < 0.02) pushEv({ t: 'pick', d: { node: i, via: ['elite', 'cell', 'detour'][Math.random() * 3 | 0] } });
    if (Math.random() < 0.01) pushEv({ t: 'max', d: { node: i, rule: 0, slot: 0, val: i } });
    if (Math.random() < 0.008) pushEv({ t: 'find', d: { kind: 'cov', item: 'stmt L' + (20 + (i % 400)) } });
  }
  pushEv({ t: 'done', d: { ms: 4200, expanded: N * 3, states: N, found: null } });
  $('play').disabled = false;
  setPlaying(true);
}

// ---- boot ---------------------------------------------------------------------------
const q = new URLSearchParams(location.search);
setMode(q.get('mode') === 'fly' ? 'fly' : 'ball');
if (q.has('demo')) demo();
else if (q.get('log')) tail(q.get('log'));
else if (q.has('at')) { $('play').disabled = false; }
if (q.has('at')) { const iv = setInterval(() => { if (log.length) { clearInterval(iv); replayTo(Math.min(+q.get('at'), log.length)); drawHud(); } }, 200); }
loop();
$('modebtn').onclick = () => setMode(mode === 'fly' ? 'ball' : 'fly');
$('colby').onchange = () => { colBy = $('colby').value; recolorAll(); drawLegend(); };

// e2e handle — live state for probes
window.__w3d = {
  get mode() { return mode; }, get nodeN() { return st.nodeN; },
  get pos() { return pos; }, get logLen() { return log.length; },
  get cam() { return cam; }, get camInv() { return camInv; },
  get uIso() { return U.uIso.elements.slice(); },
  get uPoin() { return nodeMat.uniforms.uPoin.value; },
  get nodes() { return st.nodes; }, get sel() { return sel; },
  get follow() { return follow; }, get latestI() { return latestI; },
  get levelBBox() { return levelBBox; }, get thumbMode() { return thumbMode; },
  get meta() { return meta; }, get quadVis() { return quads && quads.visible; },
  setMode, replayTo, st, H, camera,
  select(i) { sel = st.nodes[i] || null; updateHl(); showNode(); },
};
