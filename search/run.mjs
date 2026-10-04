// mimo coverage-directed solve on the wasm lane — a faithful JS port of
// dsa host/eval explore.rs's Directed strategy (beam + astar) running the
// instrumented program's wasm instead of the interpreter.
import fs from 'node:fs';
import { performance } from 'node:perf_hooks';

const DIR = process.env.DIR || '/tmp/mimo';
const M = JSON.parse(fs.readFileSync(DIR + '/manifest.json'));
const PLAN = JSON.parse(fs.readFileSync(DIR + '/plan.json'));
const SRC = fs.readFileSync(DIR + '/src_plain.mimas', 'utf8');
const WASM = fs.readFileSync(DIR + '/game.wasm');

const natName = M.natives; // id -> "::print" / "Float::floor" / "Adt(...)::rows" / "cov::hit" ...

// ---- strids ---------------------------------------------------------------
const strs = M.strs.slice();            // strid -> string
const strIds = new Map();               // string -> strid (host allocs fresh)
strs.forEach((s, i) => strIds.set(s, i));
function strid(s) {
  let id = strIds.get(s);
  if (id === undefined) { id = strs.length; strs.push(s); strIds.set(s, id); }
  return id;
}
const strOf = id => strs[Number(id)];
// level geometry from the embedded CSV (strs[0]) — coin centers +
// flag for position-subgoal steering (the game's own `data()` source).
const LROWS = strs[0].trim().split('\n').slice(1).map(l => l.split(','));
const COINS = LROWS.filter(r => r[1] === 'coin')
  .map(r => ({ sid: BigInt(strid(r[0])), x: +r[2] + +r[4] / 2, y: +r[3] + +r[5] / 2 }));
const FLAG = (() => { const r = LROWS.find(r => r[0] === 'flag');
  return { x: +r[2] + +r[4] / 2, y: +r[3] + +r[5] / 2 }; })();
// Movement graph — the same math as the game's own nodes()/can_jump()/
// reach(): horizontal surface spans + jump-arc edges (JUMP_V/GRAV/RUN,
// no fudge). BFS waypoints let seek() descend along jumpable arcs
// instead of straight-line distance into a wall.
const JV = 560, GRAV = 1800, RUNV = 300, PICK = 18, PH = 36;
const HAZ = LROWS.filter(r => r[1] === 'lava' || r[1] === 'goomba')
  .map(r => ({ l: +r[2], t: +r[3], r: +r[2] + +r[4], b: +r[3] + +r[5], k: r[1] }));
const NODES = (() => {
  const ns = [];
  for (const r of LROWS) {
    const [id, kind, x, y, w, h] = r;
    if (kind === 'ground' || kind === 'plat') {
      // split the surface at hazards sitting on it — walking across a
      // lava pool becomes a jump edge between the two halves instead
      let spans = [[+x, +x + +w]];
      for (const hz of HAZ) {
        if (!(hz.b >= +y - 4 && hz.t <= +y)) continue;
        const next = [];
        for (const [l, rr] of spans) {
          if (hz.l - 8 > l) next.push([l, Math.min(rr, hz.l - 8)]);
          if (hz.r + 8 < rr) next.push([Math.max(l, hz.r + 8), rr]);
        }
        spans = next;
      }
      for (const [l, rr] of spans) if (rr - l > 12) ns.push({ top: +y, l, r: rr, id });
    }
  }
  for (const c of COINS) ns.push({ top: c.y, l: c.x - PICK, r: c.x + PICK, id: 'coin', sid: c.sid });
  const g = LROWS.find(r => r[0] === 'flag');
  ns.push({ top: +g[3] + +g[5], l: +g[2] - 20, r: +g[2] + +g[4] + 20, id: 'flag' });
  return ns;
})();
function canJump(a, b) {
  const dy = a.top - b.top;
  if (dy > JV * JV / (2 * GRAV)) return false;
  const disc = JV * JV - 2 * GRAV * dy;
  if (disc < 0) return false;
  const t = (JV + Math.sqrt(disc)) / GRAV;
  const gap = Math.max(b.l - a.r, a.l - b.r, 0);
  return gap + 4 <= RUNV * t;
}
const EDGES = (() => {
  const e = NODES.map(() => []);
  for (let i = 0; i < NODES.length; i++)
    for (let j = 0; j < NODES.length; j++)
      if (i !== j && canJump(NODES[i], NODES[j])) e[i].push(j);
  return e;
})();
// BFS path from the player's surface to the nearest target node —
// returns the node list (full route), or null.
function route(px, fy, targets) {
  // player's node: span containing px, top nearest the feet — falls
  // back to the nearest span rect when midair / off all spans.
  let cur = -1, bd = Infinity;
  for (let i = 0; i < NODES.length; i++) {
    const n = NODES[i];
    const dx = px < n.l ? n.l - px : px > n.r ? px - n.r : 0;
    const d = Math.hypot(dx, n.top - fy);
    if (d < bd) { bd = d; cur = i; }
  }
  const prev = new Int32Array(NODES.length).fill(-2); prev[cur] = -1;
  const q = [cur];
  let hit = -1, hops = Infinity;
  for (let qi = 0; qi < q.length; qi++) {
    const i = q[qi];
    const depth = prev[i] === -1 ? 0 : 1; // hops tracked separately below
    if (targets.has(i)) { hit = i; break; }
    for (const j of EDGES[i]) if (prev[j] === -2) { prev[j] = i; q.push(j); }
  }
  if (hit < 0) return null;
  const path = [];
  for (let i = hit; i >= 0; i = prev[i]) path.unshift(i);
  return path;
}

// ---- cov sink: CovHits from coverage.rs, verbatim ---------------------------
const cov = { points: [], decs: [], dist: new Map(), edges: new Set(), open: [], operands: [], last: -1 };
// point-hit bitmap: u8[__ptmap + point_id] — hit/pass never make records.
// Presence-per-seg == the record semantics (items() only needs n>0); the
// map is cleared each take so a repeated hit re-reports in later segs.
// Edges are order-dependent and never itemized, so bitmap points skip edge().
function drainPoints() {
  const base = X.__ptmap.value, n = PLAN.points.length;
  for (let i = 0; i < n; i++) {
    if (u8[base + i]) bump(cov.points, i);
  }
  u8.fill(0, base, base + n);
}
// decision cells at __dcov + d*136: [leafbits u64][dist_t 8×f64 @8][dist_f
// 8×f64 @72] — wasm helpers min-merge here instead of writing per-cond/cmp
// records. Dist cells must START at +Inf (memory zero-init would corrupt min).
const DCELL = 136;
function initDecs() {
  const base = X.__dcov.value;
  for (let d = 0; d < PLAN.decisions.length; d++) {
    const c = base + d * DCELL;
    for (let k = 0; k < 8; k++) {
      dv.setFloat64(c + 8 + k * 8, Infinity, true);
      dv.setFloat64(c + 72 + k * 8, Infinity, true);
    }
  }
}
function drainDecs() {
  const base = X.__dcov.value;
  for (let d = 0; d < PLAN.decisions.length; d++) {
    const c = base + d * DCELL;
    for (let k = 0; k < 8; k++) {
      const t = dv.getFloat64(c + 8 + k * 8, true);
      const f = dv.getFloat64(c + 72 + k * 8, true);
      if (t < Infinity || f < Infinity) {
        near(d + ',' + k, t, f);
        dv.setFloat64(c + 8 + k * 8, Infinity, true);
        dv.setFloat64(c + 72 + k * 8, Infinity, true);
      }
    }
  }
}
const covTake = () => {
  fresh();
  drainPoints();
  drainDecs();
  covDrain();
  const h = { points: cov.points, decs: cov.decs, dist: cov.dist, edges: cov.edges };
  cov.points = []; cov.decs = []; cov.dist = new Map(); cov.edges = new Set();
  cov.open = []; cov.operands = []; cov.last = -1;
  return h;
};
const bump = (a, i) => { a[i] = (a[i] || 0) + 1; };
const edge = p => { if (cov.last >= 0 && cov.last !== p) cov.edges.add(cov.last + ',' + p); cov.last = p; };
const flag = v => v ? [0, Infinity] : [Infinity, 0];
const nearInto = (map, k, t, f) => {
  const e = map.get(k);
  if (!e) map.set(k, [t, f]); else { e[0] = Math.min(e[0], t); e[1] = Math.min(e[1], f); }
};
const near = (k, t, f) => nearInto(cov.dist, k, t, f);
const leaf = (d, k, v) => {
  for (let i = cov.open.length - 1; i >= 0; i--)
    if (cov.open[i][0] === d) {
      const L = cov.open[i][1];
      while (L.length <= k) L.push(-1);
      L[k] = v ? 1 : 0;
      return;
    }
};
const EPS = 1e-9, pos = x => Math.max(0, x);

// ---- buffered cov sink ----------------------------------------------------
// wasm writes [id|hdr|arg0..5] records (64B) at __covbuf; hdr = nargs | tag<<(8+8i)
// tags: i=0 f=1 b=2 w=3. Replay = the cov::* switch cases, same order.
const _fb = new ArrayBuffer(8), _fd = new DataView(_fb);
const bits2f = w => { _fd.setBigInt64(0, w, true); return _fd.getFloat64(0, true); };
// The only records left: 16B dec-log entries [d u32][v u32][leafbits u64]
// written by wasm-side f_dec — all other cov::* kinds aggregate in-memory
// (ptmap bitmap, decision cells, operand stack) and never make a record.
function covDrain() {
  const n = X.__covp.value;
  if (!n) return;
  fresh();
  const base = X.__covbuf.value;
  if (RECIDS) RECIDS['cov::dec'] = (RECIDS['cov::dec'] || 0) + n;
  for (let i = 0; i < n; i++) {
    const p = base + i * 16;
    const d = dv.getUint32(p, true), v = dv.getUint32(p + 4, true) !== 0;
    // leafbits snapshot: 2 bits/leaf (1=true, 2=false, 0=unreached) —
    // same row format the JS leaf-tracking used: '-1' unreached
    const lb = dv.getBigUint64(p + 8, true);
    const ncond = PLAN.decisions[d].conds.length;
    // interp's leaf vec stops at the highest TOUCHED leaf — trailing
    // unreached leaves are omitted, so the row format must match
    let maxk = -1;
    for (let k = 0; k < ncond; k++)
      if (((lb >> BigInt(k * 2)) & 3n) !== 0n) maxk = k;
    const leaves = [];
    for (let k = 0; k <= maxk; k++) {
      const b2 = Number((lb >> BigInt(k * 2)) & 3n);
      leaves.push(b2 === 1 ? '1' : b2 === 2 ? '0' : '-1');
    }
    let dh = cov.decs[d] || (cov.decs[d] = { t: 0, f: 0, rows: new Set() });
    if (v) dh.t++; else dh.f++;
    dh.rows.add(leaves.join(',') + '|' + (v ? 1 : 0));
  }
  X.__covp.value = 0;
}

function gap(op, a, b) {
  switch (op) {
    case 0: return [Math.abs(a - b), a === b ? EPS : 0];               // Equal
    case 1: return [a === b ? EPS : 0, Math.abs(a - b)];               // NotEqual
    case 2: return [pos(b - a) + (a <= b ? EPS : 0), pos(a - b)];      // Greater
    case 3: return [pos(b - a), pos(a - b) + (a >= b ? EPS : 0)];      // GreaterOrEqual
    case 4: return [pos(a - b) + (a >= b ? EPS : 0), pos(b - a)];      // Less
    case 5: return [pos(a - b), pos(b - a) + (a <= b ? EPS : 0)];      // LessOrEqual
  }
}

// ---- host objects ---------------------------------------------------------
const handles = [null];                // df/expr/synth handles — 0 is Null
const csvRowsCache = new Map();        // df handle -> materialized LRow array word
const csvByStr = new Map();            // csv text -> df handle (level CSV is static)
const newHandle = o => { handles.push(o); return BigInt(handles.length - 1); };
const getHandle = w => handles[Number(w)];
const csvDfs = [];                     // handle idx -> {cols, rows}
function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const cols = lines[0].split(',');
  return { cols, rows: lines.slice(1).map(l => l.split(',')) };
}

// ---- wasm arena materialization ------------------------------------------
let X, u8, dv;
const views = () => { u8 = new Uint8Array(X.memory.buffer); dv = new DataView(X.memory.buffer); };
const fresh = () => { if (dv.buffer !== X.memory.buffer) views(); };
function alloc(n) {
  let hp = X.__hp.value;
  const need = hp + n;
  const memBytes = X.memory.buffer.byteLength;
  if (need > memBytes) {
    X.memory.grow(Math.ceil((need - memBytes) / 65536));
    views();
  }
  X.__hp.value = (hp + n + 7) & ~7;
  return hp;
}
const f64bits = x => { const b = new ArrayBuffer(8); new DataView(b).setFloat64(0, x, true); return new DataView(b).getBigInt64(0, true); };
function writeArray(words) {
  fresh();
  const data = alloc(words.length * 8);
  const h = alloc(16);
  dv.setInt32(h, data, true); dv.setInt32(h + 4, words.length, true); dv.setInt32(h + 8, words.length, true);
  words.forEach((w, i) => dv.setBigInt64(data + i * 8, BigInt(w), true));
  return BigInt(h);
}
function writeInst(adtId, words) {
  fresh();
  const p = alloc(8 + words.length * 8);
  dv.setUint32(p, adtId, true); dv.setUint32(p + 4, 0, true);
  words.forEach((w, i) => dv.setBigInt64(p + 8 + i * 8, BigInt(w), true));
  return BigInt(p);
}
const LROW = M.adts.LRow, INPUT = M.adts.Input;
function lrow(cols, vals) {
  const w = cols.map((c, i) => {
    switch (c) {
      case 'id': case 'kind': return BigInt(strid(vals[i]));
      default: return f64bits(parseFloat(vals[i]));
    }
  });
  return writeInst(LROW.id, w);
}
function input(held, dt) {
  const h = writeArray(held.map(s => BigInt(strid(s))));
  const empty = writeArray([]);
  return writeInst(INPUT.id, [h, f64bits(320), f64bits(240), 0n, f64bits(dt), empty, empty, 0n]);
}

// ---- native dispatch -------------------------------------------------------
const saw = new Set();
const note = (name, args) => { if (!saw.has(name)) { saw.add(name); console.log('  [native]', name, '(' + args.map(a => typeof a === 'bigint' ? a + 'n' : a).join(',') + ')'); } };

function makeImport(id, ps, r) {
  const name = natName[id] || ('n' + id);
  const B = BigInt;
  const castRet = v => {
    if (v === undefined || v === null) v = 0n;
    if (r === 'f') return typeof v === 'bigint' ? Number(v) : v;
    if (r === 'b') return v ? 1 : 0;
    return typeof v === 'bigint' ? v : B(Math.trunc(v));
  };
  // decode w arg strids when a string makes sense
  const wstr = a => strs[Number(a)];
  const _cb = new ArrayBuffer(8), _cd = new DataView(_cb);
  const asF = w => { _cd.setBigInt64(0, BigInt(w), true); return _cd.getFloat64(0, true); };
  const asI = f => { _cd.setFloat64(0, f, true); return _cd.getBigInt64(0, true); };
  // bit-preserving passthrough for generic T -> T natives: the wasm sig
  // may class arg and ret differently, but the value's bits are the value
  const passthru = (v, rc) => {
    if (rc === 'f') return typeof v === 'bigint' ? asF(v) : v;
    if (rc === 'b') return typeof v === 'bigint' ? (v & 1n ? 1 : 0) : (v ? 1 : 0);
    return typeof v === 'bigint' ? v : asI(v);
  };
  return (...args) => {
    const num = i => Number(args[i]);
    // args classed 'f' carrying word payloads arrive bit-reinterpreted
    const argWord = i => ps[i] === 'f' ? asI(args[i]) : BigInt(args[i]);
    switch (name) {
      case 'cov::hit': bump(cov.points, num(0)); edge(num(0)); return castRet(0);
      case 'cov::pass': { const p = num(0); bump(cov.points, p); edge(p); return passthru(args[1], r); }
      case 'cov::begin': cov.open.push([num(0), []]); return castRet(1);
      case 'cov::lhs': case 'cov::rhs': {
        // numeric only when the operand classed i/f — w/b mirror Val non-numbers
        cov.operands.push(ps[0] === 'i' || ps[0] === 'f' ? Number(args[0]) : null);
        return passthru(args[0], r);
      }
      case 'cov::cond': { const [d, k, v] = [num(0), num(1), !!args[2]];
        near(d + ',' + k, ...flag(v)); leaf(d, k, v); return castRet(v); }
      case 'cov::cmp': { const [d, k, op, v] = [num(0), num(1), num(2), !!args[3]];
        const b = cov.operands.pop(), a = cov.operands.pop();
        if (a != null && b != null && op >= 0 && op <= 5) near(d + ',' + k, ...gap(op, a, b));
        else near(d + ',' + k, ...flag(v));
        leaf(d, k, v); return castRet(v); }
      case 'cov::dec': { const d = num(0), v = !!args[1];
        let leaves = [];
        for (let i = cov.open.length - 1; i >= 0; i--)
          if (cov.open[i][0] === d) { leaves = cov.open.splice(i, 1)[0][1]; break; }
        let dh = cov.decs[d] || (cov.decs[d] = { t: 0, f: 0, rows: new Set() });
        if (v) dh.t++; else dh.f++;
        dh.rows.add(leaves.join(',') + '|' + (v ? 1 : 0));
        return castRet(v); }
      case 'Float::floor': return castRet(Math.floor(num(0)));
      case 'Float::abs': return castRet(Math.abs(num(0)));
      case 'Float::max': return castRet(Math.max(num(0), num(1)));
      case 'Float::min': return castRet(Math.min(num(0), num(1)));
      case 'Float::to': return castRet(args[0]);   // units are advisory
      case 'std::polars::from_csv': {
        const t = wstr(argWord(0));
        note('from_csv<' + (t === undefined ? 'STRID?' : t.slice(0, 30).replace(/\n/g, '|')) + '>', args);
        PROF.csv++;
        if (!csvByStr.has(t)) csvByStr.set(t, newHandle(parseCsv(t)));
        return castRet(csvByStr.get(t));
      }
      case 'std::polars::col': return castRet(newHandle({ col: wstr(argWord(0)) }));
      case 'std::polars::lit': return castRet(newHandle({ lit: Number(args[0]) }));
      default: break;
    }
    if (name.endsWith('::rows')) {              // Adt(6)::rows -> [LRow]
      const df = getHandle(argWord(0));
      note('rows rows=' + (df ? df.rows.length : 'NODF'), args);
      PROF.rows++;
      // level CSV is static — the materialized LRow array is identical
      // every call; the bump arena never frees so one instance serves all
      const hh = Number(argWord(0));
      if (!csvRowsCache.has(hh)) csvRowsCache.set(hh, writeArray(df.rows.map(r => lrow(df.cols, r))));
      return castRet(csvRowsCache.get(hh));
    }
    if (name.endsWith('::filter')) return castRet(argWord(0)); // uncalled in practice
    if (name === 'game::input') return castRet(input([], 1 / 60));
    if (name === 'game::events') return castRet(writeArray([]));
    if (name === 'game::quitting') return castRet(0);
    if (name === 'music::synth') return castRet(newHandle({ synth: 1 }));
    if (/^Adt\(AdtId\(40\)/.test(name)) return passthru(args[0], r); // synth builders
    // draw/input/misc stubs
    note(name, args);
    return castRet(0);
  };
}

const env = {};
for (const imp of WebAssembly.Module.imports(new WebAssembly.Module(WASM))) {
  if (imp.kind !== 'function') continue;
  const m = imp.name.match(/^n(\d+)_([ifbw]*)_([ifbwv])$/);
  if (!m) throw new Error('odd import ' + imp.name);
  env[imp.name] = makeImport(Number(m[1]), m[2].split(''), m[3]);
}

// ---- hits utilities (Hits in coverage.rs) -----------------------------------
const emptyHits = () => ({ points: [], decs: [], dist: new Map(), edges: new Set() });
function mergeInto(a, b) {
  for (let i = 0; i < b.points.length; i++) a.points[i] = (a.points[i] || 0) + b.points[i];
  for (let i = 0; i < b.decs.length; i++) {
    const d = b.decs[i]; if (!d) continue;
    const t = a.decs[i] || (a.decs[i] = { t: 0, f: 0, rows: new Set() });
    t.t += d.t; t.f += d.f; for (const r of d.rows) t.rows.add(r);
  }
  for (const [k, v] of b.dist) nearInto(a.dist, k, v[0], v[1]);
  for (const e of b.edges) a.edges.add(e);
}
function items(h) {
  const o = [];
  h.points.forEach((n, i) => { if (n > 0) o.push('p:' + i); });
  h.decs.forEach((d, i) => {
    if (!d) return;
    if (d.t > 0) o.push('o:' + i + ':1');
    if (d.f > 0) o.push('o:' + i + ':0');
    for (const r of d.rows) o.push('r:' + i + ':' + r);
  });
  return o;
}
const norm = x => 1 - Math.pow(1.001, -x);
const BUCKETS = [1, 2, 3, 7, 15, 31, 127, 255];
const bucketOf = c => BUCKETS.find(b => c <= b);
const nextBucket = b => BUCKETS.indexOf(b) >= 0 ? (BUCKETS[BUCKETS.indexOf(b) + 1] || null) : null;
function hdist(h, d, want, conj, leaves) {
  const dh = h.decs[d];
  if (dh && ((want && dh.t > 0) || (!want && dh.f > 0))) return 0;
  const per = []; let any = false;
  for (let k = 0; k < Math.max(leaves, 1); k++) {
    const e = h.dist.get(d + ',' + k);
    per.push(e ? norm(want ? e[0] : e[1]) : null);
    if (e) any = true;
  }
  if (!any) return null;
  const all = (conj === 'and' && want) || (conj === 'or' && !want);
  return all ? per.reduce((s, x) => s + (x === null ? 1 : x), 0)
             : per.reduce((m, x) => x === null ? m : Math.min(m, x), Infinity);
}
// position-subgoal score: the goal leaf's RAW operand gap toward the
// wanted outcome — no `dh.t>0` shortcut (bucket goals need the gradient
// past already-hit segments), no normalization (tiny position deltas
// must differentiate). Null = leaf not evaluated in this segment.
function ogap(h, d, want, conj, leaves) {
  const per = []; let any = false;
  for (let k = 0; k < Math.max(leaves, 1); k++) {
    const e = h.dist.get(d + ',' + k);
    per.push(e ? (want ? e[0] : e[1]) : null);
    if (e) any = true;
  }
  if (!any) return null;
  const all = (conj === 'and' && want) || (conj === 'or' && !want);
  return all ? per.reduce((s, x) => s + (x === null ? Infinity : x), 0)
             : per.reduce((m, x) => x === null ? m : Math.min(m, x), Infinity);
}

// ---- the wasm Game ----------------------------------------------------------
const HOLD = 4, DEPTH = 24, BEAM = 16, BUDGET = Number(process.env.BUDGET || 4000);
const ACTIONS = [[], ['arrowup'], ['arrowdown'], ['arrowleft'], ['arrowright'],
                 ['arrowup', 'arrowleft'], ['arrowup', 'arrowright'], [' ']];

const GAME = { w: 0n, fin: false };
const FT = { inp: 0, tick: 0, draw: 0 };
function frame(held) {
  let t = performance.now();
  const inp = input(held, 1 / 60);
  const evs = writeArray([]);
  FT.inp += performance.now() - t; t = performance.now();
  GAME.w = X.b23(GAME.w, inp, evs);   // tick
  FT.tick += performance.now() - t; t = performance.now();
  const hpDraw = X.__hp.value;
  X.b25(GAME.w);                      // draw — its ~35KB of allocs are
  X.__hp.value = hpDraw;              // transient; rewind keeps snaps small
  FT.draw += performance.now() - t;
}
// snapshot three ranges: [0, __sp) live stack, [covBase, sink_base) the
// ptmap/cov region, [heap_base, __hp) the heap — the dead stack tail
// (~1MB) and the transient 8MB sink are skipped
function snapshot() {
  fresh();
  const sp = X.__sp.value, hp = X.__hp.value;
  const cb = stackBase + (1 << 20);
  const sb = X.__covbuf.value, he = sb + (8 << 20); // skip the transient sink
  const a = u8.slice(0, sp), b = u8.slice(cb, sb), c = u8.slice(he, hp);
  const img = new Uint8Array(a.length + b.length + c.length);
  img.set(a); img.set(b, a.length); img.set(c, a.length + b.length);
  return { img, sp, hp, w: GAME.w, fin: GAME.fin, cb, sb, he };
}
function restore(s) {
  views();
  const mid = s.sp + (s.sb - s.cb);
  u8.set(s.img.subarray(0, s.sp), 0);
  u8.set(s.img.subarray(s.sp, mid), s.cb);
  u8.set(s.img.subarray(mid), s.he);
  if (X.__hp.value > s.hp) u8.fill(0, s.hp, X.__hp.value);
  X.__sp.value = s.sp;
  X.__hp.value = s.hp;
  csvRowsCache.clear(); // cached array words may point past the rewound __hp
  GAME.w = s.w; GAME.fin = false;
}
// Semantic state sig (like GameSnap::sig — live values, not raw bytes):
// the W instance's 11 fields + the `got` array's element words. Dead
// bump-allocated objects in the arena never reach the hash, so two paths
// to the same play state merge regardless of allocation history.
function sig(s) {
  let h = 0xcbf29ce484222325n; const F = 0x100000001b3n, M64 = 0xffffffffffffffffn;
  const mix = v => { h ^= v & M64; h = (h * F) & M64; };
  fresh();
  const w = Number(s.w);
  for (let i = 0; i < 11; i++) mix(dv.getBigInt64(w + 8 + i * 8, true));
  const got = Number(dv.getBigInt64(w + 8 + 6 * 8, true)); // field 6 = `got`
  if (got) {
    const dp = dv.getUint32(got, true), len = dv.getUint32(got + 4, true);
    for (let i = 0; i < len; i++) mix(dv.getBigInt64(dp + i * 8, true));
  }
  mix(BigInt(s.fin ? 1 : 0));
  return h;
}
const seenKey = (s, a) => sig(s) ^ ((BigInt(a) + 1n) * 0x9E3779B97F4A7C15n);

// ---- Search (explore.rs directed) -------------------------------------------
class Search {
  constructor() {
    this.archive = []; this.seen = new Set(); this.expandedPairs = new Set(); this.covered = new Set();
    this.banMap = new Map();
    this.total = emptyHits(); this.witnesses = []; this.expanded = 0;
  }
  done() { return this.hit(); }
  hit() {
    const l = M.target_line;
    return PLAN.points.some((p, i) => p.line === l && this.covered.has('p:' + i));
  }
  record(seg, tape, error) {
    for (const it of items(seg))
      if (!this.covered.has(it)) {
        this.covered.add(it);
        this.witnesses.push({ item: it, tape: tape.slice() });
      }
    mergeInto(this.total, seg);
    return error != null;
  }
  expand(parent, a) {
    if (this.expanded % 200 === 0) console.error('expanded=' + this.expanded + ' archive=' + this.archive.length + ' covered=' + this.covered.size + ' hp=' + X.__hp.value);
    this.expanded++;
    const p = this.archive[parent];
    if (p.dead) return null;
    const pair = parent * 8 + a;
    if (this.expandedPairs.has(pair)) return null;
    this.expandedPairs.add(pair);
    let t = performance.now();
    restore(p.snap);
    PROF.restore += performance.now() - t; t = performance.now();
    const tape = p.tape.slice(); tape.push(a);
    PROF.frames += performance.now() - t; t = performance.now();
    covTake();
    PROF.cov += performance.now() - t; t = performance.now();
    let error = null;
    for (let f = 0; f < HOLD && !GAME.fin; f++) {
      try { frame(ACTIONS[a]); }
      catch (e) { error = String(e); GAME.fin = true; break; }
    }
    PROF.frames += performance.now() - t; t = performance.now();
    const seg = covTake();
    PROF.cov += performance.now() - t; t = performance.now();
    if (this.record(seg, tape, error)) return null;
    const snap = snapshot();
    PROF.snap += performance.now() - t; t = performance.now();
    const key = seenKey(snap, a);
    if (this.seen.has(key)) return null;
    this.seen.add(key);
    const dead = GAME.fin;
    const score = Number(dv.getBigInt64(Number(snap.w) + 8 + 7 * 8, true));
    const touched = new Set();
    for (const k of seg.dist.keys()) touched.add(Number(k.split(',')[0]));
    const mem = { points: p.mem.points.slice(), decs: p.mem.decs.map(d => d && { t: d.t, f: d.f, rows: new Set(d.rows) }),
                  dist: new Map(p.mem.dist), edges: new Set(p.mem.edges) };
    for (const k of [...mem.dist.keys()])
      if (touched.has(Number(k.split(',')[0]))) mem.dist.delete(k);
    for (const [k, v] of seg.dist) nearInto(mem.dist, k, v[0], v[1]);
    seg.points.forEach((n, i) => { mem.points[i] = (mem.points[i] || 0) + n; });
    seg.decs.forEach((dd, i) => {
      if (!dd) return;
      if (!mem.decs[i]) mem.decs[i] = { t: 0, f: 0, rows: new Set() };
      mem.decs[i].t += dd.t; mem.decs[i].f += dd.f;
      for (const r of dd.rows) mem.decs[i].rows.add(r);
    });
    for (const e of seg.edges) mem.edges.add(e);
    this.buckets(mem, tape);
    this.archive.push({ snap, tape, seg, mem, dead, score });
    return this.archive.length - 1;
  }
  newItems(seg, before) { let n = 0; for (const it of items(seg)) if (!before.has(it)) n++; return n; }
  beam() {
    let frontier = [0];
    for (let d = 0; d < DEPTH; d++) {
      const before = new Set(this.covered);
      const kids = [];
      for (const n of frontier)
        for (let a = 0; a < ACTIONS.length; a++) {
          if (this.expanded >= BUDGET / 2) break;
          const c = this.expand(n, a);
          if (c !== null) kids.push(c);
        }
      if (!kids.length || this.done()) break;
      kids.sort((x, y) => (this.newItems(this.archive[y].seg, before) - this.newItems(this.archive[x].seg, before)) || (x - y));
      frontier = kids.slice(0, Math.max(BEAM, 1));
    }
  }
  guard_of(line) {
    const lines = SRC.split(/(?<=\n)/);
    let lstart = 0;
    for (let i = 0; i < line - 1 && i < lines.length; i++) lstart += lines[i].length;
    const rest = SRC.slice(Math.min(lstart, SRC.length));
    const nl = rest.indexOf('\n');
    const lend = lstart + (nl < 0 ? rest.length : nl);
    let best = null;
    PLAN.decisions.forEach((dec, d) => {
      for (const [span, want] of [[dec.then_span, true], [dec.else_span, false]]) {
        if (!span) continue;
        const [a, b] = span;
        if (a <= Math.max(lstart, a) && a <= lend && b >= lend && b - a < (best ? best[2] : Infinity) && a < lend)
          best = [d, want, b - a];
      }
    });
    return best ? [best[0], best[1]] : null;
  }
  approach_in(seg, d, want) {
    let level = 0;
    for (;;) {
      const dec = PLAN.decisions[d];
      const x = hdist(seg, d, want, dec.conj, dec.conds.length);
      if (x !== null) return level + x;
      const par = dec.parent;
      if (par && PLAN.decisions[par[0]].conds.length)
        { d = par[0]; want = par[1]; level += 1 + dec.conds.length; }
      else return level + 2 + dec.conds.length;
    }
  }
  key(n, d, want) {
    const dec = PLAN.decisions[d], node = this.archive[n];
    const appr = this.approach_in(node.seg, d, want);
    let prim = hdist(node.seg, d, want, dec.conj, dec.conds.length);
    if (prim === null) prim = hdist(node.mem, d, want, dec.conj, dec.conds.length);
    const dd = node.mem.decs[d];
    const hits = dd ? (want ? dd.t : dd.f) : 0;
    return [prim === null ? 1e3 + appr : prim, appr - Math.min(hits, 8) * 0.01, node.tape.length, n];
  }
  astar(d, want, budget, goal) {
    const stop = this.expanded + budget;
    goal = goal || 'o:' + d + ':' + (want ? 1 : 0);
    const heap = [];   // min-heap on (dist, approach, tapeLen, node)
    const cmp = (a, b) => (a[0] - b[0]) || (a[1] - b[1]) || (a[2] - b[2]) || (a[3] - b[3]);
    const push = x => { heap.push(x); let i = heap.length - 1;
      while (i > 0) { const p = (i - 1) >> 1; if (cmp(heap[i], heap[p]) < 0) { [heap[i], heap[p]] = [heap[p], heap[i]]; i = p; } else break; } };
    const pop = () => { const top = heap[0], last = heap.pop();
      if (heap.length) { heap[0] = last; let i = 0;
        for (;;) { let l = 2 * i + 1, r = l + 1, m = i;
          if (l < heap.length && cmp(heap[l], heap[m]) < 0) m = l;
          if (r < heap.length && cmp(heap[r], heap[m]) < 0) m = r;
          if (m === i) break; [heap[i], heap[m]] = [heap[m], heap[i]]; i = m; } }
      return top; };
    for (let n = 0; n < this.archive.length; n++)
      if (!this.archive[n].dead) push([...this.key(n, d, want)]);
    while (heap.length) {
      const e = pop(), n = e[3];
      if (this.covered.has(goal) || this.expanded >= stop || this.done()) return;
      for (let a = 0; a < ACTIONS.length; a++) {
        const c = this.expand(n, a);
        if (c !== null) push(this.key(c, d, want));
      }
    }
  }
  // multi-goal aiming: the finish guard plus the nearest *reached but
  // uncovered* decision outcomes — coverage-directed exploration's
  // secondary targets. Budget round-robins in slices: flipping a
  // side-quest dec (coin pickup) unlocks gradient on the finish gate.
  secondary() {
    const goals = [];
    for (let d = 0; d < PLAN.decisions.length; d++) {
      for (const w of [1, 0]) {
        let goal = 'o:' + d + ':' + w;
        if (this.covered.has(goal)) {
          let c = 0;
          for (const n of this.archive) {
            const dd = n.mem.decs[d];
            const cc = dd ? (w ? dd.t : dd.f) : 0;
            if (cc > c) c = cc;
          }
          const nb = c > 0 && c <= 255 ? nextBucket(bucketOf(c)) : null;
          if (!nb || nb > 3 || this.covered.has(goal + ':' + nb)) continue;
          goal = goal + ':' + nb;
        } else {
          const touched = this.total.decs[d] ||
            [...this.total.dist.keys()].some(k => Number(k.split(',')[0]) === d);
          if (!touched) continue;
        }
        let best = Infinity;
        for (let n = 0; n < this.archive.length; n++) {
          if (this.archive[n].dead) continue;
          const k = this.key(n, d, w);
          const s = k[0] * 1e6 + k[1] * 64 + k[2];
          if (s < best) best = s;
        }
        goals.push([best, d, w, goal]);
      }
    }
    goals.sort((a, b) => a[0] - b[0]);
    return goals.slice(0, 6).map(([, d, w, g]) => [d, w, g]);
  }
  buckets(mem, tape) {
    mem.decs.forEach((dd, i) => {
      if (!dd) return;
      for (const [w, c] of [[1, dd.t], [0, dd.f]]) {
        if (!c) continue;
        for (const b of BUCKETS) {
          if (b > bucketOf(c)) break;
          const it = 'o:' + i + ':' + w + ':' + b;
          if (!this.covered.has(it)) {
            this.covered.add(it);
            this.witnesses.push({ item: it, tape: tape.slice() });
          }
        }
      }
    });
  }
  bestNode(d, w) {
    let bi = -1, bs = Infinity;
    for (let n = 0; n < this.archive.length; n++) {
      if (this.archive[n].dead) continue;
      const k = this.key(n, d, w), s = k[0] * 1e6 + k[1] * 64 + k[2];
      if (s < bs) { bs = s; bi = n; }
    }
    return bi;
  }
  // CmpLog operand→action: sim every action HOLD frames on a scratch
  // copy, commit the argmin on the goal leaf's live distance on a
  // working snapshot (not via expand — dedup may block node creation
  // while the position is still novel work). Sims' coverage records
  // for free; the final state banks into the archive when its sig is new.
  steer(n, d, w, goal, maxSteps = 40) {
    goal = goal || 'o:' + d + ':' + (w ? 1 : 0);
    const dec = PLAN.decisions[d];
    const p0 = this.archive[n];
    if (p0.dead) return;
    let wsnap = p0.snap;
    const wtape = p0.tape.slice();
    const wmem = { points: p0.mem.points.slice(),
      decs: p0.mem.decs.map(dd => dd && { t: dd.t, f: dd.f, rows: new Set(dd.rows) }),
      dist: new Map(p0.mem.dist), edges: new Set(p0.mem.edges) };
    let stall = 0, lastSc = null, lastSeg = null, steps = 0;
    for (let s = 0; s < maxSteps; s++) {
      if (this.covered.has(goal) || this.done() || this.expanded >= BUDGET) break;
      const scored = [];
      for (let a = 0; a < ACTIONS.length; a++) {
        const t = performance.now();
        restore(wsnap);
        let err = null;
        for (let f = 0; f < HOLD && !GAME.fin; f++) {
          try { frame(ACTIONS[a]); } catch (e) { err = String(e); GAME.fin = true; break; }
        }
        const seg = covTake();
        PROF.steer += performance.now() - t; PROF.steerSims++;
        this.record(seg, wtape.concat([a]), err);
        const sc = ogap(seg, d, w, dec.conj, dec.conds.length);
        scored.push([sc === null ? 1e6 : sc, sc, a]);
      }
      scored.sort((x, y) => x[0] - y[0]);
      const [rk, sc, a] = scored[0];
      if (process.env.STEER_DBG) console.error('  steer ' + goal + ' n=' + n + ' step' + s + ' lastSc=' + lastSc + ' scores=' + JSON.stringify(scored.map(x => [x[0] === 1e6 ? null : +x[0].toFixed(3), x[2]])));
      if (sc === null) break;
      this.expanded++;
      restore(wsnap);
      for (let f = 0; f < HOLD && !GAME.fin; f++) {
        try { frame(ACTIONS[a]); } catch (e) { GAME.fin = true; break; }
      }
      lastSeg = covTake();
      wtape.push(a); steps++;
      mergeInto(wmem, lastSeg);
      this.buckets(wmem, wtape);
      wsnap = snapshot();
      stall = (lastSc !== null && sc >= lastSc - 1e-9) ? stall + 1 : 0;
      lastSc = sc;
      if (GAME.fin || stall >= 2) break;
      if (this.covered.has(goal)) { PROF.steerOk++; break; }
    }
    if (steps && lastSeg) {
      const key = seenKey(wsnap, wtape[wtape.length - 1]);
      if (!this.seen.has(key)) {
        this.seen.add(key);
        this.archive.push({ snap: wsnap, tape: wtape.slice(), seg: lastSeg, mem: this.cloneMem(wmem), dead: GAME.fin, score: Number(dv.getBigInt64(Number(wsnap.w) + 8 + 7 * 8, true)) });
      }
    }
  }
  cloneMem(m) {
    return { points: m.points.slice(),
      decs: m.decs.map(dd => dd && { t: dd.t, f: dd.f, rows: new Set(dd.rows) }),
      dist: new Map(m.dist), edges: new Set(m.edges) };
  }
  bank(wsnap, wtape, wmem, lastSeg) {
    const key = seenKey(wsnap, wtape[wtape.length - 1]);
    if (this.seen.has(key)) return;
    this.seen.add(key);
    const score = Number(dv.getBigInt64(Number(wsnap.w) + 8 + 7 * 8, true));
    this.archive.push({ snap: wsnap, tape: wtape.slice(), seg: lastSeg, mem: this.cloneMem(wmem), dead: GAME.fin, score });
  }
  // AFL havoc: coverage-blind diversification — commit random actions
  // (repeat-biased, Go-Explore's p≈0.6) on a working snapshot; every
  // segment records coverage, the endpoint banks if its sig is new.
  // actions = forced tape suffix for splice, else random.
  walk(n, k, forced) {
    const p0 = this.archive[n];
    if (!p0 || p0.dead) return;
    let wsnap = p0.snap;
    const wtape = p0.tape.slice();
    const wmem = this.cloneMem(p0.mem);
    let a = wtape.length ? wtape[wtape.length - 1] : 0;
    let lastSeg = null, steps = 0;
    for (let s = 0; s < k; s++) {
      if (this.done() || this.expanded >= BUDGET) break;
      a = forced ? forced[s] : (Math.random() < 0.6 ? a : (Math.random() * ACTIONS.length) | 0);
      if (a === undefined) break;
      this.expanded++;
      restore(wsnap);
      let err = null;
      for (let f = 0; f < HOLD && !GAME.fin; f++) {
        try { frame(ACTIONS[a]); } catch (e) { err = String(e); GAME.fin = true; break; }
      }
      lastSeg = covTake();
      wtape.push(a);
      this.record(lastSeg, wtape, err);
      mergeInto(wmem, lastSeg);
      this.buckets(wmem, wtape);
      wsnap = snapshot(); steps++;
      if (GAME.fin) break;
    }
    if (steps && lastSeg) this.bank(wsnap, wtape, wmem, lastSeg);
  }
  // AFL splice: replay tape B's suffix from node A's state — route
  // fragments recombine (reach-coin-A prefix + near-coin-B tail).
  splice(nA, nB, k = 8) {
    const pb = this.archive[nB];
    if (!pb || !pb.tape.length) return;
    const i = (Math.random() * pb.tape.length) | 0;
    this.walk(nA, k, pb.tape.slice(i, i + k));
  }
  // w.got decode — [dp:u32][len:u32] header, len i64 strids — same
  // layout sig() hashes, read on live memory right after restore.
  gotIds() {
    const out = new Set();
    const w = Number(GAME.w);
    const got = Number(dv.getBigInt64(w + 8 + 6 * 8, true));
    if (!got) return out;
    const dp = dv.getUint32(got, true), len = dv.getUint32(got + 4, true);
    for (let i = 0; i < len; i++) out.add(dv.getBigInt64(dp + i * 8, true));
    return out;
  }
  // Top-k archive nodes by coins collected — seek's starting lines.
  // Later archive index wins ties: banked mid-route states carry the
  // same score as their ancestors but sit deeper along the route.
  topScore(k = 3) {
    const cand = [];
    for (let n = this.archive.length - 1; n >= 0; n--)
      if (!this.archive[n].dead) cand.push([this.archive[n].score || 0, n]);
    cand.sort((a, b) => b[0] - a[0]);   // stable: newest node wins a tie
    return cand.slice(0, k).map(x => x[1]);
  }
  // Position subgoal: descend toward the nearest uncollected coin (or
  // the flag once the board is clean). Needs no leaf — score is the
  // live (px,py) distance to the target, recomputed each step as `got`
  // updates. This is how a route gets planned where dec leaves are flat.
  seek(n, maxSteps = 150) {
    const p0 = this.archive[n];
    if (!p0 || p0.dead) return;
    let wsnap = p0.snap;
    const wtape = p0.tape.slice();
    const wmem = this.cloneMem(p0.mem);
    restore(wsnap); fresh();
    const deaths0 = dv.getBigInt64(Number(GAME.w) + 8 + 8 * 8, true);
    let lastSeg = null, steps = 0, stall = 0, lastSc = null, posStuck = 0;
    let lastPx = NaN, lastPy = NaN, gotSz = 0;
    // route ends that stalled from THIS start node — persists across
    // seeks so a permanently-stuck target isn't retried every loop
    const ban = this.banMap.get(n) || new Set();
    this.banMap.set(n, ban);
    for (let s = 0; s < maxSteps; s++) {
      if (this.done() || this.expanded >= BUDGET) break;
      restore(wsnap);
      const got = this.gotIds();
      if (got.size > gotSz) gotSz = got.size;
      const w0 = Number(GAME.w);
      if (process.env.SEEK_DBG && s === 0)
        console.error('  seek n=' + n + ' got=' + JSON.stringify([...got].map(String)) + ' score=' + dv.getFloat64(w0 + 8 + 7 * 8, true));
      const px0 = dv.getFloat64(w0 + 8, true), py0 = dv.getFloat64(w0 + 16, true);
      if (dv.getBigInt64(w0 + 8 + 8 * 8, true) !== deaths0) break; // died en route — respawn reset the route
      // BFS waypoint: next node on the jumpable route to the nearest
      // uncollected coin (or the flag once the board is clean).
      const targets = new Set();
      NODES.forEach((n, i) => { if (n.id === 'coin' && !got.has(n.sid) && !ban.has(i)) targets.add(i); });
      if (!targets.size) NODES.forEach((n, i) => { if (n.id === 'flag') targets.add(i); });
      const path = route(px0, py0 + PH, targets);
      if (!path) break;
      if (process.env.SEEK_TGT && s === 0)
        console.error('  tgt n=' + n + ' got=' + got.size + ' path=' + path.map(i => NODES[i].id + (NODES[i].sid !== undefined ? ':' + strOf(NODES[i].sid) : '')).join('>'));
      // waypoint = first path node whose span doesn't contain the
      // player — standing on path[1] advances the route, not the score
      let wi = path.length - 1;
      for (let i = 0; i < path.length; i++) {
        const n = NODES[path[i]];
        if (!(px0 >= n.l - 2 && px0 <= n.r + 2 && Math.abs(py0 + PH - n.top) < 30)) { wi = i; break; }
      }
      const wp = NODES[path[wi]];
      if (stall >= 3) {
        // the route end stalled — ban that target, reroute to the next
        ban.add(path[path.length - 1]);
        stall = 0; lastSc = null;
        continue;
      }
      const tx = Math.min(Math.max(px0, wp.l), wp.r), ty = wp.top - PH;
      // candidates: single actions (12-frame horizon) + scripted hop
      // macros for the route edge — greedy 4-frame commits can't
      // discover a jump arc over a gap/hazard, so the arc is a unit
      const dirA = tx > px0 + 4 ? 4 : tx < px0 - 4 ? 3 : 0;
      const dupA = dirA === 4 ? 6 : dirA === 3 ? 5 : 1;
      const cand = ACTIONS.map((_, i) => [i]);
      if (wi > 0 || Math.abs(tx - px0) > RUNV * HOLD / 60 + 8 || ty < py0 - 8) {
        const dyh = (py0 + PH) - wp.top;
        const dh = JV * JV - 2 * GRAV * dyh;
        const tFly = dh > 0 ? (JV + Math.sqrt(dh)) / GRAV : JV / GRAV;
        const jf = Math.max(1, Math.ceil(tFly * 60 / HOLD));
        cand.push([dirA, ...Array(jf).fill(dupA), dirA]);
        cand.push([...Array(jf).fill(dupA), dirA]);
        cand.push([dirA, dirA, ...Array(jf).fill(dupA), dirA]);
      }
      const scored = [];
      for (const seq of cand) {
        restore(wsnap);
        let err = null, best = Infinity;
        for (const ai of seq) {
          for (let f = 0; f < HOLD && !GAME.fin; f++) {
            try { frame(ACTIONS[ai]); } catch (e) { err = String(e); GAME.fin = true; break; }
            fresh();
            const w1 = Number(GAME.w);
            best = Math.min(best, Math.hypot(dv.getFloat64(w1 + 8, true) - tx,
                                             dv.getFloat64(w1 + 16, true) - ty));
          }
          if (GAME.fin) break;
        }
        const seg = covTake();
        this.record(seg, wtape.concat(seq), err);
        scored.push([best, seq]);
      }
      scored.sort((x, y) => x[0] - y[0]);
      const [sc, seq] = scored[0];
      if (process.env.SEEK_DBG) {
        fresh();
        const wp1 = Number(GAME.w);
        console.error('  seek n=' + n + ' step' + s + ' pos=(' + dv.getFloat64(wp1 + 8, true).toFixed(0) + ',' + dv.getFloat64(wp1 + 16, true).toFixed(0) + ') wp' + wi + '/' + path.length + '=' + wp.id + ' target=(' + tx + ',' + ty + ') sc=' + sc.toFixed(1) + (GAME.fin ? ' FIN' : ''));
      }
      this.expanded++;
      restore(wsnap);
      for (const ai of seq) {
        for (let f = 0; f < HOLD && !GAME.fin; f++) {
          try { frame(ACTIONS[ai]); } catch (e) { GAME.fin = true; break; }
        }
        if (GAME.fin) break;
      }
      lastSeg = covTake();
      wtape.push(...seq); steps++;
      this.record(lastSeg, wtape, null);
      mergeInto(wmem, lastSeg);
      this.buckets(wmem, wtape);
      wsnap = snapshot();
      // bank on pickups or every 8th step: mid-route states survive
      // deaths — full-density banking OOM'd the archive past ~11k snaps
      if (this.gotIds().size > gotSz || steps % 8 === 0) this.bank(wsnap, wtape, wmem, lastSeg);
      fresh();
      const pw = Number(GAME.w);
      const pxc = dv.getFloat64(pw + 8, true), pyc = dv.getFloat64(pw + 16, true);
      posStuck = (Math.abs(pxc - lastPx) < 1 && Math.abs(pyc - lastPy) < 1) ? posStuck + 1 : 0;
      lastPx = pxc; lastPy = pyc;
      stall = (lastSc !== null && sc >= lastSc - 1e-9) ? stall + 1 : 0;
      lastSc = sc;
      if (GAME.fin || posStuck >= 4) break;
    }
    if (steps && lastSeg) this.bank(wsnap, wtape, wmem, lastSeg);
  }
  // One diversification round: havoc/splice from random archive nodes.
  // Runs where there's no gradient — the fuzzer layer for flat regions.
  mutate(k = 24) {
    const live = [];
    for (let n = 0; n < this.archive.length; n++) if (!this.archive[n].dead) live.push(n);
    if (!live.length) return;
    for (let r = 0; r < k && !this.done() && this.expanded < BUDGET; r++) {
      const n = live[(Math.random() * live.length) | 0];
      if (Math.random() < 0.5) this.walk(n, 8);
      else this.splice(n, live[(Math.random() * live.length) | 0]);
    }
  }
  aim() {
    const g = this.guard_of(M.target_line);
    const primary = g ? [g] : [];
    while (this.expanded < BUDGET && !this.done()) {
      const goals = [...primary, ...this.secondary()];
      if (!goals.length) break;
      const slice = Math.max(Math.floor((BUDGET - this.expanded) / 8 / goals.length), 40);
      for (const [d, w, goal] of goals) {
        if (!this.covered.has(goal)) {
          const bn = this.bestNode(d, w);
          if (bn >= 0) this.steer(bn, d, w, goal);
        }
        this.astar(d, w, slice, goal);
        if (this.done() || this.expanded >= BUDGET) break;
      }
      this.mutate(goals.length * 8);
      // position-subgoal passes: nearest-uncollected-coin descent from
      // a couple of frontier nodes — the route planner for flat regions
      const tops = this.topScore(2);
      // diversity slot: a random progressed node — routes the leader
      // never visits stay reachable instead of starving off-frontier
      const live = [];
      for (let n = 0; n < this.archive.length; n++)
        if (!this.archive[n].dead && (this.archive[n].score || 0) > 2) live.push(n);
      if (live.length) tops.push(live[(Math.random() * live.length) | 0]);
      for (const n of tops) {
        if (this.done() || this.expanded >= BUDGET) break;
        this.seek(n);
      }
    }
  }
}

// ---- run --------------------------------------------------------------------
const RECIDS = process.env.RECIDS ? {} : null;
const PROF = { restore: 0, frames: 0, snap: 0, cov: 0, misc: 0, rows: 0, csv: 0, steer: 0, steerOk: 0, steerSims: 0 };
const t0 = performance.now();
const { instance } = await WebAssembly.instantiate(WASM, { env });
X = instance.exports; views();
initDecs();                           // dist cells must start at +Inf
const stackBase = X.__sp.value;    // == nbodies*8, the pc_table end

GAME.w = X.b22();                      // init()
frame([]);                             // one frame of default input, like Game::compile's first yield
const s0 = snapshot();
const s = new Search();
s.seen.add(sig(s0));
s.archive.push({ snap: s0, tape: [], seg: emptyHits(), mem: emptyHits(), dead: false });
s.beam();
if (!s.done()) s.aim();
const ms = performance.now() - t0;

const found = s.witnesses.find(w => w.item.startsWith('p:') &&
  PLAN.points[Number(w.item.slice(2))].line === M.target_line);
const maxScore = s.archive.reduce((m, n) => Math.max(m, n.score || 0), 0);
console.log(JSON.stringify({
  ms: Math.round(ms), expanded: s.expanded, states: s.archive.length,
  covered: s.covered.size, maxScore,
  found: found ? found.tape.map(a => ACTIONS[a].join('+') || 'none') : null,
  witnesses: s.witnesses.length,
  prof: PROF, ft: FT, recids: RECIDS,
}, null, 1));
