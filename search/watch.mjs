// mimo solver watch — replays the event log streamed from worker.mjs.
// The solver runs full-speed in the worker; here we buffer every event and
// scrub/play through the log, so pause and rewind are free.
const cv = document.getElementById('cv');
const g = cv.getContext('2d');
const $ = id => document.getElementById(id);

let worker = null;
const log = [];            // [{t, d}]
let pos = 0;               // replay head
let playing = false;
let doneInfo = null;
let meta = null;
let levelMeta = null;      // survives replayTo — rust-fuzzer mode's level

// replayed state
const KIND_COLOR = { seed: '#888', beam: '#6ea8ff', steer: '#5fd4d0',
  seek: '#7ed67e', walk: '#b48ef2', splice: '#f2a65e', bank: '#9aa4b8',
  ge: '#e8d44d',
  // Rust fuzzer FindKind names (node.kinds[0])
  Cov: '#6ea8ff', Bucket: '#5fd4d0', Max: '#ffd75e', Cell: '#7ed67e',
  State: '#b48ef2', Fault: '#ff6f6f', Edge: '#f2a65e', Dist: '#f2a65e',
  absorb: '#555c6e' };
const ROW_COLOR = { ground: '#3d4351', plat: '#4a5163', lava: '#b04040',
  goomba: '#b07040', coin: '#d8c052', flag: '#4fae62', spawn: '#4fae62', goal: '#4fae62' };
let st = freshState();
function freshState() {
  return { nodes: [], trails: [], ghosts: null, phase: '—', goal: null,
    pick: null, elites: new Map(), findList: [], covered: 0,
    expanded: 0, maxScore: 0, alive: null, cells: 0 };
}
const TRAIL_KEEP = 40;

// ---- world → canvas transform ---------------------------------------------
let view = { x: 0, y: 0, s: 1 };
function fit() {
  if (!meta) return;
  let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
  for (const r of meta.rows) {
    x0 = Math.min(x0, +r[2]); y0 = Math.min(y0, +r[3]);
    x1 = Math.max(x1, +r[2] + +r[4]); y1 = Math.max(y1, +r[3] + +r[5]);
  }
  const pad = 30;
  const w = cv.width, h = cv.height;
  const s = Math.min((w - 2 * pad) / (x1 - x0), (h - 2 * pad) / (y1 - y0));
  view = { x: x0 - pad / s, y: y0 - pad / s, s };
}
const X = wx => (wx - view.x) * view.s;
const Y = wy => (wy - view.y) * view.s;

// ---- event application -----------------------------------------------------
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

function apply(ev) {
  // wasm events arrive {t, d:{…}} from the worker; the Rust
  // FUZZ_WATCH JSONL is flat — {t, i, px, …} with no wrapper
  let { t, d } = ev;
  if (d === undefined) { d = { ...ev }; delete d.t; }
  if (t === 'meta') { meta = levelMeta = d; fit(); return; }
  if (t === 'phase') { st.phase = d.name + (d.goal !== undefined ? ' → ' + decode(d.goal) : ''); return; }
  if (t === 'goal') { st.goal = d; return; }
  // Rust fuzzer: scheduler decision — which corpus entry, which queue.
  if (t === 'pick') { st.pick = d; st.phase = 'fuzz'; return; }
  if (t === 'node') {
    st.nodes[d.i] = d;
    st.expanded++;
    st.maxScore = Math.max(st.maxScore, d.score || 0);
    return;
  }
  if (t === 'commit') {
    // driver: {kind, node, seq, trail} · rust: {parent, ok, trail, fault}
    st.trails.push({ kind: d.kind ?? (d.ok ? 'commit' : 'reject'),
      node: d.node ?? d.parent, trail: d.trail, ok: d.ok !== false });
    if (st.trails.length > TRAIL_KEEP) st.trails.shift();
    st.alive = d.trail && d.trail.length ? d.trail[d.trail.length - 1] : null;
    if (d.cells !== undefined) st.cells = d.cells;
    return;
  }
  if (t === 'sims') { st.ghosts = d; return; }
  if (t === 'find') { st.findList.push(d.kind ? `${d.kind} ${d.item}` : d.item); st.covered++; return; }
  // Rust fuzzer: a maxmap slot claimed a new elite.
  if (t === 'max') { st.elites.set(d.node, d); return; }
  if (t === 'done') { doneInfo = d; st.phase = 'done'; return; }
  if (t === 'error') { $('err').textContent = d; return; }
}

function replayTo(n) {
  st = freshState(); meta = levelMeta; doneInfo = null;
  for (let i = 0; i < n; i++) apply(log[i]);
  pos = n;
  fit();
  render();
}

// ---- render ----------------------------------------------------------------
function trail(tr, color, lw) {
  if (!tr || tr.length < 2) return;
  g.strokeStyle = color; g.lineWidth = lw; g.beginPath();
  g.moveTo(X(tr[0][0]), Y(tr[0][1]));
  for (let i = 1; i < tr.length; i++) g.lineTo(X(tr[i][0]), Y(tr[i][1]));
  g.stroke();
}

function render() {
  cv.width = cv.clientWidth * devicePixelRatio;
  cv.height = cv.clientHeight * devicePixelRatio;
  g.fillStyle = '#14161c'; g.fillRect(0, 0, cv.width, cv.height);
  if (!meta) return;
  g.lineWidth = 1;

  // level geometry
  for (const r of meta.rows) {
    const kind = r[1];
    if (!ROW_COLOR[kind]) continue;
    g.fillStyle = ROW_COLOR[kind];
    const x = X(+r[2]), y = Y(+r[3]), w = +r[4] * view.s, h = +r[5] * view.s;
    if (kind === 'coin') { g.beginPath(); g.arc(X(+r[2] + +r[4] / 2), Y(+r[3] + +r[5] / 2), 4, 0, 7); g.fill(); }
    else g.fillRect(x, y, w, Math.max(h, 2));
  }

  // committed rollout trails, fading; rejected evals (rust fuzzer) draw
  // faint red — the AFL analog of the wasm driver's ghost candidates
  st.trails.forEach((tr, i) => {
    const a = 0.08 + 0.35 * (i / st.trails.length);
    trail(tr.trail, tr.ok === false
      ? `rgba(255,120,110,${a * 0.7})` : `rgba(160,200,255,${a})`, 1.5);
  });

  // ghost candidates — what the solver just weighed
  if (st.ghosts) {
    for (const c of st.ghosts.cands) trail(c.trail, 'rgba(200,200,220,.14)', 1);
    const picked = st.ghosts.cands.find(c => c.seq === st.ghosts.pick)
      || st.ghosts.cands[0];
    if (picked) trail(picked.trail, 'rgba(255,235,130,.85)', 2);
    if (st.ghosts.goal && st.ghosts.goal.tx !== undefined) {
      const { tx, ty } = st.ghosts.goal;
      g.strokeStyle = '#ffd75e'; g.lineWidth = 1.5;
      g.beginPath(); g.moveTo(X(tx) - 8, Y(ty)); g.lineTo(X(tx) + 8, Y(ty));
      g.moveTo(X(tx), Y(ty) - 8); g.lineTo(X(tx), Y(ty) + 8); g.stroke();
    }
  }

  // archive nodes + parent links
  for (const n of st.nodes) {
    if (!n || n.px === undefined) continue;
    if (n.parent >= 0 && st.nodes[n.parent]) {
      const p = st.nodes[n.parent];
      g.strokeStyle = 'rgba(110,168,255,.10)';
      g.beginPath(); g.moveTo(X(n.px), Y(n.py)); g.lineTo(X(p.px), Y(p.py)); g.stroke();
    }
  }
  for (const n of st.nodes) {
    if (!n || n.px === undefined) continue;
    const kcol = n.kind || (n.kinds && n.kinds[0]) || '';
    g.fillStyle = n.dead ? '#5a3a3a' : (KIND_COLOR[kcol] || '#9aa4b8');
    g.beginPath(); g.arc(X(n.px), Y(n.py), kcol === 'seed' ? 5 : 3, 0, 7); g.fill();
    if (st.elites.has(n.i)) { // maxmap elite — gold ring
      g.strokeStyle = '#ffd75e'; g.lineWidth = 1.5;
      g.beginPath(); g.arc(X(n.px), Y(n.py), 5.5, 0, 7); g.stroke(); g.lineWidth = 1;
    }
    if (n === sel) { g.strokeStyle = '#fff'; g.stroke(); }
  }
  // scheduler highlight: which entry the fuzzer/explorer just picked —
  // gold = maxmap queue, green = go-explore cell, blue = AFL pick
  if (st.pick && st.nodes[st.pick.node]) {
    const p = st.nodes[st.pick.node];
    g.strokeStyle = { max: '#ffd75e', cell: '#7ed67e' }[st.pick.via] || '#9ae0ff';
    g.beginPath(); g.arc(X(p.px), Y(p.py), 8, 0, 7); g.stroke();
  }
  // live head
  if (st.alive) {
    g.fillStyle = '#fff';
    g.beginPath(); g.arc(X(st.alive[0]), Y(st.alive[1]), 4, 0, 7); g.fill();
  }

  // status text
  $('status').innerHTML =
    `phase <b>${st.phase}</b> · archive <b>${st.nodes.length}</b>` +
    ` · expanded <b>${doneInfo ? doneInfo.expanded : '…'}</b>` +
    ` · finds <b>${st.findList.length}</b>` +
    (st.cells ? ` · cells <b>${st.cells}</b>` : '') +
    ` · maxScore <b>${Math.max(st.maxScore, doneInfo?.maxScore || 0)}</b>` +
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
  }
  else if (st.goal) gd.innerHTML = `goal <b>${decode(st.goal.goal)}</b> · want <b>${st.goal.want}</b> · d<b>${st.goal.d}</b>`;
  else if (st.ghosts) gd.innerHTML = `node <b>${st.ghosts.node}</b> · ${st.ghosts.kind} · ${st.ghosts.cands.length} candidates`;
  $('evpos').textContent = pos + '/' + log.length;

  // candidates panel
  const cd = $('cands');
  if (st.ghosts) {
    const acts = meta ? meta.actions : [];
    cd.innerHTML = st.ghosts.cands.slice(0, 10).map(c => {
      const seq = c.seq.map(a => acts[a] ?? a).join(' ▸ ');
      const isPick = c.seq === st.ghosts.pick;
      return `<div class="cand${isPick ? ' pick' : ''}">${c.score === Infinity ? '∞' :
        (typeof c.score === 'number' ? c.score.toFixed(2) : c.score)} · ${seq}</div>`;
    }).join('');
  }
  // finds
  $('findlist').innerHTML = st.findList.slice(-40).reverse()
    .map(it => `<div class="find">${escapeHtml(decode(it))} <span class="ln">${it}</span></div>`).join('');
}
const escapeHtml = s => s.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

// ---- selection -------------------------------------------------------------
let sel = null;
cv.addEventListener('click', e => {
  const r = cv.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  let best = null, bd = 144; // px² threshold
  for (const n of st.nodes) {
    if (!n || n.px === undefined) continue;
    const d = (X(n.px) / devicePixelRatio - mx) ** 2 + (Y(n.py) / devicePixelRatio - my) ** 2;
    if (d < bd) { bd = d; best = n; }
  }
  sel = best;
  $('node').innerHTML = best
    ? `#<b>${best.i}</b> ← ${best.parent ?? '?'} · ${best.kind || (best.kinds || []).join('+')} · pos <b>(${(best.px || 0).toFixed(0)},${(best.py || 0).toFixed(0)})</b> · score <b>${best.score ?? '?'}</b> · coins <b>${best.got ?? '?'}</b> · tape <b>${best.tlen ?? '?'}</b>${best.dead ? ' · <b style="color:#ff8f8f">dead</b>' : ''}` +
      (st.elites.has(best.i) ? ` · <b style="color:#ffd75e">elite r${st.elites.get(best.i).rule}@${st.elites.get(best.i).slot}=${st.elites.get(best.i).val}</b>` : '')
    : 'click a dot';
  render();
});

// ---- pump ------------------------------------------------------------------
function pump() {
  if (!playing) return;
  const step = [1, 2, 4, 8, 16, 32, 64, 128][+$('speed').value - 1] || 4;
  const n = Math.min(pos + step, log.length);
  replayTo(n);
  if (pos >= log.length && doneInfo) setPlaying(false);
  requestAnimationFrame(pump);
}
function setPlaying(v) {
  playing = v;
  $('play').textContent = playing ? 'pause' : 'play';
  if (playing) requestAnimationFrame(pump);
}
$('play').onclick = () => setPlaying(!playing);
$('scrub').oninput = () => { setPlaying(false); replayTo(+$('scrub').value); };
onresize = () => { fit(); render(); };

// ---- worker ----------------------------------------------------------------
// Rust-fuzzer mode — tail the FUZZ_WATCH JSONL the `fuzz_mimo_watch`
// test writes. The level map comes from mimo/level.csv (the docpage's
// data cell), since the interpreter lane has no plan/meta to send.
let tailTimer = null;
$('tail').onclick = async () => {
  if (worker) { worker.terminate(); worker = null; }
  if (tailTimer) { clearInterval(tailTimer); tailTimer = null; }
  log.length = 0; pos = 0; doneInfo = null; meta = null; levelMeta = null;
  st = freshState(); sel = null;
  $('err').textContent = ''; $('findlist').innerHTML = '';
  try {
    const csv = await (await fetch('./mimo/level.csv')).text();
    levelMeta = meta = { rows: csv.trim().split('\n').slice(1).map(l => l.split(',')) };
    fit();
  } catch (e) { $('err').textContent = 'level.csv: ' + e; }
  const path = $('logpath').value || '/tmp/mimo-fuzz.jsonl';
  let off = -1;
  const poll = async () => {
    try {
      const r = await fetch(`/log?path=${encodeURIComponent(path)}&after=${off}`);
      const j = await r.json();
      off = j.next;
      for (const l of j.lines) {
        try { log.push(JSON.parse(l)); } catch {}
      }
      $('scrub').max = log.length;
    } catch {}
  };
  await poll();
  tailTimer = setInterval(poll, 400);
  $('play').disabled = false;
  setPlaying(true);
};

$('run').onclick = () => {
  if (worker) worker.terminate();
  if (tailTimer) { clearInterval(tailTimer); tailTimer = null; }
  log.length = 0; pos = 0; doneInfo = null; meta = null; levelMeta = null;
  st = freshState(); sel = null;
  $('err').textContent = ''; $('findlist').innerHTML = '';
  worker = new Worker('./worker.mjs', { type: 'module' });
  worker.onmessage = e => {
    const ev = e.data;
    log.push(ev);
    $('scrub').max = log.length;
    if (ev.t === 'error' || ev.t === 'done') setPlaying(pos >= log.length ? false : playing);
  };
  worker.postMessage({ cmd: 'run', base: './mimo', budget: +$('budget').value || 6000,
    strategy: $('strategy').value, depth: +$('gedepth').value || 500 });
  $('play').disabled = false;
  setPlaying(true);
};
render();
