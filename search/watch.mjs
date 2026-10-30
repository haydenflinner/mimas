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
    pick: null, lastPickVia: null, elites: new Map(), findList: [], covered: 0,
    expanded: 0, maxScore: 0, alive: null, cells: 0, grid: [],
    micro: { ok: 0, fail: 0, n: 0 },
    actNames: [], acts: [], pol: [] };
}
const TRAIL_KEEP = 40;

// ---- live swarm (tail mode) ------------------------------------------------
// On each newly applied `commit` a bright marker rides the trail's
// polyline (~MARK_MS, ease-out) while the polyline fades over ~FADE_MS.
// Spawned from pump() — i.e. only for events crossing the replay head —
// so full replays during scrub don't re-trigger them.
const liveAnims = [];          // {tr, cum, total, t0, via, parent, landed}
const commitTimes = [];        // performance.now() of commits at the head
const LIVE_MAX = 100, MARK_MS = 900, FADE_MS = 1200;

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
  if (t === 'micro') { st.micro[d.ok ? 'ok' : 'fail']++; st.micro.n += d.n || 0; return; }
  // Rust fuzzer: scheduler decision — which corpus entry, which queue.
  if (t === 'pick') { st.pick = d; st.lastPickVia = d.via; st.goal = null; st.phase = 'fuzz'; return; }
  // Go-Explore action names + bin-occupancy snapshot [bx,by,visits,
  // cells] with per-action [tries, wins] learned stats.
  if (t === 'actions') { st.actNames = d.list || []; return; }
  if (t === 'grid') { st.grid = d.bins || []; st.acts = d.acts || []; st.pol = d.pol || []; return; }
  if (t === 'node') {
    st.nodes[d.i] = d;
    st.expanded++;
    st.maxScore = Math.max(st.maxScore, d.score || 0);
    return;
  }
  if (t === 'commit') {
    // driver: {kind, node, seq, trail} · rust: {parent, ok, trail, fault}
    // stamp the pick's `via` on the log event so pump() can tint the
    // swarm marker by intent even though `d` is a throwaway copy
    ev._via = st.lastPickVia;
    st.trails.push({ kind: d.kind ?? (d.ok ? 'commit' : 'reject'),
      node: d.node ?? d.parent, trail: d.trail, ok: d.ok !== false,
      via: st.lastPickVia });
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

// ---- live swarm ------------------------------------------------------------
// Trail tint = intent: pink for elite-queue picks, green for cell picks,
// gold when the rollout banked score (landed node's score beat parent's).
function liveColor(a) {
  const l = st.nodes[a.landed], p = st.nodes[a.parent];
  if (l && p && (l.score || 0) > (p.score || 0)) return '#ffd75e';
  return a.via === 'elite' ? '#ff9ad5' : a.via === 'detour' ? '#ffb75e' : a.via === 'goal' ? '#7ee8ff' : a.via === 'cell' ? '#7ed67e' : '#9ae0ff';
}

function spawnLive(ev) {
  if (!tailTimer || !$('swarm').checked) return;   // live-tail mode only
  const d = ev.d || ev;
  const tr = d.trail;
  if (!tr || tr.length < 2) return;
  if (liveAnims.length >= LIVE_MAX) liveAnims.shift();  // drop oldest
  const cum = new Float64Array(tr.length);   // cumulative arc length
  for (let i = 1; i < tr.length; i++)
    cum[i] = cum[i - 1] + Math.hypot(tr[i][0] - tr[i - 1][0], tr[i][1] - tr[i - 1][1]);
  liveAnims.push({ tr, cum, total: cum[tr.length - 1] || 1,
    t0: performance.now(), via: ev._via, parent: d.parent, landed: d.landed });
}

function drawSwarm() {
  const now = performance.now();
  for (let i = liveAnims.length - 1; i >= 0; i--) {
    const a = liveAnims[i];
    const t = now - a.t0;
    if (t > FADE_MS) { liveAnims.splice(i, 1); continue; }
    const col = liveColor(a);
    g.globalAlpha = 1 - t / FADE_MS;
    trail(a.tr, col, 2);
    // marker: cubic ease-out along the arc-length table
    const p = Math.min(t / MARK_MS, 1);
    const dist = (1 - (1 - p) * (1 - p) * (1 - p)) * a.total;
    let j = 1;
    while (j < a.cum.length - 1 && a.cum[j] < dist) j++;
    const f = Math.min((dist - a.cum[j - 1]) / (a.cum[j] - a.cum[j - 1] || 1), 1);
    const mx = a.tr[j - 1][0] + (a.tr[j][0] - a.tr[j - 1][0]) * f;
    const my = a.tr[j - 1][1] + (a.tr[j][1] - a.tr[j - 1][1]) * f;
    g.globalAlpha = .45;                 // colored glow
    g.fillStyle = col;
    g.beginPath(); g.arc(X(mx), Y(my), 8, 0, 7); g.fill();
    g.globalAlpha = 1;                   // white core
    g.fillStyle = '#fff';
    g.beginPath(); g.arc(X(mx), Y(my), 4, 0, 7); g.fill();
  }
  g.globalAlpha = 1;
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

  // Go-Explore 96px suppression grid — each visited bin tinted by how
  // many rollout steps it has absorbed. A violet haze = "this area has
  // been walked a gazillion times, cells here are being starved".
  if (st.grid.length) {
    const mv = Math.max(...st.grid.map(b => b[2]), 1);
    for (const [bx, by, v, c] of st.grid) {
      const a = 0.05 + 0.4 * Math.sqrt(v / mv);
      g.fillStyle = `rgba(150,90,255,${a})`;
      g.fillRect(X(bx * 96), Y(by * 96), 96 * view.s, 96 * view.s);
      g.strokeStyle = `rgba(150,90,255,${Math.min(1, a + 0.1)})`;
      g.strokeRect(X(bx * 96), Y(by * 96), 96 * view.s, 96 * view.s);
      if (v > 60) {
        g.fillStyle = `rgba(220,190,255,${Math.min(1, a + 0.2)})`;
        g.font = '9px ui-monospace,monospace';
        g.fillText(`${v}`, X(bx * 96) + 3, Y(by * 96) + 10);
      }
    }
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
    g.strokeStyle = { max: '#ffd75e', cell: '#7ed67e', elite: '#ff9ad5', detour: '#ffb75e', goal: '#7ee8ff' }[st.pick.via] || '#9ae0ff';
    g.beginPath(); g.arc(X(p.px), Y(p.py), 8, 0, 7); g.stroke();
  }
  // goal-conditioned launch: crosshair on the target the rollout is
  // steering toward, linked back to its launchpad.
  if (st.goal && st.goal.x !== undefined) {
    const { x, y, node } = st.goal;
    g.strokeStyle = '#7ee8ff'; g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(X(x) - 8, Y(y)); g.lineTo(X(x) + 8, Y(y));
    g.moveTo(X(x), Y(y) - 8); g.lineTo(X(x), Y(y) + 8); g.stroke();
    const n = st.nodes[node];
    if (n && n.px !== undefined) {
      g.strokeStyle = 'rgba(126,232,255,.35)';
      g.setLineDash([4, 4]);
      g.beginPath(); g.moveTo(X(n.px), Y(n.py)); g.lineTo(X(x), Y(y)); g.stroke();
      g.setLineDash([]);
    }
  }
  // live head
  if (st.alive) {
    g.fillStyle = '#fff';
    g.beginPath(); g.arc(X(st.alive[0]), Y(st.alive[1]), 4, 0, 7); g.fill();
  }

  // live swarm — markers riding freshly committed rollout trails.
  // Additive over the static st.trails buffer; repainted by the pump
  // rAF while tailing, cleared on pause/scrub.
  if (liveAnims.length && $('swarm').checked) drawSwarm();

  // rollouts/sec — commits that crossed the replay head in the last 2s
  const cnow = performance.now();
  while (commitTimes.length && commitTimes[0] < cnow - 2000) commitTimes.shift();

  // status text
  $('status').innerHTML =
    `phase <b>${st.phase}</b> · archive <b>${st.nodes.length}</b>` +
    ` · expanded <b>${doneInfo ? doneInfo.expanded : '…'}</b>` +
    ` · finds <b>${st.findList.length}</b>` +
    ` · roll/s <b>${(commitTimes.length / 2).toFixed(1)}</b>` +
    (st.cells ? ` · cells <b>${st.cells}</b>` : '') +
    ` · maxScore <b>${Math.max(st.maxScore, doneInfo?.maxScore || 0)}</b>` +
    ((st.micro.ok + st.micro.fail) ? ` · micro <b>${st.micro.ok}</b>/${st.micro.ok + st.micro.fail} (${st.micro.n} exp)` : '') +
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

  // candidates panel — wasm ghost sims, or Go-Explore's learned
  // action table (which inputs actually grow the archive)
  const cd = $('cands');
  if (st.acts && st.acts.length) {
    const mw = Math.max(...st.acts.map(a => a[1]), 1);
    cd.innerHTML = '<div style="opacity:.6;margin-bottom:4px">learned action win-rates · bandit σ(θ·φ) in cyan</div>' +
      st.acts.map(([tries, wins], i) => {
        const rate = tries ? (wins / tries) : 0;
        const w = Math.round(90 * wins / mw);
        const pol = st.pol && st.pol[i] !== undefined
          ? `<span style="display:inline-block;height:8px;width:${Math.round(90 * st.pol[i])}px;background:#6fd3e8;vertical-align:middle"></span> ` : '';
        return `<div class="cand"><b>${st.actNames[i] ?? i}</b> ` +
          `${wins}/${tries} (${(rate * 100).toFixed(0)}%) ` +
          `<span style="display:inline-block;height:8px;width:${w}px;background:#7ed67e;vertical-align:middle"></span> ${pol}</div>`;
      }).join('');
  }
  else if (st.ghosts) {
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
  const from = pos;
  const n = Math.min(pos + step, log.length);
  replayTo(n);
  // commits that just crossed the replay head get a swarm anim + count
  // toward the roll/s HUD. Events before `from` were already spawned on
  // an earlier pass (replayTo re-applies the whole log each frame).
  const tnow = performance.now();
  for (let i = from; i < n; i++) {
    if (log[i].t === 'commit') { commitTimes.push(tnow); spawnLive(log[i]); }
  }
  if (pos >= log.length && doneInfo) setPlaying(false);
  requestAnimationFrame(pump);
}
function setPlaying(v) {
  playing = v;
  $('play').textContent = playing ? 'pause' : 'play';
  if (playing) requestAnimationFrame(pump);
  else { liveAnims.length = 0; render(); }  // frozen markers would linger
}
$('play').onclick = () => setPlaying(!playing);
$('swarm').onchange = () => { liveAnims.length = 0; render(); };
$('scrub').oninput = () => {
  setPlaying(false);
  commitTimes.length = 0;
  replayTo(+$('scrub').value);
};
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
  liveAnims.length = 0; commitTimes.length = 0;
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
  liveAnims.length = 0; commitTimes.length = 0;
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
