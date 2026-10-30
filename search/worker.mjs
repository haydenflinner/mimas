// watch GUI worker — loads the game wasm + plan over HTTP, boots the driver,
// streams solver events to the page, runs beam+aim to completion.
// The solver loop is synchronous: events post as they're emitted and the page
// buffers them into a replayable log (pause/scrub is replay-side).
import { boot } from './driver.mjs';

onmessage = async e => {
  const m = e.data;
  if (m.cmd !== 'run') return;
  const emit = (t, d) => postMessage({ t, d });
  try {
    const [manifest, plan, src, wasm] = await Promise.all([
      fetch(m.base + '/manifest.json').then(r => r.json()),
      fetch(m.base + '/plan.json').then(r => r.json()),
      fetch(m.base + '/src_plain.mimas').then(r => r.text()),
      fetch(m.base + '/game.wasm').then(r => r.arrayBuffer()),
    ]);
    const ctx = await boot({ manifest, plan, src, wasm },
      { budget: m.budget, depth: m.depth, strategy: m.strategy, onEvent: emit });
    const { s, M, PLAN, ACTIONS, PROF, FT, t0, STRATEGY } = ctx;

    // level + movement-graph geometry for the map renderer
    emit('meta', {
      rows: ctx.LROWS, coins: ctx.COINS.map(c => ({ x: c.x, y: c.y })),
      flag: ctx.FLAG, nodes: ctx.NODES.map(n => ({ top: n.top, l: n.l, r: n.r, id: n.id })),
      actions: ACTIONS.map(a => a.join('+') || 'none'),
      target_line: M.target_line,
      points: PLAN.points.map(p => ({ line: p.line })),
      decs: PLAN.decisions.map(d => ({ line: d.line, text: d.text })),
    });

    if (STRATEGY === 'goexplore') s.goexplore();
    else {
      emit('phase', { name: 'beam' });
      s.beam();
      emit('phase', { name: 'aim' });
      if (!s.done()) s.aim();
    }
    const ms = performance.now() - t0;
    const found = s.witnesses.find(w => w.item.startsWith('p:') &&
      PLAN.points[Number(w.item.slice(2))].line === M.target_line);
    postMessage({ t: 'done', d: {
      ms: Math.round(ms), strategy: STRATEGY, cells: s.cellsN || 0,
      expanded: s.expanded, states: s.archive.length,
      covered: s.covered.size,
      maxScore: s.archive.reduce((mx, n) => Math.max(mx, n.score || 0), 0),
      found: found ? found.tape.map(a => ACTIONS[a].join('+') || 'none') : null,
      witnesses: s.witnesses.length, prof: PROF, ft: FT,
    }});
  } catch (err) {
    postMessage({ t: 'error', d: String(err && err.stack || err) });
  }
};
