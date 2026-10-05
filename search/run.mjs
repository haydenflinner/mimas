// mimo coverage-directed solve on the wasm lane — thin CLI over driver.mjs.
// The solver core lives in driver.mjs; this file just loads assets from disk,
// runs beam+aim to completion, and prints the JSON summary (same shape as
// before the refactor).
import fs from 'node:fs';
import { performance } from 'node:perf_hooks';
import { boot } from './driver.mjs';

const DIR = process.env.DIR || new URL('./mimo/', import.meta.url).pathname;
const assets = {
  manifest: JSON.parse(fs.readFileSync(DIR + '/manifest.json')),
  plan: JSON.parse(fs.readFileSync(DIR + '/plan.json')),
  src: fs.readFileSync(DIR + '/src_plain.mimas', 'utf8'),
  wasm: fs.readFileSync(DIR + '/game.wasm'),
};

const ctx = await boot(assets);
const { s, M, PLAN, ACTIONS, PROF, FT, RECIDS, t0, STRATEGY } = ctx;
if (STRATEGY === 'goexplore') s.goexplore();
else { s.beam(); if (!s.done()) s.aim(); }
const ms = performance.now() - t0;

const found = s.witnesses.find(w => w.item.startsWith('p:') &&
  PLAN.points[Number(w.item.slice(2))].line === M.target_line);
const maxScore = s.archive.reduce((m, n) => Math.max(m, n.score || 0), 0);
console.log(JSON.stringify({
  ms: Math.round(ms), strategy: STRATEGY, expanded: s.expanded,
  states: s.archive.length, cells: s.cellsN || 0,
  covered: s.covered.size, maxScore,
  found: found ? found.tape.map(a => ACTIONS[a].join('+') || 'none') : null,
  witnesses: s.witnesses.length,
  prof: PROF, ft: FT, recids: RECIDS,
}, null, 1));
