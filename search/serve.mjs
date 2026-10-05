// tiny static server for the watch GUI — `node serve.mjs` → :8777
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const MIME = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript',
  '.json': 'application/json', '.wasm': 'application/wasm', '.mimas': 'text/plain' };

http.createServer((req, res) => {
  const u = new URL(req.url, 'http://x');
  // /log?path=/tmp/x.jsonl&after=N — tail a FUZZ_WATCH event file:
  // returns the bytes past `after` plus the new offset. `after=-1`
  // rewinds. Local dev tool — any readable path is fair game.
  if (u.pathname === '/log') {
    const p = u.searchParams.get('path') || '/tmp/mimo-fuzz.jsonl';
    const after = +(u.searchParams.get('after') ?? -1);
    try {
      const size = fs.statSync(p).size;
      const from = after < 0 ? 0 : Math.min(after, size);
      const fd = fs.openSync(p, 'r');
      const buf = Buffer.alloc(Math.max(0, size - from));
      fs.readSync(fd, buf, 0, buf.length, from);
      fs.closeSync(fd);
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ lines: buf.toString('utf8').split('\n').filter(Boolean), next: size }));
    } catch (e) {
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ lines: [], next: after < 0 ? 0 : after }));
    }
    return;
  }
  const p = path.join(ROOT, decodeURIComponent(u.pathname));
  if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) {
    res.writeHead(404); return res.end('nope');
  }
  res.writeHead(200, { 'content-type': MIME[path.extname(p)] || 'application/octet-stream' });
  fs.createReadStream(p).pipe(res);
}).listen(8777, () => console.log('watch GUI → http://localhost:8777/watch.html'));
