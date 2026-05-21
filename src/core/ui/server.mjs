import { createReadStream } from 'node:fs';
import { access, stat } from 'node:fs/promises';
import { createServer } from 'node:http';
import { extname, resolve, sep } from 'node:path';

const APP_ROOT = resolve(process.env.APP_ROOT || '/app');
const PORT = Number(process.env.PORT || '80');

const MIME_TYPES = {
  '.css': 'text/css; charset=utf-8',
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.jsonl': 'application/x-ndjson; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.wasm': 'application/wasm',
};

function contentTypeFor(path) {
  return MIME_TYPES[extname(path)] || 'application/octet-stream';
}

function resolveStaticPath(pathname) {
  const relativePath = pathname === '/' ? 'index.html' : pathname.replace(/^\/+/, '');
  const filePath = resolve(APP_ROOT, relativePath);
  if (filePath !== APP_ROOT && !filePath.startsWith(`${APP_ROOT}${sep}`)) {
    return null;
  }
  return filePath;
}

async function serveFile(req, res, filePath) {
  try {
    const fileStat = await stat(filePath);
    if (!fileStat.isFile()) {
      return false;
    }
    res.writeHead(200, {
      'Content-Length': String(fileStat.size),
      'Content-Type': contentTypeFor(filePath),
    });
    if (req.method === 'HEAD') {
      res.end();
      return true;
    }
    createReadStream(filePath).pipe(res);
    return true;
  } catch {
    return false;
  }
}

const server = createServer(async (req, res) => {
  try {
    if (!req.url || (req.method !== 'GET' && req.method !== 'HEAD')) {
      res.writeHead(req.url ? 405 : 400);
      res.end();
      return;
    }
    const { pathname } = new URL(req.url, 'http://127.0.0.1');
    const filePath = resolveStaticPath(pathname);
    if (!filePath) {
      res.writeHead(400, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('Invalid path\n');
      return;
    }
    if (await serveFile(req, res, filePath)) {
      return;
    }
    if (extname(pathname)) {
      res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('Not found\n');
      return;
    }
    const indexPath = resolve(APP_ROOT, 'index.html');
    await access(indexPath);
    await serveFile(req, res, indexPath);
  } catch (error) {
    console.error('Core lab server error:', error);
    res.writeHead(500, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end('Internal server error\n');
  }
});

server.listen(PORT, () => {
  console.log(`Core lab server listening on :${PORT} (appRoot=${APP_ROOT})`);
});
