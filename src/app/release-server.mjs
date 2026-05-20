import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { access, stat } from 'node:fs/promises';
import { extname, resolve, sep } from 'node:path';
import { Readable } from 'node:stream';

const APP_ROOT = resolve(process.env.APP_ROOT || '/app');
const PORT = Number(process.env.PORT || '80');
const UPSTREAM_MODEL_BASE_URL = (process.env.UPSTREAM_MODEL_BASE_URL || '').replace(/\/$/, '');

const MIME_TYPES = {
  '.css': 'text/css; charset=utf-8',
  '.html': 'text/html; charset=utf-8',
  '.ico': 'image/x-icon',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.onnx': 'application/octet-stream',
  '.ort': 'application/octet-stream',
  '.png': 'image/png',
  '.pt': 'application/octet-stream',
  '.pth': 'application/octet-stream',
  '.svg': 'image/svg+xml',
  '.txt': 'text/plain; charset=utf-8',
  '.wasm': 'application/wasm',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
};

function send(res, status, body, headers = {}) {
  res.writeHead(status, headers);
  if (body) {
    res.end(body);
    return;
  }
  res.end();
}

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

async function proxyModel(req, res, pathname) {
  if (!UPSTREAM_MODEL_BASE_URL) {
    return false;
  }

  const assetPath = pathname.replace(/^\/models\//, '/');
  const upstreamUrl = `${UPSTREAM_MODEL_BASE_URL}${assetPath}`;
  const upstreamResponse = await fetch(upstreamUrl, { redirect: 'follow' });
  const headers = {
    'Cache-Control': upstreamResponse.headers.get('cache-control') || 'public, max-age=300',
    'Content-Type': upstreamResponse.headers.get('content-type') || contentTypeFor(assetPath),
  };

  if (!upstreamResponse.ok) {
    const body = req.method === 'HEAD' ? undefined : Buffer.from(await upstreamResponse.arrayBuffer());
    send(res, upstreamResponse.status, body, headers);
    return true;
  }

  res.writeHead(200, headers);
  if (req.method === 'HEAD' || !upstreamResponse.body) {
    res.end();
    return true;
  }

  Readable.fromWeb(upstreamResponse.body).pipe(res);
  return true;
}

const server = createServer(async (req, res) => {
  try {
    if (!req.url) {
      send(res, 400, 'Missing request URL\n', { 'Content-Type': 'text/plain; charset=utf-8' });
      return;
    }

    if (req.method !== 'GET' && req.method !== 'HEAD') {
      send(res, 405, 'Method not allowed\n', {
        Allow: 'GET, HEAD',
        'Content-Type': 'text/plain; charset=utf-8',
      });
      return;
    }

    const { pathname } = new URL(req.url, 'http://127.0.0.1');

    if (pathname.startsWith('/models/')) {
      const proxied = await proxyModel(req, res, pathname);
      if (proxied) {
        return;
      }
    }

    const filePath = resolveStaticPath(pathname);
    if (!filePath) {
      send(res, 400, 'Invalid path\n', { 'Content-Type': 'text/plain; charset=utf-8' });
      return;
    }

    if (await serveFile(req, res, filePath)) {
      return;
    }

    if (extname(pathname)) {
      send(res, 404, 'Not found\n', { 'Content-Type': 'text/plain; charset=utf-8' });
      return;
    }

    const indexPath = resolve(APP_ROOT, 'index.html');
    await access(indexPath);
    await serveFile(req, res, indexPath);
  } catch (error) {
    console.error('Release server error:', error);
    send(res, 500, 'Internal server error\n', { 'Content-Type': 'text/plain; charset=utf-8' });
  }
});

server.listen(PORT, () => {
  console.log(
    `Release server listening on :${PORT} (appRoot=${APP_ROOT}, upstreamModels=${UPSTREAM_MODEL_BASE_URL || 'local'})`,
  );
});
