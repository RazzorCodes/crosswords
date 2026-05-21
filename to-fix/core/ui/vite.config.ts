import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';

const appOrtFallback = resolve(__dirname, '../../app/node_modules/onnxruntime-web');

export default defineConfig({
  plugins: [react()],
  base: process.env.VITE_CORE_LAB_BASE ?? '/',
  resolve: {
    alias: existsSync(appOrtFallback)
      ? {
          'onnxruntime-web': appOrtFallback,
        }
      : {},
  },
  server: {
    host: '127.0.0.1',
    port: 5174,
  },
});
