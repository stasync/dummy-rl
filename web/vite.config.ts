import { defineConfig } from 'vite';

export default defineConfig({
  // Relative asset paths: the same build works on GitHub Pages (sub-path) and itch.io.
  base: './',
  // Don't pre-bundle MuJoCo: it finds mujoco.wasm next to itself via new URL(..., import.meta.url).
  optimizeDeps: { exclude: ['@mujoco/mujoco'] },
  // The robot XML lives in ../assets (single source of truth, shared with training).
  server: { fs: { allow: ['..'] } },
  build: { target: 'es2022', chunkSizeWarningLimit: 1500 },
});
