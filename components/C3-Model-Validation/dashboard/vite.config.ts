/// <reference types="vitest" />
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The dashboard talks to the C3 API through a same-origin proxy at /api, so the browser never needs
// CORS and the API stays on loopback. Point it elsewhere with MQVG_API_URL. The API key is typed into
// the dashboard (kept in sessionStorage) and is never baked into the bundle.
const target = process.env.MQVG_API_URL ?? "http://127.0.0.1:8000";
const proxy = {
  "/api": { target, changeOrigin: true, rewrite: (path: string) => path.replace(/^\/api/, ""), timeout: 330_000 },
};

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    chunkSizeWarningLimit: 600,   // the Recharts vendor chunk (~540 kB, loaded only on pages with charts) is deliberate
    rollupOptions: {
      output: {
        // Charts and the Markdown renderer are the heavy parts; keep them out of the entry chunk.
        manualChunks(id: string) {
          if (/node_modules[\/](recharts|d3-|victory-vendor|recharts-scale|decimal\.js-light)/.test(id)) return "charts";
          if (/node_modules[\/](react-markdown|remark|micromark|mdast|unified|hast|vfile|unist|devlop|property-information|space-separated-tokens|comma-separated-tokens|decode-named-character-reference|character-entities|trim-lines|bail|trough|is-plain-obj|html-url-attributes|estree-util-is-identifier-name)/.test(id)) return "markdown";
          return undefined;
        },
      },
    },
  },
  server: { port: 5173, proxy },
  preview: { port: 4173, proxy },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    css: false,
    testTimeout: 15_000,
    hookTimeout: 15_000,
  },
});
