import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server forwards /api/* to the FastAPI backend (uvicorn on port 8000)
export default defineConfig({
  plugins: [react()],
  build: { chunkSizeWarningLimit: 1000 },
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
});
