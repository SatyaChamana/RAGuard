import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The React dev server (5173) proxies /api calls to the FastAPI backend (8090),
// so the browser sees a single origin and there are no CORS surprises.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8090',
    },
  },
})
