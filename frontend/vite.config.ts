import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { trainServicePlugin } from './plugins/trainService'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), trainServicePlugin()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8787',
        changeOrigin: true,
      },
    },
  },
})
