import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

// Registered after load so it never delays first paint. Safe to no-op in
// dev/non-HTTPS contexts — browsers that don't support it just skip this.
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js').catch(() => {
      // Not fatal — the app works identically without it, just without the
      // installability/offline-shell benefits.
    })
  })
}
