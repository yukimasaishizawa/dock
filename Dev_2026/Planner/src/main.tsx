import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './styles/index.css';

const container = document.getElementById('root');
if (!container) throw new Error('#root が見つかりません。');

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);

// PWA: 本番ビルドのみ Service Worker を登録します（開発中はキャッシュを避ける）。
if (import.meta.env.PROD && 'serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js').catch(() => {
      // 登録に失敗しても通常の Web サイトとして動作します。
    });
  });
}
