// Unified backend base URL helper
// Falls back to http://localhost:<port> using the configured Electron preference (default 8000).

declare global {
  interface Window {
    API_BASE_URL?: string;
    API_PORT?: number;
  }
}

export function getApiBaseUrl(): string {
  if (window.API_BASE_URL) {
    return window.API_BASE_URL.replace(/\/$/, '');
  }
  const port = typeof window.API_PORT === 'number' ? window.API_PORT : 8000;
  return `http://localhost:${port}`;
}
