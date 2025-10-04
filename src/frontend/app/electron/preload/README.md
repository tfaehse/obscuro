This directory contains the Electron preload script written in TypeScript (preload.ts).

Build considerations:
- Ensure `tsconfig.electron.json` includes this path so that `preload.ts` is compiled to `preload.js`.
- The Electron `BrowserWindow` in `main.ts` points to `../preload/preload.js`.
- If you alter the relative path, update `main.ts` accordingly.

Security:
- Only expose minimal, typed APIs through `contextBridge`.
- Avoid adding broad IPC pass-throughs.

Exports:
- `window.backend.restart()` to restart embedded backend.
- `window.backend.onReady(callback)` for backend readiness events.
