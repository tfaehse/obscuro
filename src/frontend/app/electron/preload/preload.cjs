const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('ipc', {
  invoke: (channel, ...args) => ipcRenderer.invoke(channel, ...args),
  on: (channel, listener) => {
    const wrapped = (_event, ...eventArgs) => listener(...eventArgs);
    ipcRenderer.on(channel, wrapped);
    return () => {
      ipcRenderer.removeListener(channel, wrapped);
    };
  },
});

contextBridge.exposeInMainWorld('desktopEnv', {
  isElectron: true,
});

console.log('[preload] IPC bridge exposed');
