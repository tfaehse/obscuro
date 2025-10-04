declare global {
  interface Window {
    API_BASE_URL?: string;
    API_PORT?: number;
    ipc?: {
      invoke<T = unknown>(channel: string, ...args: any[]): Promise<T>;
      on(channel: string, listener: (...args: any[]) => void): () => void;
    };
    desktopEnv?: {
      isElectron: boolean;
    };
  }
}

export {};
