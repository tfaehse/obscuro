export {};

declare global {
  interface BackendHealth {
    status: string;
    status_code: number;
    execution_provider: string | null;
    requested_providers: string[];
    active_providers: string[];
    detail?: string;
  }

  interface BackendStatus {
    running: boolean;
    method: 'docker' | 'uv' | null;
    pid: number | null;
    health: BackendHealth | null;
    extras: string[];
  }

  type AutoStartPreference = 'auto' | 'docker' | 'uv';

  interface BackendPreferences {
    autoStart: boolean;
    method: AutoStartPreference;
    port: number;
  }

  interface BackendManagerStatusPayload {
    dockerInstalled: boolean;
    uvInstalled: boolean;
    status: BackendStatus;
    backendRoot: string;
    backendSourcesFound: boolean;
    preferences: BackendPreferences;
  }

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
