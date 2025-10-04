const MAX_LOG_CHARS = 20000;

type BackendStartMethod = 'docker' | 'uv';

type BackendStartResponse = {
  ok: boolean;
  error?: string;
};

type BackendStopResponse = BackendStartResponse;

interface BackendManagerLogPayload {
  channel: 'stdout' | 'stderr';
  data: string;
}

class BackendManagerUI {
  private readonly dockerStatusEl = document.getElementById('docker-status') as HTMLSpanElement;
  private readonly uvStatusEl = document.getElementById('uv-status') as HTMLSpanElement;
  private readonly backendStateEl = document.getElementById('backend-state') as HTMLSpanElement;
  private readonly installHintEl = document.getElementById('install-hint') as HTMLParagraphElement;
  private readonly backendProviderStatusEl = document.getElementById('backend-provider-status') as HTMLSpanElement;
  private readonly backendProviderDetailEl = document.getElementById('backend-provider-detail') as HTMLElement;
  private readonly controlHintEl = document.getElementById('control-hint') as HTMLParagraphElement;
  private readonly logEl = document.getElementById('backend-log') as HTMLPreElement;
  private readonly refreshBtn = document.getElementById('refresh-status') as HTMLButtonElement;
  private readonly closeBtn = document.getElementById('close-window') as HTMLButtonElement;
  private readonly startDockerBtn = document.getElementById('start-docker') as HTMLButtonElement;
  private readonly startUvBtn = document.getElementById('start-uv') as HTMLButtonElement;
  private readonly stopBtn = document.getElementById('stop-backend') as HTMLButtonElement;
  private readonly apiBaseUrlEl = document.getElementById('api-base-url') as HTMLElement;
  private readonly backendRootStatusEl = document.getElementById('backend-root-status') as HTMLSpanElement;
  private readonly backendRootPathEl = document.getElementById('backend-root-path') as HTMLElement;
  private readonly autoStartCheckbox = document.getElementById('auto-start-toggle') as HTMLInputElement;
  private readonly autoStartMethodSelect = document.getElementById('auto-start-method') as HTMLSelectElement;
  private readonly autoStartHintEl = document.getElementById('auto-start-hint') as HTMLParagraphElement;
  private readonly backendPortInput = document.getElementById('backend-port') as HTMLInputElement;
  private readonly dockerCommandExampleEl = document.getElementById('docker-command-example') as HTMLElement;
  private readonly uvCommandExampleEl = document.getElementById('uv-command-example') as HTMLElement;
  private readonly ipc: Window['ipc'] | null = window.ipc ?? null;

  private availability = { dockerInstalled: false, uvInstalled: false };
  private backendStatus: BackendStatus = { running: false, method: null, pid: null, health: null, extras: [] };
  private backendRoot = '';
  private backendSourcesFound = false;
  private preferences: BackendPreferences = {
    autoStart: false,
    method: 'auto',
    port: typeof window.API_PORT === 'number' ? window.API_PORT : 8000,
  };
  private logBuffer = 'Ready.\n';
  private disposers: Array<() => void> = [];
  private refreshing = false;

  constructor() {
    const initialPort = this.preferences.port;
    this.backendPortInput.value = String(initialPort);
    this.apiBaseUrlEl.textContent = window.API_BASE_URL ?? `http://localhost:${initialPort}`;
    this.updateCommandExamples(initialPort);
    this.setupListeners();
    this.disableActions(true);
    void this.refresh();
  }

  private setupListeners(): void {
    this.refreshBtn.addEventListener('click', () => {
      void this.refresh();
    });

    this.closeBtn.addEventListener('click', () => {
      window.close();
    });

    this.startDockerBtn.addEventListener('click', () => {
      void this.startBackend('docker');
    });

    this.startUvBtn.addEventListener('click', () => {
      void this.startBackend('uv');
    });

    this.stopBtn.addEventListener('click', () => {
      void this.stopBackend();
    });

    this.autoStartCheckbox.addEventListener('change', () => {
      void this.updatePreferences({ autoStart: this.autoStartCheckbox.checked });
    });

    this.autoStartMethodSelect.addEventListener('change', () => {
      const value = this.autoStartMethodSelect.value as AutoStartPreference;
      void this.updatePreferences({ method: value });
    });

    this.backendPortInput.addEventListener('change', () => {
      void this.handlePortChange();
    });

    if (this.ipc) {
      this.disposers.push(this.ipc.on('backend-manager:status-updated', (status: BackendStatus) => {
        this.backendStatus = status;
        this.render();
      }));

      this.disposers.push(this.ipc.on('backend-manager:log', (payload: BackendManagerLogPayload) => {
        this.appendLog(payload);
      }));

      this.disposers.push(this.ipc.on('backend-manager:preferences-updated', (prefs: BackendPreferences) => {
        if (typeof prefs.port === 'number') {
          window.API_PORT = prefs.port;
        }
        this.preferences = prefs;
        this.renderPreferences();
      }));
    }

    window.addEventListener('beforeunload', () => {
      for (const dispose of this.disposers) {
        try {
          dispose();
        } catch (error) {
          console.warn('Failed to dispose IPC listener', error);
        }
      }
      this.disposers = [];
    });
  }

  private async refresh(): Promise<void> {
    if (this.refreshing) {
      return;
    }
    this.refreshing = true;
    if (!this.ipc) {
      if (window.desktopEnv?.isElectron) {
        this.setHint('Backend bridge failed to load. Restart the desktop app to regain controls.', 'danger');
      } else {
        this.setHint('Desktop backend controls require the Electron app. Running in read-only mode.', 'warn');
      }
      this.render();
      this.refreshing = false;
      this.disableActions(false);
      return;
    }
    this.setHint('Re-checking environment…', 'info');
    try {
      const payload = await this.ipc.invoke<BackendManagerStatusPayload>('backend-manager:status');
      this.availability = {
        dockerInstalled: payload.dockerInstalled,
        uvInstalled: payload.uvInstalled,
      };
      this.backendStatus = payload.status;
      this.backendRoot = payload.backendRoot;
      this.backendSourcesFound = payload.backendSourcesFound;
      this.preferences = payload.preferences;
      this.render();
      if (!payload.dockerInstalled && !payload.uvInstalled) {
        this.setHint('Install Docker or uv to launch the backend from this window.', 'warn');
      } else {
        this.setHint('Environment check complete.', 'info');
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error while checking status.';
      this.setHint(message, 'danger');
    } finally {
      this.refreshing = false;
      this.disableActions(false);
    }
  }

  private async startBackend(method: BackendStartMethod): Promise<void> {
    if (!this.ipc) {
      if (window.desktopEnv?.isElectron) {
        this.setHint('Backend bridge failed to load. Restart the desktop app.', 'danger');
      } else {
        this.setHint('Backend controls are unavailable outside the desktop application.', 'danger');
      }
      return;
    }
    this.disableActions(true);
    this.setHint(`Starting backend via ${method}…`, 'info');
    try {
      const response = await this.ipc.invoke<BackendStartResponse>('backend-manager:start', { method });
      if (!response.ok) {
        this.setHint(response.error ?? `Failed to start backend via ${method}.`, 'danger');
      } else {
        this.setHint(`Backend launch requested via ${method}.`, 'info');
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : `Failed to start backend via ${method}.`;
      this.setHint(message, 'danger');
    } finally {
      await this.refresh();
      this.disableActions(false);
    }
  }

  private async stopBackend(): Promise<void> {
    if (!this.ipc) {
      if (window.desktopEnv?.isElectron) {
        this.setHint('Backend bridge failed to load. Restart the desktop app.', 'danger');
      } else {
        this.setHint('Backend controls are unavailable outside the desktop application.', 'danger');
      }
      return;
    }
    this.disableActions(true);
    this.setHint('Stopping backend…', 'info');
    try {
      const response = await this.ipc.invoke<BackendStopResponse>('backend-manager:stop');
      if (!response.ok) {
        this.setHint(response.error ?? 'Failed to stop backend.', 'danger');
      } else {
        this.setHint('Backend stop requested.', 'info');
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to stop backend.';
      this.setHint(message, 'danger');
    } finally {
      await this.refresh();
      this.disableActions(false);
    }
  }

  private async handlePortChange(): Promise<void> {
    const raw = this.backendPortInput.value.trim();
    const parsed = Number.parseInt(raw, 10);
    if (!Number.isFinite(parsed) || parsed < 1 || parsed > 65535) {
      this.setHint('Enter a valid port between 1 and 65535.', 'danger');
      this.backendPortInput.value = String(this.preferences.port);
      return;
    }
    if (this.backendStatus.running) {
      this.backendPortInput.value = String(this.preferences.port);
      this.setHint('Stop the backend before changing the port.', 'warn');
      return;
    }
    if (parsed === this.preferences.port) {
      return;
    }
    await this.updatePreferences({ port: parsed });
  }

  private disableActions(disabled: boolean): void {
    const ipcUnavailable = !this.ipc;
    this.refreshBtn.disabled = disabled || ipcUnavailable;
    this.startDockerBtn.disabled = disabled || ipcUnavailable || !this.availability.dockerInstalled || this.backendStatus.running;
    this.startUvBtn.disabled = disabled || ipcUnavailable || !this.availability.uvInstalled || this.backendStatus.running;
    this.stopBtn.disabled = disabled || ipcUnavailable || !this.backendStatus.running;
    this.autoStartCheckbox.disabled = disabled || ipcUnavailable;
    this.autoStartMethodSelect.disabled = disabled || ipcUnavailable || !this.autoStartCheckbox.checked;
    this.backendPortInput.disabled = disabled || ipcUnavailable || this.backendStatus.running;
  }

  private render(): void {
    this.renderEnvironment();
    this.renderPreferences();
    this.disableActions(false);
  }

  private updateBadge(element: HTMLElement, tone: 'ok' | 'warn' | 'danger' | 'info', text: string): void {
    element.className = `status-badge ${tone}`;
    element.textContent = text;
  }

  private renderEnvironment(): void {
    this.updateBadge(this.dockerStatusEl, this.availability.dockerInstalled ? 'ok' : 'danger', this.availability.dockerInstalled ? 'Available' : 'Missing');
    this.updateBadge(this.uvStatusEl, this.availability.uvInstalled ? 'ok' : 'danger', this.availability.uvInstalled ? 'Available' : 'Missing');

    if (!this.availability.dockerInstalled && !this.availability.uvInstalled) {
      this.installHintEl.classList.remove('hidden');
    } else {
      this.installHintEl.classList.add('hidden');
    }

    if (this.backendStatus.running) {
      const methodLabel = this.backendStatus.method ?? 'unknown';
      const pidLabel = this.backendStatus.pid ? ` (pid ${this.backendStatus.pid})` : '';
      this.updateBadge(this.backendStateEl, 'ok', `Running via ${methodLabel}${pidLabel}`);
    } else {
      this.updateBadge(this.backendStateEl, 'warn', 'Not running');
    }

    this.updateBadge(this.backendRootStatusEl, this.backendSourcesFound ? 'ok' : 'warn', this.backendSourcesFound ? 'Found' : 'Missing');
    this.backendRootPathEl.textContent = this.backendRoot || 'Not detected';

    if (this.backendStatus.health) {
      const { status, status_code, execution_provider, requested_providers, active_providers, detail } = this.backendStatus.health;
      let tone: 'ok' | 'warn' | 'danger';
      if (status_code === 0) {
        tone = 'ok';
      } else if (status_code > 0) {
        tone = 'warn';
      } else {
        tone = 'danger';
      }
      const providerLabel = execution_provider ?? 'Unknown';
      this.updateBadge(this.backendProviderStatusEl, tone, `${providerLabel} (${status})`);
      const details: string[] = [];
      if (requested_providers && requested_providers.length) {
        details.push(`requested: ${requested_providers.join(', ')}`);
      }
      if (active_providers && active_providers.length) {
        details.push(`active: ${active_providers.join(', ')}`);
      }
      if (this.backendStatus.extras && this.backendStatus.extras.length) {
        details.push(`extras: ${this.backendStatus.extras.join(', ')}`);
      }
      if (detail && status_code !== 0) {
        details.push(detail);
      }
      this.backendProviderDetailEl.textContent = details.join(' • ') || 'No provider information available.';
    } else {
      const tone = this.backendStatus.running ? 'warn' : 'info';
      const label = this.backendStatus.running ? 'Waiting…' : 'Unavailable';
      this.updateBadge(this.backendProviderStatusEl, tone, label);
      const extraDetail = this.backendStatus.extras && this.backendStatus.extras.length
        ? `extras: ${this.backendStatus.extras.join(', ')}`
        : null;
      const baseDetail = this.backendStatus.running
        ? 'Awaiting health report from backend.'
        : 'Start the backend to view execution provider status.';
      this.backendProviderDetailEl.textContent = extraDetail ? `${baseDetail} • ${extraDetail}` : baseDetail;
    }
  }

  private renderPreferences(): void {
    this.autoStartCheckbox.checked = this.preferences.autoStart;
    this.autoStartMethodSelect.value = this.preferences.method;
    this.autoStartMethodSelect.disabled = !this.preferences.autoStart || !this.ipc;
    const port = typeof this.preferences.port === 'number' ? this.preferences.port : (typeof window.API_PORT === 'number' ? window.API_PORT : 8000);
    this.backendPortInput.value = String(port);
    const baseUrl = window.API_BASE_URL ?? `http://localhost:${port}`;
    this.apiBaseUrlEl.textContent = baseUrl;
    this.updateCommandExamples(port);

    let hint = 'Preferences are saved automatically.';
    let tone: 'info' | 'warn' | 'danger' | 'ok' = 'info';

    if (!this.preferences.autoStart) {
      hint = 'Auto-start is disabled. Enable the toggle to launch the backend automatically.';
    } else if (this.preferences.method === 'auto') {
      if (this.availability.uvInstalled || this.availability.dockerInstalled) {
        const preferred = this.availability.uvInstalled ? 'uv' : 'Docker';
        hint = `Auto-detect will start ${preferred} when the app opens.`;
        tone = 'ok';
      } else {
        hint = 'Auto-detect is enabled, but no runtime is currently available.';
        tone = 'warn';
      }
    } else if (this.preferences.method === 'uv') {
      if (this.availability.uvInstalled) {
        hint = 'The backend will start with uv when the app launches.';
        tone = 'ok';
      } else {
        hint = 'uv is not available. Install it or choose a different method.';
        tone = 'warn';
      }
    } else if (this.preferences.method === 'docker') {
      if (this.availability.dockerInstalled) {
        hint = 'The backend will start with Docker when the app launches.';
        tone = 'ok';
      } else {
        hint = 'Docker is not available. Install it or choose a different method.';
        tone = 'warn';
      }
    }

    this.autoStartHintEl.textContent = hint;
    this.autoStartHintEl.className = `hint tone-${tone}`;
  }

  private async updatePreferences(partial: Partial<BackendPreferences>): Promise<void> {
    if (!this.ipc) {
      this.setHint('Backend bridge failed to load. Preferences cannot be saved.', 'danger');
      return;
    }
    try {
      const updated = await this.ipc.invoke<BackendPreferences>('backend-manager:set-preferences', partial);
      this.preferences = updated;
      if (typeof updated.port === 'number') {
        window.API_PORT = updated.port;
      }
      this.renderPreferences();
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to update preferences.';
      this.setHint(message, 'danger');
      this.renderPreferences();
    }
  }

  private setHint(message: string, tone: 'info' | 'warn' | 'danger' | 'ok'): void {
    this.controlHintEl.textContent = message;
    this.controlHintEl.className = `hint tone-${tone}`;
  }

  private appendLog(payload: BackendManagerLogPayload): void {
    const prefix = payload.channel === 'stderr' ? '[stderr] ' : '';
    this.logBuffer += `${prefix}${payload.data}`;
    if (this.logBuffer.length > MAX_LOG_CHARS) {
      this.logBuffer = this.logBuffer.slice(this.logBuffer.length - MAX_LOG_CHARS);
    }
    this.logEl.textContent = this.logBuffer;
    this.logEl.scrollTop = this.logEl.scrollHeight;
  }

  private updateCommandExamples(port: number): void {
    const safePort = Number.isFinite(port) ? port : 8000;
    if (this.dockerCommandExampleEl) {
      this.dockerCommandExampleEl.textContent = `docker run --rm -p ${safePort}:${safePort} blur-gui-backend`;
    }
    if (this.uvCommandExampleEl) {
      this.uvCommandExampleEl.textContent = `uv run uvicorn blur_api.serve:app --host 0.0.0.0 --port ${safePort}`;
    }
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    new BackendManagerUI();
  });
} else {
  new BackendManagerUI();
}
