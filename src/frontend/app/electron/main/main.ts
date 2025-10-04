import { app, BrowserWindow, dialog, ipcMain } from 'electron';
import { spawn, spawnSync, ChildProcessWithoutNullStreams } from 'child_process';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { fileURLToPath } from 'url';

// ES module equivalents of __dirname and __filename
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DEFAULT_PORT = 8000;

function resolveApiUrlOverride(): string | null {
  const argIndex = process.argv.indexOf('--api-url');
  if (argIndex >= 0 && process.argv.length > argIndex + 1) {
    const value = process.argv[argIndex + 1];
    if (value) {
      return value;
    }
  }
  const envUrl = process.env.API_URL;
  if (envUrl) {
    return envUrl;
  }
  return null;
}

const API_URL_OVERRIDE = (() => {
  const override = resolveApiUrlOverride();
  return override ? override.replace(/\/$/, '') : null;
})();

const AUTO_START_OVERRIDE = (process.env.BLUR_BACKEND_AUTOSTART ?? '').toLowerCase();

function resolvePreloadPath(): string {
  const appPath = app.getAppPath();
  return path.join(appPath, 'app/electron/preload/preload.cjs');
}

function resolveIconPath(): string | undefined {
  const candidates: string[] = [];
  const appPath = app.getAppPath();
  candidates.push(path.join(appPath, 'app/electron/icon.png'));
  candidates.push(path.join(getRepoRoot(), 'src/frontend/app/electron/icon.png'));
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  return undefined;
}

function expandUserPath(input: string): string {
  if (!input) {
    return input;
  }
  if (input.startsWith('~')) {
    return path.join(os.homedir(), input.slice(1));
  }
  return input;
}

function resolveDataRootDir(): string {
  const override = process.env.BLUR_DATA_DIR;
  if (override) {
    return path.resolve(expandUserPath(override));
  }

  if (process.platform === 'win32') {
    const localAppData = process.env.LOCALAPPDATA;
    if (localAppData) {
      return path.join(localAppData, 'blur_gui');
    }
    return path.join(os.homedir(), 'AppData', 'Local', 'blur_gui');
  }

  if (process.platform === 'darwin') {
    return path.join(os.homedir(), 'Library', 'Application Support', 'blur_gui');
  }

  const xdgData = process.env.XDG_DATA_HOME;
  if (xdgData) {
    return path.join(path.resolve(expandUserPath(xdgData)), 'blur_gui');
  }

  return path.join(os.homedir(), '.local', 'share', 'blur_gui');
}

function resolveModelsDir(): string {
  const override = process.env.BLUR_MODELS_DIR;
  if (override) {
    return path.resolve(expandUserPath(override));
  }
  return path.join(resolveDataRootDir(), 'models');
}

type BackendMethod = 'docker' | 'uv';

type AutoStartPreference = 'auto' | BackendMethod;

interface BackendPreferences {
  autoStart: boolean;
  method: AutoStartPreference;
  port: number;
}

const DEFAULT_PREFERENCES: BackendPreferences = {
  autoStart: false,
  method: 'auto',
  port: DEFAULT_PORT,
};

let preferencesCache: BackendPreferences | null = null;

function getPreferencesPath(): string {
  const userDataDir = app.getPath('userData');
  return path.join(userDataDir, 'backend-manager.json');
}

function normaliseMethod(value: unknown): AutoStartPreference {
  if (value === 'uv' || value === 'docker') {
    return value;
  }
  return 'auto';
}

function normalisePort(value: unknown): number {
  if (typeof value === 'number' && Number.isFinite(value)) {
    const port = Math.round(value);
    if (port >= 1 && port <= 65535) {
      return port;
    }
  }
  if (typeof value === 'string' && value.trim().length > 0) {
    const parsed = Number.parseInt(value, 10);
    if (Number.isFinite(parsed) && parsed >= 1 && parsed <= 65535) {
      return parsed;
    }
  }
  return DEFAULT_PORT;
}

function readPreferencesFromDisk(): BackendPreferences {
  try {
    const filePath = getPreferencesPath();
    if (!fs.existsSync(filePath)) {
      return { ...DEFAULT_PREFERENCES };
    }
    const raw = fs.readFileSync(filePath, 'utf8');
    const parsed = JSON.parse(raw ?? '{}');
    const autoStart = typeof parsed.autoStart === 'boolean' ? parsed.autoStart : DEFAULT_PREFERENCES.autoStart;
    const method = normaliseMethod(parsed.method);
    const port = normalisePort(parsed.port);
    return { autoStart, method, port };
  } catch (error) {
    console.warn('Failed to read backend preferences, using defaults.', error);
    return { ...DEFAULT_PREFERENCES };
  }
}

function savePreferencesToDisk(preferences: BackendPreferences): BackendPreferences {
  const filePath = getPreferencesPath();
  try {
    fs.mkdirSync(path.dirname(filePath), { recursive: true });
    fs.writeFileSync(filePath, JSON.stringify(preferences, null, 2), 'utf8');
  } catch (error) {
    console.warn('Failed to persist backend preferences.', error);
  }
  preferencesCache = preferences;
  return preferences;
}

function getBackendPreferences(): BackendPreferences {
  if (preferencesCache) {
    return preferencesCache;
  }
  preferencesCache = readPreferencesFromDisk();
  return preferencesCache;
}

function updateBackendPreferences(partial: Partial<BackendPreferences>): BackendPreferences {
  const current = getBackendPreferences();
  const next: BackendPreferences = {
    autoStart: typeof partial.autoStart === 'boolean' ? partial.autoStart : current.autoStart,
    method: partial.method ? normaliseMethod(partial.method) : current.method,
    port: typeof partial.port !== 'undefined' ? normalisePort(partial.port) : current.port,
  };
  const changed = next.autoStart !== current.autoStart || next.method !== current.method || next.port !== current.port;
  if (!changed) {
    return current;
  }
  const saved = savePreferencesToDisk(next);
  broadcastBackendPreferences(saved);
  return saved;
}

function getConfiguredPort(): number {
  const prefs = getBackendPreferences();
  return typeof prefs.port === 'number' ? prefs.port : DEFAULT_PORT;
}

function getEffectiveApiBaseUrl(): string {
  if (API_URL_OVERRIDE) {
    return API_URL_OVERRIDE;
  }
  const port = getConfiguredPort();
  return `http://localhost:${port}`;
}

function buildRendererApiInjection(): string {
  const port = getConfiguredPort();
  const lines = [`window.API_PORT = ${port};`];
  if (API_URL_OVERRIDE) {
    lines.push(`window.API_BASE_URL = ${JSON.stringify(API_URL_OVERRIDE)};`);
  } else {
    lines.push('window.API_BASE_URL = undefined;');
  }
  lines.push("console.log('API config set to:', window.API_BASE_URL ?? '(dynamic port)', 'port', window.API_PORT);");
  return lines.join('\n');
}

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
  method: BackendMethod | null;
  pid: number | null;
  extras: string[];
  health: BackendHealth | null;
}

const DOCKER_IMAGE_BASE = process.env.BLUR_BACKEND_DOCKER_IMAGE_BASE ?? 'ghcr.io/tfaehse/blur-gui-backend';
const DOCKER_CPU_IMAGE = process.env.BLUR_BACKEND_DOCKER_CPU_IMAGE ?? `${DOCKER_IMAGE_BASE}:latest`;
const DOCKER_GPU_IMAGE = process.env.BLUR_BACKEND_DOCKER_GPU_IMAGE ?? `${DOCKER_IMAGE_BASE}:gpu`;

const backendState: { process: ChildProcessWithoutNullStreams | null; method: BackendMethod | null; extras: string[] } = {
  process: null,
  method: null,
  extras: [],
};

let backendHealth: BackendHealth | null = null;

const HEALTH_TIMEOUT_MS = 2000;
const BACKEND_STATUS_REFRESH_DELAY_MS = 2000;

function detectBackendExtras(): string[] {
  const override = process.env.BLUR_BACKEND_EXTRAS;
  if (override && override.trim().length > 0) {
    return override
      .split(',')
      .map((token) => token.trim())
      .filter((token) => token.length > 0);
  }

  const extras: string[] = [];
  if (commandAvailable('nvidia-smi', ['--version'])) {
    extras.push('gpu');
  }
  return extras;
}

const backendWindows: Set<BrowserWindow> = new Set();

function broadcastBackendPreferences(preferences: BackendPreferences): void {
  for (const win of BrowserWindow.getAllWindows()) {
    if (!win.isDestroyed()) {
      win.webContents.send('backend-manager:preferences-updated', preferences);
    }
  }
}

async function refreshBackendHealth(): Promise<void> {
  if (!backendState.process) {
    backendHealth = null;
    return;
  }
  if (typeof fetch !== 'function') {
    backendHealth = {
      status: 'error',
      status_code: -1,
      execution_provider: null,
      requested_providers: [],
      active_providers: [],
      detail: 'Fetch API is unavailable in the Electron main process.',
    };
    return;
  }
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), HEALTH_TIMEOUT_MS);
    const response = await fetch(`${getEffectiveApiBaseUrl()}/healthz`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
      signal: controller.signal,
    });
    clearTimeout(timer);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const payload = await response.json();
    const requested = Array.isArray(payload?.requested_providers)
      ? payload.requested_providers.map((item: unknown) => String(item))
      : [];
    const active = Array.isArray(payload?.active_providers)
      ? payload.active_providers.map((item: unknown) => String(item))
      : [];
    backendHealth = {
      status: typeof payload?.status === 'string' ? payload.status : 'unknown',
      status_code: typeof payload?.status_code === 'number'
        ? payload.status_code
        : typeof payload?.status === 'string' && payload.status.toLowerCase() === 'ok'
          ? 0
          : -1,
      execution_provider: payload?.execution_provider ? String(payload.execution_provider) : null,
      requested_providers: requested,
      active_providers: active,
      detail: typeof payload?.detail === 'string' ? payload.detail : undefined,
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unable to contact backend health endpoint.';
    backendHealth = {
      status: 'error',
      status_code: -1,
      execution_provider: null,
      requested_providers: [],
      active_providers: [],
      detail: message,
    };
  }
}

function getRepoRoot(): string {
  const appPath = app.getAppPath();
  // When running from source this resolves to <repo>/src/frontend
  // Package builds ship a flattened structure but still allow stepping up.
  return path.resolve(appPath, '..', '..');
}

function commandAvailable(command: string, args: string[] = ['--version']): boolean {
  try {
    const result = spawnSync(command, args, { stdio: 'ignore' });
    return !result.error;
  } catch (error: unknown) {
    if (error && typeof error === 'object' && 'code' in error && (error as { code?: string }).code === 'ENOENT') {
      return false;
    }
    return false;
  }
}

function hasBackendSources(backendPath: string): boolean {
  return fs.existsSync(path.join(backendPath, 'pyproject.toml')) &&
    fs.existsSync(path.join(backendPath, 'src', 'blur_api'));
}

function resolveBackendRoot(): { root: string; hasSources: boolean } {
  const candidates: string[] = [];
  if (process.env.BLUR_BACKEND_ROOT) {
    candidates.push(path.resolve(expandUserPath(process.env.BLUR_BACKEND_ROOT)));
  }
  if (app.isPackaged) {
    candidates.push(path.join(process.resourcesPath, 'backend'));
  }
  candidates.push(getRepoRoot());

  for (const candidate of candidates) {
    if (hasBackendSources(candidate)) {
      return { root: candidate, hasSources: true };
    }
  }

  const fallback = candidates[0] ?? getRepoRoot();
  return { root: fallback, hasSources: hasBackendSources(fallback) };
}

function isBackendMethod(value: string): value is BackendMethod {
  return value === 'docker' || value === 'uv';
}

function isAutoStartPreference(value: string): value is AutoStartPreference {
  return value === 'auto' || isBackendMethod(value);
}

function resolveAutoStartMethod(preference: AutoStartPreference): BackendMethod | null {
  if (preference === 'uv') {
    return 'uv';
  }
  if (preference === 'docker') {
    return 'docker';
  }
  if (commandAvailable('uv', ['--version'])) {
    return 'uv';
  }
  if (commandAvailable('docker', ['version'])) {
    return 'docker';
  }
  return null;
}

async function startBackend(method: BackendMethod): Promise<{ ok: boolean; error?: string }> {
  const readiness = ensureBackendNotRunning();
  if (!readiness.ok) {
    return readiness;
  }

  const { root: backendRoot, hasSources } = resolveBackendRoot();
  const modelsPath = resolveModelsDir();
  const port = getConfiguredPort();

  try {
    fs.mkdirSync(modelsPath, { recursive: true });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error creating models directory.';
    return { ok: false, error: `Failed to prepare models directory: ${message}` };
  }

  const env: Record<string, string | undefined> = {
    ...process.env,
    BLUR_MODELS_DIR: modelsPath,
    BLUR_BACKEND_ROOT: backendRoot,
  };

  if (method === 'docker') {
    if (!commandAvailable('docker', ['version'])) {
      return { ok: false, error: 'Docker is not available on this system.' };
    }

    const extras = detectBackendExtras();
    const wantsGpu = extras.includes('gpu');
    const containerModelsDir = '/data/models';
    const args = [
      'run',
      '--rm',
      '-p',
      `${port}:${port}`,
      '-e',
      `BLUR_MODELS_DIR=${containerModelsDir}`,
      '-v',
      `${modelsPath}:${containerModelsDir}`,
    ];

    if (wantsGpu) {
      args.push('--gpus', 'all');
    }

    const imageName = wantsGpu ? DOCKER_GPU_IMAGE : DOCKER_CPU_IMAGE;
    args.push(imageName);

    if (extras.length > 0) {
      env.BLUR_BACKEND_EXTRAS = extras.join(',');
    }

    try {
      const proc = spawn('docker', args, {
        cwd: backendRoot,
        env,
      });
      attachBackendProcess(proc, method, extras);
      return { ok: true };
    } catch (error) {
      console.error('Failed to start backend via Docker:', error);
      return { ok: false, error: error instanceof Error ? error.message : 'Unknown error starting Docker backend.' };
    }
  }

  if (method === 'uv') {
    if (!commandAvailable('uv', ['--version'])) {
      return { ok: false, error: 'uv is not available on this system.' };
    }
    if (!hasSources) {
      return {
        ok: false,
        error: `Backend sources not found at ${backendRoot}. Install the blur-gui Python package separately or set BLUR_BACKEND_ROOT to the checkout path.`,
      };
    }

    const extras = detectBackendExtras();
    try {
      const proc = spawn(
        'uv',
        [
          'run',
          ...extras.flatMap((extra) => ['--extra', extra]),
          'uvicorn',
          'blur_api.serve:app',
          '--host',
          '0.0.0.0',
          '--port',
          String(port),
        ],
        {
          cwd: backendRoot,
          env,
        },
      );
      attachBackendProcess(proc, method, extras);
      return { ok: true };
    } catch (error) {
      console.error('Failed to start backend via uv:', error);
      return { ok: false, error: error instanceof Error ? error.message : 'Unknown error starting uv backend.' };
    }
  }

  return { ok: false, error: `Unsupported backend start method: ${method}` };
}

function currentBackendStatus(): BackendStatus {
  return {
    running: backendState.process !== null,
    method: backendState.method,
    pid: backendState.process?.pid ?? null,
    extras: backendState.extras,
    health: backendHealth,
  };
}

async function emitBackendStatus(): Promise<void> {
  if (backendState.process) {
    await refreshBackendHealth();
  } else {
    backendHealth = null;
  }
  const status = currentBackendStatus();
  for (const win of BrowserWindow.getAllWindows()) {
    if (!win.isDestroyed()) {
      win.webContents.send('backend-manager:status-updated', status);
    }
  }
}

function ensureBackendNotRunning(): { ok: boolean; error?: string } {
  if (backendState.process) {
    return { ok: false, error: 'Backend is already running.' };
  }
  return { ok: true };
}

function clearBackendProcess(): void {
  backendState.process = null;
  backendState.method = null;
  backendState.extras = [];
  backendHealth = null;
  void emitBackendStatus();
}

function stopBackendProcess(signal: NodeJS.Signals = 'SIGINT'): { ok: boolean; error?: string } {
  const proc = backendState.process;
  if (!proc) {
    return { ok: false, error: 'Backend is not running.' };
  }

  try {
    if (!proc.killed) {
      proc.kill(signal);
    }
  } catch (error) {
    console.error('Failed to stop backend process:', error);
    return { ok: false, error: error instanceof Error ? error.message : 'Unknown error stopping backend.' };
  }

  clearBackendProcess();
  return { ok: true };
}

function attachBackendProcess(proc: ChildProcessWithoutNullStreams, method: BackendMethod, extras: string[]): void {
  backendState.process = proc;
  backendState.method = method;
  backendState.extras = extras;

  proc.stdout.setEncoding('utf8');
  proc.stderr.setEncoding('utf8');

  const forward = (channel: 'stdout' | 'stderr', data: string) => {
    for (const win of backendWindows) {
      if (!win.isDestroyed()) {
        win.webContents.send('backend-manager:log', { channel, data });
      }
    }
  };

  proc.stdout.on('data', (chunk: string | Buffer) => forward('stdout', chunk.toString()));
  proc.stderr.on('data', (chunk: string | Buffer) => forward('stderr', chunk.toString()));

  proc.once('exit', (code, signal) => {
    if (backendState.process === proc) {
      clearBackendProcess();
    }
    const message = signal ? `Backend exited (signal ${signal})` : `Backend exited (code ${code ?? 'unknown'})`;
    forward('stderr', `${message}\n`);
  });
  void emitBackendStatus();
  setTimeout(() => {
    if (backendState.process === proc) {
      void emitBackendStatus();
    }
  }, 2000);
}

function createWindow(): void {
  const iconPath = resolveIconPath();
  const windowOptions: Electron.BrowserWindowConstructorOptions = {
    width: 1320,
    height: 860,
    minWidth: 1200,
    minHeight: 760,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: resolvePreloadPath(),
    }
  };
  if (iconPath) {
    windowOptions.icon = iconPath as string | Electron.NativeImage;
  }
  const win = new BrowserWindow(windowOptions);

  // Pass the API port to the renderer process
  win.webContents.on('did-finish-load', () => {
    win.webContents.executeJavaScript(buildRendererApiInjection());
  });

  // In development, load from file system
  // In production, this would be packaged differently
  win.loadFile(path.join(__dirname, '../../../app/renderer/index.html'));

  // Open DevTools in development
  if (process.env.NODE_ENV === 'development') {
    win.webContents.openDevTools();
  }
}

function createBackendWindow(): BrowserWindow {
  const win = new BrowserWindow({
    width: 720,
    height: 640,
    minWidth: 520,
    minHeight: 480,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: resolvePreloadPath(),
    },
    title: 'Backend Manager',
  });

  backendWindows.add(win);

  win.on('closed', () => {
    backendWindows.delete(win);
  });

  win.webContents.on('did-finish-load', () => {
    win.webContents.executeJavaScript(buildRendererApiInjection());
    win.webContents.send('backend-manager:status-updated', currentBackendStatus());
    void emitBackendStatus();
  });

  win.loadFile(path.join(__dirname, '../../../app/renderer/backend.html'));

  if (process.env.NODE_ENV === 'development') {
    win.webContents.openDevTools({ mode: 'detach' });
  }

  return win;
}

// IPC handlers for file dialogs
ipcMain.handle('show-save-dialog', async (_event, options) => {
  const result = await dialog.showSaveDialog(options);
  return result;
});

ipcMain.handle('show-open-dialog', async (_event, options) => {
  const result = await dialog.showOpenDialog(options);
  return result;
});

ipcMain.handle('backend:restart', async () => {
  console.log('Renderer requested backend restart (noop when using external backend)');
  return { port: null };
});

ipcMain.handle('backend-manager:open-window', async () => {
  for (const win of backendWindows) {
    if (!win.isDestroyed()) {
      win.focus();
      return { ok: true };
    }
  }
  createBackendWindow();
  return { ok: true };
});

ipcMain.handle('backend-manager:status', async () => {
  if (backendState.process) {
    await refreshBackendHealth();
  } else {
    backendHealth = null;
  }
  const backendInfo = resolveBackendRoot();
  return {
    dockerInstalled: commandAvailable('docker', ['version']),
    uvInstalled: commandAvailable('uv', ['--version']),
    status: currentBackendStatus(),
    backendRoot: backendInfo.root,
    backendSourcesFound: backendInfo.hasSources,
    preferences: getBackendPreferences(),
  };
});

ipcMain.handle('backend-manager:get-preferences', async () => {
  return getBackendPreferences();
});

ipcMain.handle('backend-manager:set-preferences', async (_event, payload: Partial<BackendPreferences>) => {
  const next: Partial<BackendPreferences> = {};
  if (typeof payload.autoStart === 'boolean') {
    next.autoStart = payload.autoStart;
  }
  if (typeof payload.method === 'string' && isAutoStartPreference(payload.method)) {
    next.method = payload.method;
  }
  if (typeof payload.port !== 'undefined') {
    const desiredPort = normalisePort(payload.port);
    const currentPort = getConfiguredPort();
    if (backendState.process && desiredPort !== currentPort) {
      throw new Error('Stop the backend before changing the port.');
    }
    next.port = desiredPort;
  }
  return updateBackendPreferences(next);
});

ipcMain.handle('backend-manager:start', async (_event, { method }: { method: BackendMethod }) => {
  return startBackend(method);
});

ipcMain.handle('backend-manager:stop', async () => {
  return stopBackendProcess('SIGTERM');
});

// App event handlers
app.whenReady().then(async () => {
  createWindow();

  let preference: AutoStartPreference | null = null;
  if (isAutoStartPreference(AUTO_START_OVERRIDE) && AUTO_START_OVERRIDE.length > 0) {
    preference = AUTO_START_OVERRIDE;
  } else {
    const prefs = getBackendPreferences();
    if (prefs.autoStart) {
      preference = prefs.method;
    }
  }

  if (preference) {
    const resolvedMethod = preference === 'auto' ? resolveAutoStartMethod('auto') : preference;
    if (resolvedMethod) {
      const result = await startBackend(resolvedMethod);
      if (!result.ok) {
        console.error(`Failed to auto-start backend (${resolvedMethod}): ${result.error ?? 'unknown error'}`);
      }
    } else {
      console.warn('Auto-start requested but no available backend runtime was detected.');
    }
  }

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  stopBackendProcess('SIGTERM');
});
