import { PreviewUI } from './ui/preview.js';
import { ProgressUI } from './ui/progress.js';
import { PanelsUI } from './ui/panels.js';
import { ShortcutsUI } from './ui/shortcuts.js';
import { WiringUI } from './ui/wiring.js';
import { getApiBaseUrl } from './services/port.js';

// Initialize the application
class App {
  private previewUI: PreviewUI;
  private progressUI: ProgressUI;
  private panelsUI: PanelsUI;
  private shortcutsUI: ShortcutsUI; // eslint-disable-line @typescript-eslint/no-unused-vars
  private wiringUI: WiringUI; // eslint-disable-line @typescript-eslint/no-unused-vars

  constructor() {
    console.log('Initializing Obscuro application...');

    // Initialize UI modules
    console.log('Creating PreviewUI...');
    this.previewUI = new PreviewUI();

    console.log('Creating ProgressUI...');
    this.progressUI = new ProgressUI();

    console.log('Creating PanelsUI...');
    this.panelsUI = new PanelsUI();

    console.log('Creating ShortcutsUI...');
    this.shortcutsUI = new ShortcutsUI(this.previewUI);

    console.log('Creating WiringUI...');
    this.wiringUI = new WiringUI(this.previewUI, this.progressUI);

    // Initialize panels with saved states
    console.log('Initializing panels...');
    this.panelsUI.initialize();

    console.log('Obscuro application initialized successfully');
    // Start backend indicator polling
    this.setupBackendIndicator();
  }

  private setupBackendIndicator(): void {
    const dot = document.getElementById('backend-status-dot');
    const label = document.getElementById('backend-status-label');
    const restartBtn = document.getElementById('backend-restart-btn') as HTMLButtonElement | null;
    const manageBtn = document.getElementById('backend-manage-btn') as HTMLButtonElement | null;
    if (!dot || !label) return;

    let aborted = false;
    let preferenceDisposer: (() => void) | null = null;

    const setState = (state: 'ok' | 'error' | 'connecting') => {
      const baseUrl = getApiBaseUrl();
      let labelText = baseUrl.replace(/^https?:\/\//, '');
      dot.classList.remove('ok', 'err', 'connecting');
      if (state === 'ok') {
        dot.classList.add('ok');
        label.textContent = labelText;
      } else if (state === 'error') {
        dot.classList.add('err');
        label.textContent = `offline ${labelText}`;
      } else {
        dot.classList.add('connecting');
        label.textContent = `connecting… ${labelText}`;
      }
    };

    const check = async () => {
      if (aborted) return;
      try {
        const res = await fetch(`${getApiBaseUrl()}/healthz`, { method: 'HEAD' });
        if (res.ok) setState('ok'); else setState('error');
      } catch {
        setState('error');
      }
    };

    // Initial & interval polling
    setState('connecting');
    check();
    const interval = setInterval(check, 5000);

    restartBtn?.addEventListener('click', () => {
      setState('connecting');
      check();
    });

    if (window.ipc?.on) {
      preferenceDisposer = window.ipc.on('backend-manager:preferences-updated', (prefs: BackendPreferences) => {
        if (typeof prefs.port === 'number' && prefs.port !== window.API_PORT) {
          window.API_PORT = prefs.port;
          setState('connecting');
          void check();
        }
      });
    }

    manageBtn?.addEventListener('click', async () => {
      try {
        if (!window.ipc?.invoke) {
          if (window.desktopEnv?.isElectron) {
            window.alert('Backend controls are unavailable because the preload bridge failed to initialize. Try restarting the app.');
            console.error('Electron preload bridge missing despite Electron environment.');
          } else {
            window.alert('Backend manager is only available when running the desktop application.');
            console.warn('IPC bridge is unavailable; cannot open backend manager window.');
          }
          return;
        }
        const response = await window.ipc.invoke<{ ok?: boolean; error?: string }>('backend-manager:open-window');
        if (response && response.ok === false) {
          console.error('Failed to open backend manager window:', response.error);
        }
      } catch (error) {
        console.error('Failed to open backend manager window:', error);
      }
    });

    window.addEventListener('beforeunload', () => {
      aborted = true;
      clearInterval(interval);
      preferenceDisposer?.();
      preferenceDisposer = null;
    });
  }
}

// Wait for DOM to be ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    try {
      new App();
    } catch (error) {
      console.error('Failed to initialize app:', error);
    }
  });
} else {
  try {
    new App();
  } catch (error) {
    console.error('Failed to initialize app:', error);
  }
}
