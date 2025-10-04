type PanelKey =
  | 'model'
  | 'blur'
  | 'detection'
  | 'tracking'
  | 'video'
  | 'advanced'
  | 'debug';

const STORAGE_KEY = 'ui.active-panel';

export class PanelsUI {
  private tabButtons: HTMLButtonElement[];
  private panels: HTMLElement[];
  private activeKey: PanelKey | null = null;

  constructor() {
    this.tabButtons = Array.from(document.querySelectorAll<HTMLButtonElement>('[data-panel-target]'));
    this.panels = Array.from(document.querySelectorAll<HTMLElement>('[data-panel]'));

    this.setupEventListeners();
  }

  private setupEventListeners(): void {
    this.tabButtons.forEach((button) => {
      button.addEventListener('click', () => {
        const key = button.dataset.panelTarget as PanelKey | undefined;
        if (!key) return;
        this.setActivePanel(key, true, true);
      });
    });
  }

  private setActivePanel(key: PanelKey, persist = false, focusTab = false): void {
    if (this.activeKey === key) return;
    this.activeKey = key;

    this.tabButtons.forEach((button) => {
      const isActive = button.dataset.panelTarget === key;
      button.classList.toggle('active', isActive);
      button.setAttribute('aria-selected', isActive ? 'true' : 'false');
      if (isActive && focusTab) {
        button.focus({ preventScroll: true });
      }
    });

    this.panels.forEach((panel) => {
      const isActive = panel.dataset.panel === key;
      panel.classList.toggle('active', isActive);
      panel.toggleAttribute('hidden', !isActive);
    });

    if (persist) {
      try {
        localStorage.setItem(STORAGE_KEY, key);
      } catch {
        /* ignore persistence failures */
      }
    }
  }

  private loadActivePanel(): PanelKey | null {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (!stored) return null;
      if (this.panels.some(panel => panel.dataset.panel === stored)) {
        return stored as PanelKey;
      }
      return null;
    } catch {
      return null;
    }
  }

  initialize(): void {
    const startKey = this.loadActivePanel() ?? (this.tabButtons[0]?.dataset.panelTarget as PanelKey | undefined) ?? null;
    if (startKey) {
      this.setActivePanel(startKey);
    }
  }

  openPanel(panelId: string): void {
    if (this.panels.some(panel => panel.dataset.panel === panelId)) {
      this.setActivePanel(panelId as PanelKey, true);
    }
  }

  closePanel(_panelId: string): void {
    // No-op in tabbed layout; panels remain accessible via tabs.
  }

  togglePanelById(panelId: string): void {
    this.openPanel(panelId);
  }
}
