import { store } from '../state/store.js';

export class ProgressUI {
  private progressFill: HTMLElement;
  private progressPhase: HTMLElement;
  private statusText: HTMLElement;
  private startButton: HTMLButtonElement;
  private cancelButton: HTMLButtonElement;
  private resetButton: HTMLButtonElement;
  private lastError: string | null = null;

  constructor() {
    this.progressFill = document.getElementById('progress-fill') as HTMLElement;
    this.progressPhase = document.getElementById('progress-phase') as HTMLElement;
    this.statusText = document.getElementById('progress-status') as HTMLElement;
    this.startButton = document.getElementById('start-processing') as HTMLButtonElement;
    this.cancelButton = document.getElementById('cancel-processing') as HTMLButtonElement;
    this.resetButton = document.getElementById('reset-backend') as HTMLButtonElement;

    this.setupStoreSubscriptions();
    this.updateUIState();
  }

  private setupStoreSubscriptions(): void {
    store.on('video:loaded', () => {
      this.updateUIState();
    });

    store.on('video:cleared', () => {
      this.updateUIState();
    });

    store.on('processing:started', () => {
      this.lastError = null;
      this.updateUIState();
    });

    store.on('processing:progress', (event) => {
      this.updateProgress(event);
    });

    store.on('processing:completed', () => {
      this.updateUIState();
    });

    store.on('processing:cancelled', () => {
      this.updateUIState();
    });

    store.on('processing:error', ({ error }) => {
      this.lastError = error;
      this.showError(error);
      this.updateUIState();
    });
  }

  private updateUIState(): void {
    const hasVideo = store.getCurrentVideo() !== null;
    const isProcessing = store.isProcessing();

    // Update button states
    this.startButton.disabled = !hasVideo || isProcessing;

    if (isProcessing) {
      this.cancelButton.classList.remove('hidden');
    } else {
      this.cancelButton.classList.add('hidden');
    }

    // Show reset button if there was an error
    if (this.lastError) {
      this.resetButton?.classList.remove('hidden');
    } else {
      this.resetButton?.classList.add('hidden');
    }

    // Reset progress if not processing
    if (!isProcessing) {
      this.progressFill.style.width = '0%';
      if (!store.getCurrentVideo()) {
        this.statusText.textContent = 'Select a video to begin';
        this.progressPhase.textContent = 'IDLE';
      } else {
        this.statusText.textContent = 'Ready to process';
        this.progressPhase.textContent = 'READY';
      }
    }
  }

  private updateProgress(event: any): void {
    const progress = Math.min(100, Math.max(0, event.progress || 0));
    this.progressFill.style.width = `${progress}%`;

    const message = store.getProgressMessage();
    const stage = typeof event.stage === 'string' && event.stage.length > 0
      ? event.stage.toUpperCase()
      : 'PROCESSING';

    this.progressPhase.textContent = stage;

    const percent = progress.toFixed(0);
    this.statusText.textContent = `${percent}% · ${message}`;

    // Handle different status types
    if (event.status === 'done') {
      this.progressPhase.textContent = 'COMPLETE';
      this.statusText.textContent = 'Processing complete!';
    } else if (event.status === 'cancelled') {
      this.progressPhase.textContent = 'CANCELLED';
      this.statusText.textContent = 'Processing cancelled.';
    }
  }

  private showError(error: string): void {
    this.progressPhase.textContent = 'ERROR';
    this.statusText.textContent = `Error: ${error}`;
    this.statusText.style.color = 'var(--danger)';

    // Reset color after a delay
    setTimeout(() => {
      this.statusText.style.color = '';
    }, 5000);
  }

  // Public methods for external control
  setProgress(value: number, message: string, stage?: string): void {
    const progress = Math.min(100, Math.max(0, value));
    this.progressFill.style.width = `${progress}%`;

    if (stage && stage.length > 0) {
      this.progressPhase.textContent = stage.toUpperCase();
      this.statusText.textContent = `${progress.toFixed(0)}% · ${message}`;
    } else {
      this.statusText.textContent = `${progress.toFixed(0)}% · ${message}`;
    }
  }

  showSuccess(message: string): void {
    this.progressPhase.textContent = 'COMPLETE';
    this.statusText.textContent = message;
    this.statusText.style.color = 'var(--success)';

    setTimeout(() => {
      this.statusText.style.color = '';
    }, 3000);
  }
}
