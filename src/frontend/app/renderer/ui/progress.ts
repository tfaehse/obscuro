import { store } from '../state/store.js';

export class ProgressUI {
  private progressBar: HTMLProgressElement;
  private statusText: HTMLElement;
  private startButton: HTMLButtonElement;
  private cancelButton: HTMLButtonElement;

  constructor() {
    this.progressBar = document.getElementById('progress-bar') as HTMLProgressElement;
    this.statusText = document.getElementById('progress-status') as HTMLElement;
    this.startButton = document.getElementById('start-processing') as HTMLButtonElement;
    this.cancelButton = document.getElementById('cancel-processing') as HTMLButtonElement;

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
      this.showError(error);
      this.updateUIState();
    });
  }

  private updateUIState(): void {
    const hasVideo = store.getCurrentVideo() !== null;
    const isProcessing = store.isProcessing();

    // Update button states
    this.startButton.disabled = !hasVideo || isProcessing;
  // Pause removed – only start/cancel flows supported

    if (isProcessing) {
      this.cancelButton.classList.remove('hidden');
    } else {
      this.cancelButton.classList.add('hidden');
    }

    // Reset progress if not processing
    if (!isProcessing) {
      this.progressBar.value = 0;
      if (!store.getCurrentVideo()) {
        this.statusText.textContent = 'Select a video to begin';
      } else {
        this.statusText.textContent = 'Ready to process';
      }
    }
  }

  private updateProgress(event: any): void {
    this.progressBar.value = event.progress;
    const message = store.getProgressMessage();
    const prefix = typeof event.stage === 'string' && event.stage.length > 0 ? `${event.stage}: ` : '';
    this.statusText.textContent = `${prefix}${message}`;

    // Handle different status types
    if (event.status === 'done') {
      this.statusText.textContent = 'Processing complete!';
    } else if (event.status === 'cancelled') {
      this.statusText.textContent = 'Processing cancelled.';
    }
  }

  private showError(error: string): void {
    this.statusText.textContent = `Error: ${error}`;
    this.statusText.style.color = 'var(--danger)';

    // Reset color after a delay
    setTimeout(() => {
      this.statusText.style.color = '';
    }, 5000);
  }

  // Public methods for external control
  setProgress(value: number, message: string, stage?: string): void {
    this.progressBar.value = value;
    if (stage && stage.length > 0) {
      this.statusText.textContent = `${stage}: ${message}`;
    } else {
      this.statusText.textContent = message;
    }
  }

  showSuccess(message: string): void {
    this.statusText.textContent = message;
    this.statusText.style.color = 'var(--ok)';

    setTimeout(() => {
      this.statusText.style.color = '';
    }, 3000);
  }
}
