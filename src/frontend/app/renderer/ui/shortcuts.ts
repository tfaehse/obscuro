import { store } from '../state/store.js';
import type { PreviewUI } from './preview.js';

export class ShortcutsUI {
  private previewUI: PreviewUI;

  constructor(previewUI: PreviewUI) {
    this.previewUI = previewUI;
    this.setupKeyboardListeners();
  }

  private setupKeyboardListeners(): void {
    window.addEventListener('keydown', (e) => this.handleKeydown(e));
  }

  private handleKeydown(e: KeyboardEvent): void {
    // Don't hijack shortcuts when user is typing in form fields
    if (this.isEditableTarget(e.target as HTMLElement)) {
      return;
    }

    switch (e.code) {
      case 'Space':
        e.preventDefault();
        this.handleSpaceKey();
        break;

      case 'ArrowLeft':
        e.preventDefault();
        this.previewUI.stepFrame(-1);
        break;

      case 'ArrowRight':
        e.preventDefault();
        this.previewUI.stepFrame(1);
        break;

      case 'Enter':
        e.preventDefault();
        this.handleEnterKey();
        break;

      case 'Escape':
        e.preventDefault();
        this.handleEscapeKey();
        break;

      // Additional shortcuts
      case 'KeyO':
        if (e.ctrlKey || e.metaKey) {
          e.preventDefault();
          this.openVideoFile();
        }
        break;

      case 'KeyS':
        if (e.ctrlKey || e.metaKey) {
          e.preventDefault();
          // Could trigger save/export functionality
        }
        break;
    }
  }

  private handleSpaceKey(): void {
    const videoElement = document.getElementById('video-preview') as HTMLVideoElement;

    if (!videoElement || !store.getCurrentVideo()) {
      return;
    }

    if (!videoElement.paused) {
      // Video is playing - pause it (which will trigger overlay generation)
      videoElement.pause();
    }
    // Always generate overlay when paused - no toggle needed
  }

  private handleEnterKey(): void {
    const startButton = document.getElementById('start-processing') as HTMLButtonElement;

    if (startButton && !startButton.disabled) {
      startButton.click();
    }
  }

  private handleEscapeKey(): void {
    const cancelButton = document.getElementById('cancel-processing') as HTMLButtonElement;

    if (cancelButton && !cancelButton.classList.contains('hidden')) {
      // Cancel ongoing processing
      cancelButton.click();
    } else {
      // No processing - refresh UI and generate overlay for current frame
      const videoElement = document.getElementById('video-preview') as HTMLVideoElement;
      if (videoElement && store.getCurrentVideo()) {
        videoElement.pause();
        this.previewUI.generateOverlayForCurrentFrame();
      }
    }
  }

  private openVideoFile(): void {
    const fileInput = document.getElementById('video-upload') as HTMLInputElement;
    if (fileInput) {
      fileInput.click();
    }
  }

  private isEditableTarget(element: HTMLElement): boolean {
    if (!element) return false;

    const tagName = element.tagName.toLowerCase();
    const isEditable = tagName === 'input' ||
                      tagName === 'textarea' ||
                      tagName === 'select' ||
                      element.isContentEditable;

    return isEditable;
  }

  // Public method to register additional shortcuts
  addShortcut(key: string, handler: (e: KeyboardEvent) => void, options?: {
    ctrl?: boolean;
    alt?: boolean;
    shift?: boolean;
    meta?: boolean;
  }): void {
    window.addEventListener('keydown', (e) => {
      if (this.isEditableTarget(e.target as HTMLElement)) {
        return;
      }

      const matchesModifiers = (!options?.ctrl || e.ctrlKey) &&
                              (!options?.alt || e.altKey) &&
                              (!options?.shift || e.shiftKey) &&
                              (!options?.meta || e.metaKey);

      if (e.code === key && matchesModifiers) {
        e.preventDefault();
        handler(e);
      }
    });
  }

  // Display current shortcuts to user
  getShortcutHelp(): string[] {
    return [
      'Space - Toggle blur preview (when paused)',
      'Enter - Start processing',
      'Escape - Cancel processing / Refresh preview',
      '← → - Step frame by frame',
      'Ctrl+O - Open video file'
    ];
  }
}
