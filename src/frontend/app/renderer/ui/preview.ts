import { store } from '../state/store.js';
import { apiService } from '../services/api.js';

export class PreviewUI {
  private videoElement: HTMLVideoElement;
  private overlayElement: HTMLImageElement;
  private loadingSpinner: HTMLElement;
  private scrubRange: HTMLInputElement;
  private frameDisplay: HTMLElement;
  private durationDisplay: HTMLElement;
  private playPauseBtn: HTMLButtonElement;
  private prevFrameBtn: HTMLButtonElement;
  private nextFrameBtn: HTMLButtonElement;
  private playIcon: HTMLElement;
  private pauseIcon: HTMLElement;
  private placeholderImage: HTMLImageElement | null;
  private isGeneratingOverlay = false;
  private overlayAbortController: AbortController | null = null;
  private currentVideoUrl: string | null = null;

  constructor() {
    this.videoElement = document.getElementById('video-preview') as HTMLVideoElement;
    this.overlayElement = document.getElementById('overlay-image') as HTMLImageElement;
    this.loadingSpinner = document.getElementById('loading-spinner') as HTMLElement;
    this.scrubRange = document.getElementById('scrub-range') as HTMLInputElement;
    this.frameDisplay = document.getElementById('frame-display') as HTMLElement;
    this.durationDisplay = document.getElementById('duration-display') as HTMLElement;
    this.playPauseBtn = document.getElementById('play-pause-btn') as HTMLButtonElement;
    this.prevFrameBtn = document.getElementById('prev-frame-btn') as HTMLButtonElement;
    this.nextFrameBtn = document.getElementById('next-frame-btn') as HTMLButtonElement;
    this.playIcon = document.getElementById('play-icon') as HTMLElement;
    this.pauseIcon = document.getElementById('pause-icon') as HTMLElement;
    this.placeholderImage = document.getElementById('placeholder-image') as HTMLImageElement | null;

    this.setupEventListeners();
    this.setupStoreSubscriptions();
    this.setPlaceholderVisible(store.getCurrentVideo() === null);
  }

  private setupEventListeners(): void {
    // Video events
    this.videoElement.addEventListener('loadedmetadata', () => {
      this.updateScrubRange();
      this.updateTimeDisplay();
    });

    this.videoElement.addEventListener('timeupdate', () => {
      this.updateScrubRange();
      this.updateTimeDisplay();
      this.updateCurrentFrame();
    });

    this.videoElement.addEventListener('pause', () => {
      console.log('[Preview] pause event');
      this.generateOverlayForCurrentFrame();
      this.updatePlayPauseButton(true);
    });

    this.videoElement.addEventListener('play', () => {
      this.hideOverlay();
      this.updatePlayPauseButton(false);
    });

    // Scrub range events
    this.scrubRange.addEventListener('input', () => {
      this.seekToTime(parseFloat(this.scrubRange.value));
    });

    // Playback controls
    this.playPauseBtn.addEventListener('click', () => {
      this.togglePlayPause();
    });

    this.prevFrameBtn.addEventListener('click', () => {
      this.previousFrame();
    });

    this.nextFrameBtn.addEventListener('click', () => {
      this.nextFrame();
    });

    // Drag and drop
    const videoContainer = document.getElementById('video-container');
    if (videoContainer) {
      videoContainer.addEventListener('dragover', (e) => {
        e.preventDefault();
        videoContainer.classList.add('drag-over');
      });

      videoContainer.addEventListener('dragleave', () => {
        videoContainer.classList.remove('drag-over');
      });

      videoContainer.addEventListener('drop', (e) => {
        e.preventDefault();
        videoContainer.classList.remove('drag-over');

        const files = Array.from(e.dataTransfer?.files || []);
        const videoFile = files.find(file => file.type.startsWith('video/'));

        if (videoFile) {
          this.loadVideo(videoFile);
        }
      });
    }
  }

  private setupStoreSubscriptions(): void {
    store.on('video:loaded', ({ file }) => {
      console.log('[Preview] video:loaded event');
      this.displayVideo(file);
      this.updateControlsState();
      this.setPlaceholderVisible(false);
      // Pause and generate overlay for first frame when video loads
      this.videoElement.pause();
      setTimeout(() => this.generateOverlayForCurrentFrame(), 100);
    });
    store.on('video:cleared', () => {
      this.clearVideo();
      this.updateControlsState();
      this.setPlaceholderVisible(true);
    });

    store.on('ui:overlay-toggled', ({ visible }) => {
      if (visible && this.videoElement.paused) {
        this.generateOverlayForCurrentFrame();
      } else {
        this.hideOverlay();
      }
    });

    store.on('ui:frame-changed', () => {
      if (store.isOverlayVisible() && this.videoElement.paused) {
        this.generateOverlayForCurrentFrame();
      }
    });
  }

  async loadVideo(file: File): Promise<void> {
    try {
      // Load video into HTML5 element to extract metadata
      const url = URL.createObjectURL(file);
      this.videoElement.src = url;
      this.videoElement.load();

      // Wait for video metadata to load
      await new Promise<void>((resolve, reject) => {
        const onLoadedMetadata = () => {
          this.videoElement.removeEventListener('loadedmetadata', onLoadedMetadata);
          this.videoElement.removeEventListener('error', onError);
          resolve();
        };

        const onError = () => {
          this.videoElement.removeEventListener('loadedmetadata', onLoadedMetadata);
          this.videoElement.removeEventListener('error', onError);
          URL.revokeObjectURL(url);
          reject(new Error('Failed to load video metadata'));
        };

        this.videoElement.addEventListener('loadedmetadata', onLoadedMetadata);
        this.videoElement.addEventListener('error', onError);
      });

      // Extract video info from HTML5 video element
      const videoInfo = {
        duration: this.videoElement.duration,
        width: this.videoElement.videoWidth,
        height: this.videoElement.videoHeight,
        fps: 30, // Default fps, could be extracted from file if needed
        frame_count: Math.round(this.videoElement.duration * 30)
      };

      // Store the current URL for cleanup later
      this.currentVideoUrl = url;

      store.setVideo(file, videoInfo);
    } catch (error) {
      console.error('Failed to load video:', error);
      alert('Failed to load video. Please try again.');
    }
  }

  private displayVideo(file: File): void {
    // Video is already loaded with correct URL from loadVideo method
    // Fixed 16:9 container will handle padding automatically via object-fit: contain
    console.log('DisplayVideo called - video should already be loaded');
  }

  private clearVideo(): void {
    this.videoElement.src = '';
    this.hideOverlay();
    this.updateScrubRange();
    this.updateTimeDisplay();
    this.setPlaceholderVisible(true);

    // Clean up the blob URL
    if (this.currentVideoUrl) {
      URL.revokeObjectURL(this.currentVideoUrl);
      this.currentVideoUrl = null;
    }
  }

  private updateScrubRange(): void {
    if (this.videoElement.duration) {
      this.scrubRange.max = this.videoElement.duration.toString();
      this.scrubRange.value = this.videoElement.currentTime.toString();
      this.scrubRange.disabled = false;
    } else {
      this.scrubRange.disabled = true;
      this.scrubRange.value = '0';
    }
  }

  private updateTimeDisplay(): void {
    const current = this.formatTime(this.videoElement.currentTime || 0);
    const duration = this.formatTime(this.videoElement.duration || 0);
    this.durationDisplay.textContent = `${current} / ${duration}`;

    if (this.videoElement.duration) {
      const videoInfo = store.getVideoInfo();
      if (videoInfo) {
        const frameIndex = Math.floor((this.videoElement.currentTime / this.videoElement.duration) * videoInfo.frame_count);
        this.frameDisplay.textContent = `Frame ${frameIndex + 1} / ${videoInfo.frame_count}`;
      }
    }
  }

  private updateCurrentFrame(): void {
    if (this.videoElement.duration) {
      const videoInfo = store.getVideoInfo();
      if (videoInfo) {
        const frameIndex = Math.floor((this.videoElement.currentTime / this.videoElement.duration) * videoInfo.frame_count);
        store.setCurrentFrame(frameIndex);
      }
    }
  }

  private formatTime(seconds: number): string {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }

  private seekToTime(time: number): void {
    this.videoElement.currentTime = time;
    this.updateCurrentFrame();
  }

  async generateOverlayForCurrentFrame(): Promise<void> {
    const currentVideo = store.getCurrentVideo();
    const videoInfo = store.getVideoInfo();

    if (!currentVideo || !videoInfo || this.isGeneratingOverlay) {
      return;
    }

    // Cancel any pending request
    if (this.overlayAbortController) {
      this.overlayAbortController.abort();
    }

    this.overlayAbortController = new AbortController();
    this.isGeneratingOverlay = true;
    this.showLoadingSpinner();

    try {
      const frameIndex = store.getCurrentFrame();
      const config = store.getConfig();

      const response = await apiService.generateOverlay(currentVideo, {
        frame_index: frameIndex,
        config: {
          model: { name: config.model.name },
          blur: { type: config.blur.type, strength: config.blur.strength },
          detection: {
            confidence_threshold: config.detection.confidence_threshold,
            low_score_threshold: config.detection.low_score_threshold,
            batch_size: config.detection.batch_size,
            inference_size: config.detection.inference_size,
            sahi_overlap_ratio: config.detection.sahi_overlap_ratio,
            single_pass: config.detection.single_pass,
            disable_masks: config.detection.disable_masks,
            classes_to_blur: config.detection.classes_to_blur,
          },
          tracking: {
            type: config.tracking.type,
            use_offline_linker: config.tracking.use_offline_linker,
            params: { ...config.tracking.params },
          },
          video: { codec: config.video.codec, quality: (config.video.quality ?? null) },
          debug: config.debug,
          log_level: config.log_level,
        }
      });

      // Display overlay
      this.overlayElement.src = `data:image/png;base64,${response.overlay_image}`;
      this.overlayElement.style.display = 'block';

    } catch (error) {
      console.error('Failed to generate overlay:', error);
      this.hideOverlay();
    } finally {
      this.hideLoadingSpinner();
      this.isGeneratingOverlay = false;
      this.overlayAbortController = null;
    }
  }

  private showLoadingSpinner(): void {
    this.loadingSpinner.classList.remove('hidden');
  }

  private hideLoadingSpinner(): void {
    this.loadingSpinner.classList.add('hidden');
  }

  private hideOverlay(): void {
    this.overlayElement.style.display = 'none';
  }

    // Public methods for keyboard shortcuts
  stepFrame(direction: 1 | -1): void {
    const videoInfo = store.getVideoInfo();
    if (!videoInfo || !this.videoElement.duration) return;

    const frameTime = 1 / videoInfo.fps;
    const newTime = Math.max(0, Math.min(this.videoElement.duration, this.videoElement.currentTime + direction * frameTime));
    console.log('[Preview] stepFrame:', { direction, newTime, currentTime: this.videoElement.currentTime });
    this.seekToTime(newTime);

    // Pause and generate overlay after frame change
    this.videoElement.pause();
    setTimeout(() => this.generateOverlayForCurrentFrame(), 50);
  }

  // Playback control methods
  private togglePlayPause(): void {
    if (!store.getCurrentVideo()) return;

    if (this.videoElement.paused) {
      this.videoElement.play();
      this.updatePlayPauseButton(false);
    } else {
      this.videoElement.pause();
      this.updatePlayPauseButton(true);
    }
  }

  private previousFrame(): void {
    this.stepFrame(-1);
  }

  private nextFrame(): void {
    this.stepFrame(1);
  }

  private updatePlayPauseButton(isPaused: boolean): void {
    if (isPaused) {
      this.playIcon.classList.remove('hidden');
      this.pauseIcon.classList.add('hidden');
    } else {
      this.playIcon.classList.add('hidden');
      this.pauseIcon.classList.remove('hidden');
    }
  }

  private updateControlsState(): void {
    const hasVideo = store.getCurrentVideo() !== null;

    this.playPauseBtn.disabled = !hasVideo;
    this.prevFrameBtn.disabled = !hasVideo;
    this.nextFrameBtn.disabled = !hasVideo;
    this.scrubRange.disabled = !hasVideo;

    if (hasVideo) {
      this.updatePlayPauseButton(this.videoElement.paused);
    }
    this.setPlaceholderVisible(!hasVideo);
  }

  private setPlaceholderVisible(visible: boolean): void {
    if (!this.placeholderImage) return;
    this.placeholderImage.classList.toggle('hidden', !visible);
  }

  toggleOverlay(): void {
    store.toggleOverlay();
  }
}
