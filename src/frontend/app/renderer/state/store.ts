import type { AnonymizerConfig, ModelInfo, ProgressEvent, VideoInfo } from '../types/api.js';

// Event types for the store
export type StoreEvents = {
  'video:loaded': { file: File; info: VideoInfo };
  'video:cleared': {};
  'processing:started': { jobId: string };
  'processing:progress': ProgressEvent;
  'processing:completed': { jobId: string };
  'processing:cancelled': {};
  'processing:error': { error: string };
  'config:updated': { config: Partial<AnonymizerConfig> };
  'ui:overlay-toggled': { visible: boolean };
  'ui:frame-changed': { frameIndex: number };
  'models:updated': { models: ModelInfo[] };
};

type EventHandler<T = any> = (data: T) => void;

export class Store {
  private listeners: Map<keyof StoreEvents, Set<EventHandler>> = new Map();
  private state = {
    currentVideo: null as File | null,
    videoInfo: null as VideoInfo | null,
    currentJobId: null as string | null,
    isProcessing: false,
    progress: 0,
    progressMessage: '',
  config: this.getDefaultConfig(),
    overlayVisible: false,
    currentFrame: 0,
    models: [] as ModelInfo[],
  };

  // Subscribe to events
  on<K extends keyof StoreEvents>(event: K, handler: EventHandler<StoreEvents[K]>): () => void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(handler);

    // Return unsubscribe function
    return () => {
      this.listeners.get(event)?.delete(handler);
    };
  }

  // Emit events
  emit<K extends keyof StoreEvents>(event: K, data: StoreEvents[K]): void {
    const handlers = this.listeners.get(event);
    if (handlers) {
      handlers.forEach(handler => {
        try {
          handler(data);
        } catch (error) {
          console.error(`Error in event handler for ${event}:`, error);
        }
      });
    }
  }

  // State getters
  getState() {
    return { ...this.state };
  }

  getCurrentVideo(): File | null {
    return this.state.currentVideo;
  }

  getVideoInfo(): VideoInfo | null {
    return this.state.videoInfo;
  }

  getCurrentJobId(): string | null {
    return this.state.currentJobId;
  }

  isProcessing(): boolean {
    return this.state.isProcessing;
  }

  getProgress(): number {
    return this.state.progress;
  }

  getProgressMessage(): string {
    return this.state.progressMessage;
  }

  getConfig(): AnonymizerConfig {
    return { ...this.state.config };
  }

  isOverlayVisible(): boolean {
    return this.state.overlayVisible;
  }

  getCurrentFrame(): number {
    return this.state.currentFrame;
  }

  getModels(): ModelInfo[] {
    return [...this.state.models];
  }

  // State actions
  setVideo(file: File, info: VideoInfo): void {
    this.state.currentVideo = file;
    this.state.videoInfo = info;
    this.emit('video:loaded', { file, info });
  }

  clearVideo(): void {
    this.state.currentVideo = null;
    this.state.videoInfo = null;
    this.state.currentFrame = 0;
    this.emit('video:cleared', {});
  }

  startProcessing(jobId: string): void {
    this.state.currentJobId = jobId;
    this.state.isProcessing = true;
    this.state.progress = 0;
    this.state.progressMessage = 'Starting...';
    this.emit('processing:started', { jobId });
  }

  updateProgress(event: ProgressEvent): void {
    this.state.progress = event.progress;
    this.state.progressMessage = event.stage_message || event.message;

    // Fallback if backend omitted job_id in progress event
    const effectiveJobId = event.job_id || this.state.currentJobId || 'unknown';

    if (event.status === 'done') {
      this.state.isProcessing = false;
      // Keep currentJobId until after download attempt
      this.emit('processing:completed', { jobId: effectiveJobId });
    } else if (event.status === 'error') {
      this.state.isProcessing = false;
      this.state.currentJobId = null;
      this.emit('processing:error', { error: event.error || 'Unknown error' });
    } else if (event.status === 'cancelled') {
      this.state.isProcessing = false;
      this.state.currentJobId = null;
      this.emit('processing:cancelled', {});
    }

    this.emit('processing:progress', event);
  }

  cancelProcessing(): void {
    this.state.isProcessing = false;
    this.state.currentJobId = null;
    this.emit('processing:cancelled', {});
  }

  updateConfig(partialConfig: Partial<AnonymizerConfig>): void {
    // Shallow merge only at top-level; nested sections must be provided nested
    this.state.config = { ...this.state.config, ...partialConfig } as AnonymizerConfig;
    this.emit('config:updated', { config: partialConfig });
  }

  toggleOverlay(): void {
    this.state.overlayVisible = !this.state.overlayVisible;
    this.emit('ui:overlay-toggled', { visible: this.state.overlayVisible });
  }

  setOverlayVisible(visible: boolean): void {
    if (this.state.overlayVisible !== visible) {
      this.state.overlayVisible = visible;
      this.emit('ui:overlay-toggled', { visible });
    }
  }

  setCurrentFrame(frameIndex: number): void {
    if (this.state.currentFrame !== frameIndex) {
      this.state.currentFrame = frameIndex;
      this.emit('ui:frame-changed', { frameIndex });
    }
  }

  private getDefaultConfig(): AnonymizerConfig {
    return {
      model: { name: '1280_nano_seg' },
      blur: { type: 'gaussian', strength: 10 },
      // Detection settings (with batch_size for performance tuning)
      detection: {
        confidence_threshold: 0.5,
        low_score_threshold: 0.1,
        batch_size: 8,
        use_sahi: true,
        inference_size: 1920,
        sahi_overlap_ratio: 0.2,
        classes_to_blur: ['plate', 'head'],
      },
      tracking: {
        type: 'bytetrack',
        use_offline_linker: true,
        params: {
          distance_gate: 0.05,
          confirm_after_N: 2,
          max_misses_M: 10,
          use_low_score_pool: true,
          use_visual_tracker: false,
          vt_max_age: 6,
          bbox_dilate_pct: 0.2,
          temporal_smooth_alpha: 1.0,
          ema_alpha: 0.6,
          embedding_similarity_gate: 0.55,
          distance_gate_hi: 0.05,
          distance_gate_lo: 0.02,
          cam_motion_comp: false,
          flow_backend: 'LK',
          vt_backend: 'TrackerNano',
          drift_gate: 0.15,
          process_noise: 1,
          offline_linker_max_misses: 30,
          offline_linker_per_frame_gate: 0.05,
        },
      },
      video: { codec: 'h264', quality: null }, // quality null => backend auto; user can override
      debug: false,
      log_level: 'INFO',
    };
  }

  setModels(models: ModelInfo[]): void {
    this.state.models = models;

    if (models.length > 0 && !models.some(model => model.name === this.state.config.model.name)) {
      const nextModel = models[0];
      if (nextModel) {
        this.updateConfig({ model: { name: nextModel.name } });
      }
    } else if (models.length === 0 && this.state.config.model.name) {
      this.updateConfig({ model: { name: '' } });
    }

    this.emit('models:updated', { models });
  }
}

// Singleton store instance
export const store = new Store();
