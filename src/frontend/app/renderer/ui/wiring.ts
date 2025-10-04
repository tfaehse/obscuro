import { store } from '../state/store.js';
import { apiService } from '../services/api.js';
import type { PreviewUI } from './preview.js';
import type { ProgressUI } from './progress.js';
import { ConfigController } from './components/config-controller.js';

export class WiringUI {
  private readonly previewUI: PreviewUI;
  private readonly progressUI: ProgressUI;
  private readonly configController: ConfigController;
  private initialDataLoaded = false;
  private initialDataTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(previewUI: PreviewUI, progressUI: ProgressUI) {
    this.previewUI = previewUI;
    this.progressUI = progressUI;
    this.configController = new ConfigController();

    this.setupEventListeners();
    this.setupStoreSubscriptions();
    this.configController.bindControls();

    void this.loadInitialData();
  }

  private setupEventListeners(): void {
    const videoUpload = document.getElementById('video-upload') as HTMLInputElement | null;
    const selectVideoBtn = document.getElementById('select-video-btn') as HTMLButtonElement | null;

    if (selectVideoBtn && videoUpload) {
      selectVideoBtn.addEventListener('click', () => videoUpload.click());
      videoUpload.addEventListener('change', (event) => {
        const file = (event.target as HTMLInputElement).files?.[0];
        if (file) {
          void this.previewUI.loadVideo(file);
        }
      });
    }

    const startButton = document.getElementById('start-processing') as HTMLButtonElement | null;
    const cancelButton = document.getElementById('cancel-processing') as HTMLButtonElement | null;

    startButton?.addEventListener('click', () => {
      void this.startProcessing();
    });
    cancelButton?.addEventListener('click', () => {
      void this.cancelProcessing();
    });
  }

  private setupStoreSubscriptions(): void {
    store.on('processing:completed', ({ jobId }) => {
      void apiService.downloadVideo(jobId);
      this.progressUI.showSuccess('Video processed successfully!');
    });

    store.on('processing:error', ({ error }) => {
      alert(`Processing failed: ${error}`);
    });

    store.on('config:updated', () => {
      this.configController.sync(store.getConfig());
      this.configController.renderModels(store.getModels());
    });

    store.on('models:updated', ({ models }) => {
      this.configController.renderModels(models);
    });
  }

  private async loadInitialData(): Promise<void> {
    try {
      const options = await apiService.fetchConfigOptions();
      const models = options.model.files ?? (await apiService.listModels());

      this.configController.applyOptions(options);
      store.setModels(models);
      this.configController.renderModels(models);
      this.configController.sync(store.getConfig());

      this.initialDataLoaded = true;
      if (this.initialDataTimer) {
        clearTimeout(this.initialDataTimer);
        this.initialDataTimer = null;
      }
    } catch (error) {
      console.error('Failed to load configuration options:', error);
      if (!this.initialDataLoaded) {
        this.scheduleInitialDataRetry();
      }
    }
  }

  private scheduleInitialDataRetry(): void {
    if (this.initialDataTimer) return;
    this.initialDataTimer = setTimeout(() => {
      this.initialDataTimer = null;
      void this.loadInitialData();
    }, 3000);
  }

  private async startProcessing(): Promise<void> {
    const currentVideo = store.getCurrentVideo();
    if (!currentVideo) {
      alert('Please select a video file first.');
      return;
    }

    try {
      const config = store.getConfig();
      const jobId = await apiService.startVideoProcessing(currentVideo, config);
      store.startProcessing(jobId);

      apiService.subscribeToProgress(
        jobId,
        event => store.updateProgress(event),
        (error) => {
          console.error('SSE connection error:', error);
          store.updateProgress({
            job_id: jobId,
            status: 'error',
            progress: 0,
            message: 'Connection error',
            error: 'SSE connection failed',
          });
        },
      );
    } catch (error) {
      console.error('Failed to start processing:', error);
      alert(`Failed to start processing: ${error}`);
    }
  }

  private async cancelProcessing(): Promise<void> {
    const jobId = store.getCurrentJobId();
    if (!jobId) return;

    try {
      await apiService.cancelJob(jobId);
      store.cancelProcessing();
    } catch (error) {
      console.error('Failed to cancel job:', error);
      alert(`Failed to cancel processing: ${error}`);
    }
  }

  async handleVideoFile(file: File): Promise<void> {
    await this.previewUI.loadVideo(file);
  }
}
