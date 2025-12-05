import type {
  AnonymizerConfig,
  ConfigOptions,
  ModelInfo,
  ProgressEvent,
  JobResponse,
  OverlayResponse,
  OverlayRequest,
} from '../types/api.js';
import { store } from '../state/store.js';
import { getApiBaseUrl } from './port.js';

export class APIService {
  private eventSource: EventSource | null = null;

  // Always compute current base URL so dynamic port updates (window.API_PORT) are honored
  private get baseUrl(): string {
    return getApiBaseUrl();
  }

  // Upload video and start processing
  async startVideoProcessing(videoFile: File, config: AnonymizerConfig): Promise<string> {
    const formData = new FormData();
    formData.append('input', videoFile);  // Backend expects 'input', not 'video_file'
    formData.append('config', JSON.stringify(config));
    formData.append('output_filename', 'output.mp4');  // Add required output_filename

    const response = await fetch(`${this.baseUrl}/blur/video_file`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Failed to start processing: ${errorText}`);
    }

    const result: JobResponse = await response.json();
    return result.job_id;
  }

  // Subscribe to progress updates via SSE
  subscribeToProgress(jobId: string, onProgress: (event: ProgressEvent) => void, onError?: (error: Event) => void): void {
    if (this.eventSource) {
      this.eventSource.close();
    }

    const sseUrl = `${this.baseUrl}/blur/video_progress/${jobId}`;
    const debugProgress = Boolean((window as any).__OBSCURO_DEBUG_PROGRESS__);
    if (debugProgress) {
      console.debug('[progress] subscribing', { jobId, sseUrl });
    }

    this.eventSource = new EventSource(sseUrl);
    let finished = false; // track terminal state so natural close doesn't become an error

    this.eventSource.onopen = () => {
      if (debugProgress) {
        console.debug('[progress] stream opened', { jobId });
      }
    };

    this.eventSource.onmessage = (event) => {
      try {
        const data: ProgressEvent = JSON.parse(event.data);
        // Some backend progress events might omit job_id; ensure it's always present
        if (!(data as any).job_id) {
          (data as any).job_id = jobId;
        }
        if (debugProgress) {
          console.debug('[progress] event', data);
        }
        onProgress(data);
        if (data.status === 'done' || data.status === 'error' || data.status === 'cancelled') {
          // Mark finished and proactively close so onerror (close event) is ignored.
          finished = true;
          if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
          }
        }
      } catch (error) {
        console.error('Failed to parse progress event:', error);
      }
    };

    this.eventSource.onerror = (event) => {
      if (finished) {
        // Normal end-of-stream close; ignore.
        return;
      }
      console.error('SSE connection error before completion:', event);
      if (debugProgress) {
        console.debug('[progress] stream error', { jobId, event });
      }
      if (store.isProcessing()) {
        store.updateProgress({
          job_id: jobId,
          progress: store.getProgress(),
          status: 'error',
          stage: 'disconnected',
          stage_message: 'Connection to backend lost',
          message: 'Backend disconnected',
          error: 'SSE connection failed'
        } as any);
      }
      if (this.eventSource) {
        this.eventSource.close();
        this.eventSource = null;
      }
      if (onError) onError(event);
    };
  }

  // Cancel a running job
  async cancelJob(jobId: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/blur/cancel_video/${jobId}`, {
      method: 'POST',
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Failed to cancel job: ${errorText}`);
    }
  }

  // Reset backend state
  async resetBackend(): Promise<void> {
    const response = await fetch(`${this.baseUrl}/blur/reset`, {
      method: 'POST',
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Failed to reset backend: ${errorText}`);
    }
  }

  // Get download URL for completed job
  getDownloadUrl(jobId: string): string {
    return `${this.baseUrl}/blur/download/${jobId}`;
  }

  async fetchConfigOptions(): Promise<ConfigOptions> {
    const response = await fetch(`${this.baseUrl}/blur/config/options`);
    if (!response.ok) {
      throw new Error(`Failed to load configuration options: ${response.statusText}`);
    }
    return response.json();
  }

  async listModels(): Promise<ModelInfo[]> {
    const response = await fetch(`${this.baseUrl}/blur/models`);
    if (!response.ok) {
      throw new Error(`Failed to list models: ${response.statusText}`);
    }
    const data = await response.json();
    return data.models ?? [];
  }

  async uploadModel(file: File, name?: string): Promise<ModelInfo[]> {
    const formData = new FormData();
    formData.append('file', file);
    if (name) {
      formData.append('name', name);
    }

    const response = await fetch(`${this.baseUrl}/blur/models`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Failed to upload model: ${text}`);
    }

    const data = await response.json();
    return data.models ?? [];
  }

  async deleteModel(filename: string): Promise<ModelInfo[]> {
    const response = await fetch(`${this.baseUrl}/blur/models/${encodeURIComponent(filename)}`, {
      method: 'DELETE',
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Failed to delete model: ${text}`);
    }

    const data = await response.json();
    return data.models ?? [];
  }


  // Generate overlay for specific frame using /blur/frame endpoint
  async generateOverlay(videoFile: File, request: OverlayRequest): Promise<OverlayResponse> {
    // Extract current frame from video as image blob
    const frameBlob = await this.extractFrameFromVideo(videoFile, request.frame_index);

    const formData = new FormData();
    formData.append('input', frameBlob, 'frame.jpg');

    // Supply nested overrides as JSON (partial AnonymizerConfig)
    formData.append('config', JSON.stringify(request.config || {}));

    const response = await fetch(`${this.baseUrl}/blur/frame`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Failed to generate overlay: ${errorText}`);
    }

    // Convert response blob to base64
    const blob = await response.blob();
    const base64 = await this.blobToBase64(blob);

    return {
      overlay_image: base64.includes(',') ? (base64.split(',')[1] || '') : base64, // Remove data:image/jpeg;base64, prefix if present
      detections: [] // Frame endpoint doesn't return detection coordinates
    };
  }

  // Helper method to extract frame from video
  private async extractFrameFromVideo(videoFile: File, frameIndex: number): Promise<Blob> {
    return new Promise((resolve, reject) => {
      const video = document.createElement('video');
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');

      video.onloadedmetadata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // Calculate time for the frame
        const frameTime = (frameIndex / 30); // Assuming 30 fps
        video.currentTime = frameTime;
      };

      video.onseeked = () => {
        if (ctx) {
          ctx.drawImage(video, 0, 0);
          canvas.toBlob((blob) => {
            if (blob) {
              resolve(blob);
            } else {
              reject(new Error('Failed to extract frame'));
            }
          }, 'image/jpeg', 0.8);
        } else {
          reject(new Error('Canvas context not available'));
        }

        // Clean up
        URL.revokeObjectURL(video.src);
      };

      video.onerror = () => {
        reject(new Error('Failed to load video'));
      };

      video.src = URL.createObjectURL(videoFile);
    });
  }

  // Helper method to convert blob to base64
  private async blobToBase64(blob: Blob): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  // Clean up resources
  disconnect(): void {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }
  }

  // Trigger download of completed video
  downloadVideo(jobId: string, filename?: string): void {
    const downloadUrl = this.getDownloadUrl(jobId);
    const a = document.createElement('a');
    a.href = downloadUrl;
    a.download = filename || `blurred_video_${jobId}.mp4`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }
}

// Singleton instance (dynamic port aware)
export const apiService = new APIService();
