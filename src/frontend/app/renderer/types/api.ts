// Canonical config types mirroring backend AnonymizerConfig

export interface ModelConfig {
  name: string;
}

export interface ModelInfo {
  name: string;
  filename: string;
  size_bytes: number;
  immutable?: boolean;
}

export type BlurType = 'gaussian' | 'pixelate' | 'blackout' | 'black' | 'debug';
export interface BlurConfig {
  type: BlurType;
  strength: number; // 1-100
}

export interface DetectionConfig {
  plate_threshold: number; // 0-1
  face_threshold: number;  // 0-1
  batch_size: number;      // 1-256
  use_sahi: boolean;
  inference_size: number;
  sahi_overlap_ratio: number;
}

export type TrackerType = 'dummy' | 'bytetrack' | 'botsort' | 'hybrid_sot';

export interface TrackerParams {
  distance_gate: number;
  confirm_after_N: number;
  max_misses_M: number;
  offline_linker_max_misses: number;
  offline_linker_per_frame_gate: number;
  use_low_score_pool: boolean;
  use_visual_tracker: boolean;
  vt_max_age: number;
  bbox_dilate_pct: number;
  temporal_smooth_alpha: number;
  ema_alpha: number;
  high_thresh: number;
  low_thresh: number;
  distance_gate_hi: number;
  distance_gate_lo: number;
  cam_motion_comp: boolean;
  flow_backend: string;
  vt_backend: string;
  drift_gate: number;
  process_noise: number;
}

export interface TrackingConfig {
  type: TrackerType;
  params: TrackerParams;
  use_offline_linker: boolean;
}

export interface VideoConfig {
  codec: string;               // h264, hevc, vp8, vp9
  quality?: number | null;     // 1-51 (optional)
}

export interface AnonymizerConfig {
  model: ModelConfig;
  blur: BlurConfig;
  detection: DetectionConfig;
  tracking: TrackingConfig;
  video: VideoConfig;
  debug: boolean;
  log_level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL';
}

export interface ConfigOptions {
  model: {
    available: string[];
    current: string;
    files?: ModelInfo[];
  };
  blur: {
    types: BlurType[];
    current_type: BlurType;
    current_strength: number;
    strength_range: [number, number];
  };
  detection: {
    current_plate_threshold: number;
    current_face_threshold: number;
    current_batch_size: number | null;
    threshold_range: [number, number];
    use_sahi: boolean;
    current_inference_size: number;
    inference_size_range: [number, number];
    current_sahi_overlap: number;
    sahi_overlap_range: [number, number];
  };
  tracking: {
    types: string[];
    current_type: string;
    params: TrackerParams;
    ranges: Record<string, [number, number]>;
    use_offline_linker: boolean;
  };
  video: {
    codecs: string[];
    current_codec: string;
    current_quality: number | null;
    quality_range: [number, number];
  };
  global: {
    log_levels: Array<'DEBUG' | 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL'>;
    current_debug: boolean;
    current_log_level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL';
  };
}

export interface ProgressEvent {
  job_id: string;
  status: 'running' | 'done' | 'error' | 'cancelled';
  progress: number; // 0-100
  stage?: string;
  stage_message?: string;
  message: string;
  error?: string;
  sequence?: number;
  updated_at?: number;
}

export interface JobResponse {
  job_id: string;
}

export interface Detection {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  class_name: string;
}

export interface VideoInfo {
  duration: number;
  fps: number;
  width: number;
  height: number;
  frame_count: number;
}

export interface OverlayRequest {
  frame_index: number;
  config: Partial<AnonymizerConfig>;
}

export interface OverlayResponse {
  overlay_image: string; // base64 encoded image
  detections: Detection[];
}
