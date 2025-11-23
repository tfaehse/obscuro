import { store } from '../../state/store.js';
import { apiService } from '../../services/api.js';
import type { AnonymizerConfig, ConfigOptions, ModelInfo, TrackerParams } from '../../types/api.js';

type Parser<T> = (value: string | boolean) => T;

const TRACKER_PRESETS: Record<string, TrackerParams> = {
  dummy: {
    distance_gate: 0.4,
    confirm_after_N: 1,
    max_misses_M: 1,
    offline_linker_max_misses: 30,
    offline_linker_per_frame_gate: 0.05,
    use_low_score_pool: false,
    use_visual_tracker: false,
    vt_max_age: 6,
    bbox_dilate_pct: 0.15,
    temporal_smooth_alpha: 0.7,
    ema_alpha: 0.6,
    high_thresh: 0.6,
    low_thresh: 0.2,
    embedding_similarity_gate: 0.55,
    distance_gate_hi: 0.05,
    distance_gate_lo: 0.02,
    cam_motion_comp: false,
    flow_backend: 'LK',
    vt_backend: 'TrackerNano',
    drift_gate: 0.15,
    process_noise: 1,
  },
  bytetrack: {
    distance_gate: 0.05,
    confirm_after_N: 2,
    max_misses_M: 10,
    offline_linker_max_misses: 30,
    offline_linker_per_frame_gate: 0.05,
    use_low_score_pool: true,
    use_visual_tracker: false,
    vt_max_age: 6,
    bbox_dilate_pct: 0.2,
    temporal_smooth_alpha: 1.0,
    ema_alpha: 0.6,
    high_thresh: 0.4,
    low_thresh: 0.1,
    embedding_similarity_gate: 0.55,
    distance_gate_hi: 0.05,
    distance_gate_lo: 0.02,
    cam_motion_comp: false,
    flow_backend: 'LK',
    vt_backend: 'TrackerNano',
    drift_gate: 0.15,
    process_noise: 1,
  },
  botsort: {
    distance_gate: 0.4,
    confirm_after_N: 3,
    max_misses_M: 5,
    offline_linker_max_misses: 30,
    offline_linker_per_frame_gate: 0.025,
    use_low_score_pool: true,
    use_visual_tracker: false,
    vt_max_age: 6,
    bbox_dilate_pct: 0.2,
    temporal_smooth_alpha: 1.0,
    ema_alpha: 0.6,
    high_thresh: 0.4,
    low_thresh: 0.1,
    embedding_similarity_gate: 0.55,
    distance_gate_hi: 0.05,
    distance_gate_lo: 0.02,
    cam_motion_comp: true,
    flow_backend: 'LK',
    vt_backend: 'TrackerNano',
    drift_gate: 0.15,
    process_noise: 1,
  },
  hybrid_sot: {
    distance_gate: 0.05,
    confirm_after_N: 5,
    max_misses_M: 2,
    offline_linker_max_misses: 30,
    offline_linker_per_frame_gate: 0.05,
    use_low_score_pool: false,
    use_visual_tracker: true,
    vt_max_age: 10,
    bbox_dilate_pct: 0.25,
    temporal_smooth_alpha: 1.0,
    ema_alpha: 0.6,
    high_thresh: 0.6,
    low_thresh: 0.2,
    embedding_similarity_gate: 0.55,
    distance_gate_hi: 0.05,
    distance_gate_lo: 0.02,
    cam_motion_comp: false,
    flow_backend: 'LK',
    vt_backend: 'TrackerNano',
    drift_gate: 0.05,
    process_noise: 1,
  },
  fused: {
    distance_gate: 0.1,
    confirm_after_N: 3,
    max_misses_M: 5,
    offline_linker_max_misses: 30,
    offline_linker_per_frame_gate: 0.05,
    use_low_score_pool: true,
    use_visual_tracker: false,
    vt_max_age: 6,
    bbox_dilate_pct: 0.2,
    temporal_smooth_alpha: 1.0,
    ema_alpha: 0.6,
    high_thresh: 0.5,
    low_thresh: 0.2,
    embedding_similarity_gate: 0.55,
    distance_gate_hi: 0.08,
    distance_gate_lo: 0.15,
    cam_motion_comp: false,
    flow_backend: 'LK',
    vt_backend: 'TrackerNano',
    drift_gate: 0.05,
    process_noise: 1,
  },
};

export class ConfigController {
  private readonly valueDisplayIds = [
    'blur-strength',
    'plate-threshold',
    'face-threshold',
    'batch-size',
    'inference-size',
    'sahi-overlap',
    'tracker-ema-alpha',
    'track-distance-gate',
    'track-confirm-after',
    'track-max-misses',
    'track-vt-max-age',
    'track-bbox-dilate',
    'track-temporal-alpha',
    'track-drift-gate',
    'track-emb-sim-gate',
    'video-quality',
  ];

  bindControls(): void {
    this.bindSectionField('model-select', 'model', 'name', v => String(v));

    this.bindSectionField('blur-type', 'blur', 'type', v => v as any);
    this.bindSectionField('blur-strength', 'blur', 'strength', v => parseInt(String(v), 10));

    this.bindSectionField('plate-threshold', 'detection', 'plate_threshold', v => parseFloat(String(v)));
    this.bindSectionField('face-threshold', 'detection', 'face_threshold', v => parseFloat(String(v)));
    this.bindSectionField('batch-size', 'detection', 'batch_size', v => parseInt(String(v), 10));
    this.bindSectionField('use-sahi', 'detection', 'use_sahi', v => Boolean(v));
    this.bindSectionField('inference-size', 'detection', 'inference_size', v => parseInt(String(v), 10));
    this.bindSectionField('sahi-overlap', 'detection', 'sahi_overlap_ratio', v => parseFloat(String(v)));

    this.bindSectionField('tracker-type', 'tracking', 'type', v => v as any, () => this.onTrackerTypeChanged());
    this.bindSectionField('track-offline-linker', 'tracking', 'use_offline_linker', v => Boolean(v));

    this.bindTrackingParam('tracker-ema-alpha', 'ema_alpha', v => parseFloat(String(v)));
    this.bindTrackingParam('track-distance-gate', 'distance_gate', v => parseFloat(String(v)));
    this.bindTrackingParam('track-distance-high', 'distance_gate_hi', v => parseFloat(String(v)));
    this.bindTrackingParam('track-distance-low', 'distance_gate_lo', v => parseFloat(String(v)));
    this.bindTrackingParam('track-confirm-after', 'confirm_after_N', v => parseInt(String(v), 10));
    this.bindTrackingParam('track-max-misses', 'max_misses_M', v => parseInt(String(v), 10));
    this.bindTrackingParam('track-low-score-pool', 'use_low_score_pool', v => Boolean(v));
    this.bindTrackingParam('track-high-thresh', 'high_thresh', v => parseFloat(String(v)));
    this.bindTrackingParam('track-low-thresh', 'low_thresh', v => parseFloat(String(v)));
    this.bindTrackingParam('track-emb-sim-gate', 'embedding_similarity_gate', v => parseFloat(String(v)));
    this.bindTrackingParam('track-cam-motion', 'cam_motion_comp', v => Boolean(v));
    this.bindTrackingParam('track-flow-backend', 'flow_backend', v => String(v));
    this.bindTrackingParam('track-use-visual', 'use_visual_tracker', v => Boolean(v));
    this.bindTrackingParam('track-vt-backend', 'vt_backend', v => String(v));
    this.bindTrackingParam('track-vt-max-age', 'vt_max_age', v => parseInt(String(v), 10));
    this.bindTrackingParam('track-bbox-dilate', 'bbox_dilate_pct', v => parseFloat(String(v)));
    this.bindTrackingParam('track-temporal-alpha', 'temporal_smooth_alpha', v => parseFloat(String(v)));
    this.bindTrackingParam('track-drift-gate', 'drift_gate', v => parseFloat(String(v)));

    this.bindSectionField('video-codec', 'video', 'codec', v => String(v));
    this.bindSectionField('video-quality', 'video', 'quality', v => parseInt(String(v), 10));

    this.bindGlobalCheckbox('debug-mode', 'debug');
    this.bindGlobalSelect('log-level', 'log_level');

    this.setupQualityOverride();
    this.attachValueBadgeListeners();
    this.setupModelManager();
    this.refreshValueBadges();
    this.updateParamVisibility();
    this.updateDebugConfig();
  }

  applyOptions(options: ConfigOptions): void {
    this.populateTrackerSelect(options.tracking.types);
    this.applyRange('track-distance-gate', options.tracking.ranges['distance_gate']);
    this.applyRange('track-distance-high', options.tracking.ranges['distance_gate_hi']);
    this.applyRange('track-distance-low', options.tracking.ranges['distance_gate_lo']);
    this.applyRange('track-confirm-after', options.tracking.ranges['confirm_after_N']);
    this.applyRange('track-max-misses', options.tracking.ranges['max_misses_M']);
    this.applyRange('track-vt-max-age', options.tracking.ranges['vt_max_age']);
    this.applyRange('track-bbox-dilate', options.tracking.ranges['bbox_dilate_pct']);
    this.applyRange('track-temporal-alpha', options.tracking.ranges['temporal_smooth_alpha']);
    this.applyRange('track-drift-gate', options.tracking.ranges['drift_gate']);

    if (options.detection.inference_size_range) {
      const [min, max] = options.detection.inference_size_range;
      const inferenceInput = document.getElementById('inference-size') as HTMLInputElement | null;
      if (inferenceInput) {
        inferenceInput.min = String(min);
        inferenceInput.max = String(max);
      }
    }

    const current = store.getConfig();
    const detectionBatch = options.detection.current_batch_size ?? current.detection.batch_size;

    store.updateConfig({
      model: { name: options.model.current },
      blur: { type: options.blur.current_type as any, strength: options.blur.current_strength },
      detection: {
        plate_threshold: options.detection.current_plate_threshold,
        face_threshold: options.detection.current_face_threshold,
        batch_size: detectionBatch,
        use_sahi: options.detection.use_sahi ?? current.detection.use_sahi,
        inference_size: options.detection.current_inference_size ?? current.detection.inference_size,
        sahi_overlap_ratio: options.detection.current_sahi_overlap ?? current.detection.sahi_overlap_ratio,
        classes_to_blur: options.detection.current_classes ?? current.detection.classes_to_blur,
      },
      tracking: {
        type: options.tracking.current_type as any,
        use_offline_linker: options.tracking.use_offline_linker ?? current.tracking.use_offline_linker,
        params: { ...options.tracking.params },
      },
      video: {
        codec: options.video.current_codec,
        quality: options.video.current_quality,
      },
      debug: options.global.current_debug,
      log_level: options.global.current_log_level,
    });
  }

  renderModels(models: ModelInfo[]): void {
    this.populateModelSelect(models);
    const list = document.getElementById('model-list');
    if (!list) return;

    const currentConfig = store.getConfig();
    list.innerHTML = '';

    if (models.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'model-item';
      empty.textContent = 'No models installed. Use "Add model" to upload a .onnx checkpoint.';
      list.appendChild(empty);
      return;
    }

    for (const model of models) {
      const item = document.createElement('div');
      item.className = 'model-item';
      if (model.name === currentConfig.model.name) {
        item.classList.add('active');
      }

      const meta = document.createElement('div');
      meta.className = 'meta';
      const title = document.createElement('strong');
      title.textContent = model.name;
      const badge = document.createElement('span');
      badge.className = 'badge';
      badge.textContent = this.formatBytes(model.size_bytes);
      meta.appendChild(title);
      meta.appendChild(badge);

      const actions = document.createElement('div');
      actions.className = 'actions';
      const deleteBtn = document.createElement('button');
      deleteBtn.className = 'btn danger';
      deleteBtn.type = 'button';
      const isActive = model.name === currentConfig.model.name;
      deleteBtn.textContent = isActive ? 'Remove (active)' : 'Remove';
      if (model.immutable) {
        deleteBtn.textContent = 'Built-in';
        deleteBtn.classList.remove('danger');
        deleteBtn.classList.add('disabled');
        deleteBtn.disabled = true;
        deleteBtn.title = 'Built-in models cannot be removed';
      } else {
        deleteBtn.disabled = models.length === 1 && isActive;
      }
      if (!model.immutable) {
        deleteBtn.addEventListener('click', async () => {
          if (!confirm(`Remove model \"${model.name}\"?`)) {
            return;
          }
          try {
            const updated = await apiService.deleteModel(model.filename);
            store.setModels(updated);
          } catch (error) {
            console.error('Failed to delete model:', error);
            alert(`Failed to delete model: ${error}`);
          }
        });
      }
      actions.appendChild(deleteBtn);

      item.appendChild(meta);
      item.appendChild(actions);
      list.appendChild(item);
    }
  }

  sync(config: AnonymizerConfig): void {
    this.setValue('model-select', config.model.name);
    this.setValue('blur-type', config.blur.type);
    this.setValue('blur-strength', config.blur.strength);
    this.setValue('plate-threshold', config.detection.plate_threshold);
    this.setValue('face-threshold', config.detection.face_threshold);
    this.setValue('batch-size', config.detection.batch_size);
    this.setValue('use-sahi', config.detection.use_sahi);
    this.setValue('inference-size', config.detection.inference_size);
    this.setValue('sahi-overlap', config.detection.sahi_overlap_ratio);
    this.updateBatchSizeLock(config.model.name);
    this.setValue('tracker-type', config.tracking.type);
    this.setValue('track-offline-linker', config.tracking.use_offline_linker);

    const params = config.tracking.params;
    this.setValue('track-distance-gate', params.distance_gate);
    this.setValue('track-distance-high', params.distance_gate_hi);
    this.setValue('track-distance-low', params.distance_gate_lo);
    this.setValue('track-confirm-after', params.confirm_after_N);
    this.setValue('track-max-misses', params.max_misses_M);
    this.setValue('track-low-score-pool', params.use_low_score_pool);
    this.setValue('tracker-ema-alpha', params.ema_alpha);
    this.setValue('track-high-thresh', params.high_thresh);
    this.setValue('track-low-thresh', params.low_thresh);
    this.setValue('track-cam-motion', params.cam_motion_comp);
    this.setValue('track-flow-backend', params.flow_backend);
    this.setValue('track-use-visual', params.use_visual_tracker);
    this.setValue('track-vt-backend', params.vt_backend);
    this.setValue('track-vt-max-age', params.vt_max_age);
    this.setValue('track-bbox-dilate', params.bbox_dilate_pct);
    this.setValue('track-temporal-alpha', params.temporal_smooth_alpha);
    this.setValue('track-drift-gate', params.drift_gate);

    this.setValue('video-codec', config.video.codec);

    const qualityOverride = document.getElementById('video-quality-override') as HTMLInputElement | null;
    const qualitySlider = document.getElementById('video-quality') as HTMLInputElement | null;
    const qualityLabel = document.getElementById('video-quality-value');
    if (qualityOverride && qualitySlider) {
      if (config.video.quality === null || config.video.quality === undefined) {
        qualityOverride.checked = false;
        qualitySlider.disabled = true;
      } else {
        qualityOverride.checked = true;
        qualitySlider.disabled = false;
        qualitySlider.value = String(config.video.quality);
      }
      if (qualityLabel) {
        qualityLabel.textContent = qualityOverride.checked ? qualitySlider.value : 'auto';
      }
    }

    this.setValue('video-quality', config.video.quality ?? 23);
    this.setValue('debug-mode', config.debug);
    this.setValue('log-level', config.log_level);

    this.updateParamVisibility();
    this.refreshValueBadges();
    this.updateDebugConfig();
  }

  updateDebugConfig(): void {
    const debugConfigElement = document.getElementById('debug-config');
    if (!debugConfigElement) return;
    debugConfigElement.textContent = JSON.stringify(store.getConfig(), null, 2);
  }

  updateParamVisibility(forcedType?: string): void {
    const trackerType = forcedType ?? store.getConfig().tracking.type;
    document.querySelectorAll<HTMLElement>('[data-scope]').forEach(el => {
      const scope = el.getAttribute('data-scope');
      if (!scope) return;
      const visible = scope.split(',').map(s => s.trim()).includes(trackerType) || scope.includes('all');
      el.style.display = visible ? '' : 'none';
    });
  }

  private bindSectionField<T>(
    elementId: string,
    section: keyof AnonymizerConfig,
    field: string,
    parser: Parser<T>,
    post?: () => void,
  ): void {
    const el = document.getElementById(elementId) as HTMLInputElement | HTMLSelectElement | null;
    if (!el) return;
    const handler = () => {
      const raw = this.extractValue(el);
      const value = parser(raw);
      const current = store.getConfig();
      const sectionValue = { ...(current as any)[section], [field]: value };
      store.updateConfig({ [section]: sectionValue } as Partial<AnonymizerConfig>);
      post?.();
      this.refreshValueBadges();
    };
    el.addEventListener('input', handler);
    el.addEventListener('change', handler);
  }

  private bindTrackingParam<T>(
    elementId: string,
    param: keyof TrackerParams,
    parser: Parser<T>,
  ): void {
    const el = document.getElementById(elementId) as HTMLInputElement | HTMLSelectElement | null;
    if (!el) return;
    const handler = () => {
      const raw = this.extractValue(el);
      const value = parser(raw);
      const current = store.getConfig();
      const params = { ...current.tracking.params, [param]: value };
      store.updateConfig({ tracking: { ...current.tracking, params } } as Partial<AnonymizerConfig>);
      this.refreshValueBadges();
    };
    el.addEventListener('input', handler);
    el.addEventListener('change', handler);
  }

  private bindGlobalCheckbox(elementId: string, field: 'debug'): void {
    const el = document.getElementById(elementId) as HTMLInputElement | null;
    if (!el) return;
    el.addEventListener('change', () => {
      store.updateConfig({ [field]: el.checked } as Partial<AnonymizerConfig>);
      this.updateDebugConfig();
    });
  }

  private bindGlobalSelect(elementId: string, field: 'log_level'): void {
    const el = document.getElementById(elementId) as HTMLSelectElement | null;
    if (!el) return;
    el.addEventListener('change', () => {
      store.updateConfig({ [field]: el.value as any } as Partial<AnonymizerConfig>);
      this.updateDebugConfig();
    });
  }

  private extractValue(el: HTMLInputElement | HTMLSelectElement): string | boolean {
    if (el instanceof HTMLInputElement && el.type === 'checkbox') {
      return el.checked;
    }
    return el.value;
  }

  private setupQualityOverride(): void {
    const qualityOverride = document.getElementById('video-quality-override') as HTMLInputElement | null;
    const qualitySlider = document.getElementById('video-quality') as HTMLInputElement | null;
    const qualityLabel = document.getElementById('video-quality-value');
    if (!qualityOverride || !qualitySlider) return;

    const applyQualityState = () => {
      const currentVideo = store.getConfig().video;
      if (qualityOverride.checked) {
        qualitySlider.disabled = false;
        const parsed = parseInt(qualitySlider.value, 10);
        store.updateConfig({ video: { ...currentVideo, quality: parsed } } as Partial<AnonymizerConfig>);
      } else {
        qualitySlider.disabled = true;
        store.updateConfig({ video: { ...currentVideo, quality: null } } as Partial<AnonymizerConfig>);
      }
      if (qualityLabel) {
        qualityLabel.textContent = qualityOverride.checked ? qualitySlider.value : 'auto';
      }
    };

    qualityOverride.addEventListener('change', applyQualityState);
    qualitySlider.addEventListener('input', applyQualityState);
    applyQualityState();
  }

  private attachValueBadgeListeners(): void {
    this.valueDisplayIds.forEach(id => {
      const el = document.getElementById(id);
      if (!el) return;
      el.addEventListener('input', () => this.refreshValueBadges());
    });
  }

  private refreshValueBadges(): void {
    this.valueDisplayIds.forEach(id => {
      const slider = document.getElementById(id) as HTMLInputElement | null;
      const label = document.getElementById(`${id}-value`);
      if (slider && label) {
        label.textContent = slider.value;
      }
    });
  }

  private onTrackerTypeChanged(): void {
    const cfg = store.getConfig();
    const preset = TRACKER_PRESETS[cfg.tracking.type];
    if (!preset) {
      this.updateParamVisibility();
      return;
    }
    store.updateConfig({
      tracking: {
        type: cfg.tracking.type,
        use_offline_linker: cfg.tracking.use_offline_linker,
        params: { ...preset },
      },
    } as Partial<AnonymizerConfig>);
    this.updateParamVisibility(cfg.tracking.type);
    this.sync(store.getConfig());
  }

  private populateTrackerSelect(types: string[]): void {
    const trackerSelect = document.getElementById('tracker-type') as HTMLSelectElement | null;
    if (!trackerSelect) return;
    trackerSelect.innerHTML = '';
    for (const type of types) {
      const opt = document.createElement('option');
      opt.value = type;
      opt.textContent = type;
      trackerSelect.appendChild(opt);
    }
  }

  private populateModelSelect(models: ModelInfo[]): void {
    const modelSelect = document.getElementById('model-select') as HTMLSelectElement | null;
    if (!modelSelect) return;
    const current = store.getConfig();
    modelSelect.innerHTML = '';
    for (const model of models) {
      const opt = document.createElement('option');
      opt.value = model.name;
      opt.textContent = model.name;
      if (model.name === current.model.name) {
        opt.selected = true;
      }
      modelSelect.appendChild(opt);
    }
  }

  private setupModelManager(): void {
    const uploadInput = document.getElementById('model-upload-input') as HTMLInputElement | null;
    const uploadButton = document.getElementById('model-upload-button') as HTMLButtonElement | null;

    uploadButton?.addEventListener('click', () => uploadInput?.click());

    uploadInput?.addEventListener('change', async () => {
      const file = uploadInput.files?.[0];
      if (!file) return;
      try {
        const models = await apiService.uploadModel(file);
        store.setModels(models);
      } catch (error) {
        console.error('Failed to upload model:', error);
        alert(`Failed to upload model: ${error}`);
      } finally {
        uploadInput.value = '';
      }
    });
  }

  private isStaticBatchModel(name: string | null): boolean {
    return typeof name === 'string' && name.endsWith('_b1');
  }

  private updateBatchSizeLock(modelName: string | null): void {
    const locked = this.isStaticBatchModel(modelName);
    const slider = document.getElementById('batch-size') as HTMLInputElement | null;
    const badge = document.getElementById('batch-size-value');
    const hint = document.getElementById('batch-size-lock-hint');
    if (slider) {
      slider.disabled = locked;
      if (locked) {
        slider.value = '1';
        if (badge) {
          badge.textContent = '1';
        }
        const current = store.getConfig();
        if (current.detection.batch_size !== 1) {
          store.updateConfig({
            detection: { ...current.detection, batch_size: 1 },
          } as Partial<AnonymizerConfig>);
        }
      }
    }
    if (hint) {
      hint.classList.toggle('hidden', !locked);
    }
  }

  private applyRange(elementId: string, range?: [number, number]): void {
    if (!range) return;
    const input = document.getElementById(elementId) as HTMLInputElement | null;
    if (!input || input.type !== 'range') return;
    input.min = String(range[0]);
    input.max = String(range[1]);
  }

  private setValue(elementId: string, value: string | number | boolean | null | undefined): void {
    if (value === undefined) return;
    const el = document.getElementById(elementId) as HTMLInputElement | HTMLSelectElement | null;
    if (!el) return;
    if (el instanceof HTMLInputElement && el.type === 'checkbox') {
      el.checked = Boolean(value);
    } else if (value !== null) {
      el.value = String(value);
    }
    const label = document.getElementById(`${elementId}-value`);
    if (label && value !== null && value !== undefined) {
      label.textContent = String(value);
    }
  }

  private formatBytes(bytes: number): string {
    if (!Number.isFinite(bytes)) return '';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
}
