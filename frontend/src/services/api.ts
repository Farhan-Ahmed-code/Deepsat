import { ProjectInfo, DatasetInfo, ModelMetricsData, PredictionResult, SampleImageItem } from '../types';

const API_BASE = '/api';

export async function fetchHealth(): Promise<{ status: string; models: any }> {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error('Health check failed');
    return await res.json();
  } catch (err) {
    return {
      status: 'offline',
      models: { cnn: { available: true }, resnet18: { available: true } }
    };
  }
}

export async function fetchProjectInfo(): Promise<ProjectInfo> {
  const res = await fetch(`${API_BASE}/project-info`);
  if (!res.ok) throw new Error('Failed to fetch project info');
  return res.json();
}

export async function fetchDatasetInfo(): Promise<DatasetInfo> {
  const res = await fetch(`${API_BASE}/dataset-info`);
  if (!res.ok) throw new Error('Failed to fetch dataset info');
  return res.json();
}

export async function fetchModelsInfo(): Promise<any> {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error('Failed to fetch models info');
  return res.json();
}

export async function fetchModelMetrics(): Promise<ModelMetricsData> {
  const res = await fetch(`${API_BASE}/model-metrics`);
  if (!res.ok) throw new Error('Failed to fetch model metrics');
  return res.json();
}

export async function fetchSampleImages(): Promise<SampleImageItem[]> {
  const res = await fetch(`${API_BASE}/sample-images?samples_per_class=3`);
  if (!res.ok) throw new Error('Failed to fetch sample images');
  const data = await res.json();
  return data.samples;
}

export async function predictImage(
  modelName: 'resnet18' | 'cnn',
  file?: File,
  samplePath?: string
): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append('model_name', modelName);
  
  if (file) {
    formData.append('file', file);
  } else if (samplePath) {
    formData.append('sample_path', samplePath);
  } else {
    throw new Error('Must provide either an image file or a sample dataset path');
  }

  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: 'Prediction failed' }));
    throw new Error(errorData.detail || 'Prediction request failed');
  }

  return res.json();
}

