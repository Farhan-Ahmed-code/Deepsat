export interface Student {
  name: string;
  role: string;
  programme: string;
}

export interface FacultyGuide {
  name: string;
  designation: string;
  department: string;
  institution: string;
}

export interface ProjectInfo {
  project_title: string;
  course: string;
  programme: string;
  semester: string;
  institution: {
    university: string;
    school: string;
    department: string;
  };
  students: Student[];
  faculty_guide: FacultyGuide;
  abstract: string;
  problem_statement: string;
  objectives: string[];
  methodology_steps: Array<{
    step: number;
    title: string;
    desc: string;
  }>;
  future_scope: Array<{
    title: string;
    desc: string;
  }>;
}

export interface EuroSATClass {
  className: string;
  index: number;
  train: number;
  val: number;
  test: number;
  total: number;
  desc: string;
}

export interface DatasetInfo {
  dataset_name: string;
  source: string;
  total_images: number;
  num_classes: number;
  image_resolution: string;
  channels: number;
  color_space: string;
  splits: {
    train: { count: number; percentage: number };
    val: { count: number; percentage: number };
    test: { count: number; percentage: number };
  };
  normalization: {
    type: string;
    mean: number[];
    std: number[];
  };
  augmentation: string[];
  classes: EuroSATClass[];
}

export interface ModelSummary {
  name: string;
  accuracy: number;
  loss: number;
  precision: number;
  recall: number;
  f1_score: number;
  misclassified: number;
  correct: number;
  error_rate: number;
}

export interface ModelMetricsData {
  summary: {
    test_samples: number;
    cnn: ModelSummary;
    resnet18: ModelSummary;
    improvement: {
      accuracy_diff: string;
      loss_reduction: string;
      f1_diff: string;
      error_reduction_count: number;
      error_reduction_percentage: string;
    };
  };
  epochs_data: Array<{
    epoch: number;
    cnn_accuracy?: number;
    cnn_val_accuracy?: number;
    cnn_loss?: number;
    cnn_val_loss?: number;
    resnet_accuracy?: number;
    resnet_val_accuracy?: number;
    resnet_loss?: number;
    resnet_val_loss?: number;
  }>;
  per_class_comparison: Array<{
    className: string;
    support: number;
    cnn_precision: number;
    resnet_precision: number;
    cnn_recall: number;
    resnet_recall: number;
    cnn_f1: number;
    resnet_f1: number;
    f1_improvement: number;
  }>;
  confusion_matrices: {
    classes?: string[];
    cnn?: number[][];
    resnet18?: number[][];
  };
  classes: string[];
}

export interface ClassProbability {
  className: string;
  index: number;
  probability: number;
  percentage: number;
}

export interface PredictionResult {
  status: string;
  predicted_class: string;
  class_index: number;
  confidence: number;
  confidence_percentage: number;
  model_used: {
    key: string;
    name: string;
    parameters: number;
  };
  image_metadata: {
    name: string;
    source: string;
    original_dimensions: string;
    processed_dimensions: string;
  };
  latency: {
    inference_time_ms: number;
    total_time_ms: number;
  };
  probabilities: ClassProbability[];
  ranked_probabilities: ClassProbability[];
}

export interface SampleImageItem {
  className: string;
  filename: string;
  path: string;
  url: string;
}

