import React from 'react';
import {
  Database,
  Search,
  Sliders,
  RefreshCw,
  Split,
  Cpu,
  Layers,
  GraduationCap,
  BarChart2,
  Code2
} from 'lucide-react';

export const Methodology: React.FC = () => {
  const steps = [
    {
      num: 1,
      title: 'Dataset Collection & Verification',
      icon: Database,
      tag: 'EuroSAT RGB',
      desc: 'Downloaded and extracted the EuroSAT dataset containing 27,000 georeferenced Sentinel-2 satellite scenes across 10 distinct Land Use and Land Cover (LULC) classes.',
      details: 'Resolution: 64 x 64 pixels, 3 channels (RGB), 10 classes with approximately 2,000 to 3,000 images per category.'
    },
    {
      num: 2,
      title: 'Exploratory Data Analysis (EDA)',
      icon: Search,
      tag: 'Statistical Profiling',
      desc: 'Computed channel-wise statistics across all 27,000 satellite images to determine exact per-band mean and standard deviation vectors required for optimal numerical convergence.',
      details: 'Mean: R=0.3484, G=0.3824, B=0.4085 | Std Dev: R=0.2102, G=0.1413, B=0.1186'
    },
    {
      num: 3,
      title: 'Image Preprocessing Pipeline',
      icon: Sliders,
      tag: 'Z-Score Normalization',
      desc: 'Standardized all imagery by converting raw uint8 [0-255] pixels to float32 [0.0-1.0], followed by channel-wise Z-score normalization: Z = (X - mean) / std.',
      details: 'Centers channel activations around zero mean with unit variance to eliminate illumination gradient disparities across satellite swaths.'
    },
    {
      num: 4,
      title: 'Data Augmentation (Training-Only)',
      icon: RefreshCw,
      tag: 'Invariance Regularization',
      desc: 'Applied stochastic spatial augmentations exclusively to the training pipeline to build rotational and directional invariance without modifying test integrity.',
      details: 'Random Horizontal Flip (p=0.5), Random Vertical Flip (p=0.5), Random Rotation up to +/-18 degrees.'
    },
    {
      num: 5,
      title: 'Stratified Dataset Splitting',
      icon: Split,
      tag: '70 / 20 / 10 Partition',
      desc: 'Executed stratified partitioning based on ground truth labels to ensure identical class distributions across all three splits.',
      details: 'Training: 18,900 images (70%) | Validation: 5,400 images (20%) | Independent Test: 2,700 images (10%)'
    },
    {
      num: 6,
      title: 'High-Performance tf.data Pipeline',
      icon: Code2,
      tag: 'Parallel I/O & Prefetching',
      desc: "Leveraged TensorFlow's native tf.data.Dataset API for high-throughput disk I/O, multi-threaded mapping, buffer shuffling, and AUTOTUNE memory prefetching.",
      details: 'num_parallel_calls=AUTOTUNE, buffer_size=len(split), batch_size=32, prefetch(AUTOTUNE) to maximize GPU/CPU overlap.'
    },
    {
      num: 7,
      title: 'Custom Baseline CNN Construction',
      icon: Cpu,
      tag: '111.9K Parameters',
      desc: 'Engineered a custom sequential CNN architecture composed of 3 convolutional blocks (32, 64, 128 filters) with Batch Normalization, ReLU, MaxPool, Global Average Pooling, and Dropout.',
      details: 'Designed as an ultra-compact, low-latency baseline calibrated for 64x64 satellite patches.'
    },
    {
      num: 8,
      title: 'ResNet-18 Implementation (From Scratch)',
      icon: Layers,
      tag: '11.19M Parameters',
      desc: 'Constructed the full 18-layer Residual Network (ResNet-18) entirely from scratch using the Keras functional API without ImageNet pretrained weights.',
      details: 'Features a 7x7 stem conv, 4 residual stages with 8 basic residual blocks, identity shortcuts, and 1x1 projection convolutions for downsampling.'
    },
    {
      num: 9,
      title: 'Rigorous Model Training Protocol',
      icon: GraduationCap,
      tag: 'Adam Optimizer',
      desc: 'Trained both models under identical training conditions for 20 epochs using the Adam optimizer (lr=0.001) and Sparse Categorical Crossentropy.',
      details: 'Utilized ModelCheckpoint (saving highest validation accuracy weights) and ReduceLROnPlateau (factor=0.2, patience=3, min_lr=1e-6).'
    },
    {
      num: 10,
      title: 'Comprehensive Empirical Evaluation',
      icon: BarChart2,
      tag: '2,700 Test Samples',
      desc: 'Benchmarked both trained models on the held-out test split, generating classification reports, macro precision, recall, F1-score, and confusion matrices.',
      details: 'ResNet-18 achieved 94.04% accuracy vs CNN 91.11% (+2.93%), reducing misclassifications from 240 to 161 (32.9% error reduction).'
    },
    {
      num: 11,
      title: 'Full-Stack Deployment & Demonstration',
      icon: Code2,
      tag: 'FastAPI + React Vite',
      desc: 'Integrated the pre-warmed Keras inference engines into an asynchronous FastAPI REST backend and a React Vite dashboard for university project examination.',
      details: 'Provides sub-50ms inference latency, real-time Softmax probability ranking, and one-click testing across all 10 land cover categories.'
    }
  ];

  return (
    <div className="space-y-6 pb-12">
      {/* Page Header */}
      <div className="bg-white border border-slate-200 rounded-lg p-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h2 className="text-xl font-bold text-slate-900">Research Methodology</h2>
            <p className="text-sm text-slate-500 mt-1">
              11-stage end-to-end pipeline — from raw satellite image ingestion to live model deployment.
            </p>
          </div>
          <div className="shrink-0 bg-slate-50 border border-slate-200 rounded-lg px-4 py-2 text-sm">
            <span className="text-slate-500 text-xs block">Reproducibility Seed</span>
            <span className="font-mono font-semibold text-slate-800">SEED = 42 (NumPy & TF)</span>
          </div>
        </div>
      </div>

      {/* 11-Step Pipeline */}
      <div className="space-y-3">
        {steps.map((step) => {
          const Icon = step.icon;
          return (
            <div
              key={step.num}
              className="bg-white border border-slate-200 rounded-lg p-4 flex flex-col md:flex-row md:items-start gap-4"
            >
              {/* Step Number & Icon */}
              <div className="flex items-center gap-3 md:w-64 shrink-0">
                <div className="w-8 h-8 rounded-full bg-blue-600 text-white flex items-center justify-center text-sm font-bold shrink-0">
                  {step.num}
                </div>
                <div>
                  <h4 className="text-sm font-semibold text-slate-900">{step.title}</h4>
                  <span className="inline-block mt-0.5 text-xs text-blue-700 bg-blue-50 border border-blue-200 rounded px-1.5 py-0.5">
                    {step.tag}
                  </span>
                </div>
              </div>

              {/* Description */}
              <div className="flex-1 space-y-1">
                <p className="text-sm text-slate-600 leading-relaxed">{step.desc}</p>
                <p className="text-xs text-slate-400">{step.details}</p>
              </div>
            </div>
          );
        })}
      </div>

      {/* Technical Detail Boxes */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Normalization Formula */}
        <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-3">
          <div className="flex items-center gap-2">
            <Sliders className="w-4 h-4 text-slate-400" />
            <h4 className="text-sm font-semibold text-slate-900">Preprocessing Normalization Formula</h4>
          </div>
          <p className="text-sm text-slate-600 leading-relaxed">
            Satellite images possess varying atmospheric lighting and soil reflectance.
            Channel-wise Z-score standardization is applied per pixel channel c in [R, G, B]:
          </p>
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 font-mono text-sm text-center text-slate-800">
            Z = ((Image / 255.0) - Channel_Mean) / Channel_Std
          </div>
          <div className="text-xs text-slate-500 font-mono space-y-1">
            <div>Mean: [0.3484, 0.3824, 0.4085] (Red, Green, Blue)</div>
            <div>Std Dev: [0.2102, 0.1413, 0.1186] (Red, Green, Blue)</div>
          </div>
        </div>

        {/* tf.data Pipeline */}
        <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-3">
          <div className="flex items-center gap-2">
            <Code2 className="w-4 h-4 text-slate-400" />
            <h4 className="text-sm font-semibold text-slate-900">TensorFlow tf.data Execution Pipeline</h4>
          </div>
          <p className="text-sm text-slate-600 leading-relaxed">
            To prevent GPU/CPU starvation during training epochs, the input pipeline asynchronously prefetches
            batches while backpropagation executes on current tensors:
          </p>
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 font-mono text-xs text-slate-700 space-y-1">
            <div>dataset = dataset.shuffle(buffer_size=N, seed=42)</div>
            <div>dataset = dataset.map(preprocess, num_parallel_calls=AUTOTUNE)</div>
            <div>dataset = dataset.batch(batch_size=32)</div>
            <div>dataset = dataset.prefetch(buffer_size=AUTOTUNE)</div>
          </div>
          <p className="text-xs text-slate-500">
            Guarantees zero I/O bottleneck across all 20 training epochs.
          </p>
        </div>
      </div>
    </div>
  );
};
