import React, { useState, useEffect, useRef } from 'react';
import {
  Upload,
  Cpu,
  Layers,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  Image as ImageIcon,
  BarChart2
} from 'lucide-react';
import { predictImage, fetchSampleImages } from '../services/api';
import { PredictionResult, SampleImageItem } from '../types';

interface LiveClassificationProps {
  initialSample?: SampleImageItem | null;
}

export const LiveClassification: React.FC<LiveClassificationProps> = ({ initialSample }) => {
  const [selectedModel, setSelectedModel] = useState<'resnet18' | 'cnn'>('resnet18');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [selectedSamplePath, setSelectedSamplePath] = useState<string | null>(null);
  const [selectedSampleName, setSelectedSampleName] = useState<string | null>(null);

  const [sampleImages, setSampleImages] = useState<SampleImageItem[]>([]);
  const [activeCategoryFilter, setActiveCategoryFilter] = useState<string>('All');

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    async function loadSamples() {
      try {
        const samples = await fetchSampleImages();
        setSampleImages(samples);
        if (initialSample) {
          handleSelectSample(initialSample);
        } else if (samples.length > 0) {
          const defaultSample = samples.find(s => s.className === 'SeaLake') || samples[0];
          handleSelectSample(defaultSample);
        }
      } catch (err) {
        console.error('Failed to load sample images:', err);
      }
    }
    loadSamples();
  }, [initialSample]);

  const handleSelectSample = (sample: SampleImageItem) => {
    setSelectedFile(null);
    setSelectedSamplePath(sample.path);
    setSelectedSampleName(sample.filename);
    setPreviewUrl(sample.url);
    setError(null);
    setResult(null);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        setError('Please select a valid image file (JPEG, PNG, TIFF).');
        return;
      }
      setSelectedFile(file);
      setSelectedSamplePath(null);
      setSelectedSampleName(file.name);
      setPreviewUrl(URL.createObjectURL(file));
      setError(null);
      setResult(null);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setSelectedSamplePath(null);
      setSelectedSampleName(file.name);
      setPreviewUrl(URL.createObjectURL(file));
      setError(null);
      setResult(null);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const runPrediction = async () => {
    if (!selectedFile && !selectedSamplePath) {
      setError('Please upload an image or select a sample satellite tile first.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await predictImage(
        selectedModel,
        selectedFile || undefined,
        selectedSamplePath || undefined
      );
      setResult(res);
    } catch (err: any) {
      setError(err.message || 'Inference failed. Check backend connection.');
    } finally {
      setLoading(false);
    }
  };

  const categories = ['All', 'Forest', 'SeaLake', 'Highway', 'River', 'Residential', 'AnnualCrop', 'Industrial'];
  const displayedSamples = activeCategoryFilter === 'All'
    ? sampleImages
    : sampleImages.filter(s => s.className === activeCategoryFilter);

  return (
    <div className="space-y-6 pb-12">
      {/* Page Header */}
      <div className="bg-white border border-slate-200 rounded-lg p-6">
        <h2 className="text-xl font-bold text-slate-900">Live Satellite Image Classification</h2>
        <p className="text-sm text-slate-500 mt-1">
          Select a sample EuroSAT tile or upload your own image, choose a model, and run inference to see real-time classification results.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Input (5 cols) */}
        <div className="lg:col-span-5 space-y-4">

          {/* Model Selection */}
          <div className="bg-white border border-slate-200 rounded-lg p-5">
            <h3 className="text-sm font-semibold text-slate-700 mb-3">Select Model</h3>
            <div className="flex gap-3">
              <button
                onClick={() => setSelectedModel('resnet18')}
                className={`flex-1 flex items-center justify-center gap-2 py-2.5 px-3 rounded-md border text-sm font-medium transition-colors ${
                  selectedModel === 'resnet18'
                    ? 'bg-blue-600 text-white border-blue-600'
                    : 'bg-white text-slate-600 border-slate-300 hover:border-slate-400'
                }`}
              >
                <Layers className="w-4 h-4" />
                ResNet-18 (94.04%)
              </button>
              <button
                onClick={() => setSelectedModel('cnn')}
                className={`flex-1 flex items-center justify-center gap-2 py-2.5 px-3 rounded-md border text-sm font-medium transition-colors ${
                  selectedModel === 'cnn'
                    ? 'bg-blue-600 text-white border-blue-600'
                    : 'bg-white text-slate-600 border-slate-300 hover:border-slate-400'
                }`}
              >
                <Cpu className="w-4 h-4" />
                Baseline CNN (91.11%)
              </button>
            </div>
          </div>

          {/* Image Upload / Preview */}
          <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-slate-700 flex items-center gap-2">
                <Upload className="w-4 h-4 text-slate-400" />
                Image Input
              </h3>
              {previewUrl && (
                <button
                  onClick={() => {
                    setSelectedFile(null);
                    setSelectedSamplePath(null);
                    setPreviewUrl(null);
                    setResult(null);
                  }}
                  className="text-xs text-slate-400 hover:text-rose-500 transition-colors"
                >
                  Clear
                </button>
              )}
            </div>

            {/* Drop Zone */}
            <div
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-lg p-4 flex flex-col items-center justify-center text-center cursor-pointer transition-colors min-h-[200px] ${
                previewUrl
                  ? 'border-blue-300 bg-blue-50'
                  : 'border-slate-300 hover:border-slate-400 bg-slate-50'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileUpload}
                className="hidden"
              />

              {previewUrl ? (
                <div className="flex flex-col items-center gap-3">
                  <div className="w-32 h-32 rounded-lg overflow-hidden border border-slate-200 shadow-sm">
                    <img
                      src={previewUrl}
                      alt="Input Preview"
                      className="w-full h-full object-cover"
                      style={{ imageRendering: 'pixelated' }}
                    />
                  </div>
                  <div className="text-center">
                    <p className="text-sm font-medium text-slate-700 truncate max-w-[220px]">
                      {selectedSampleName || 'Uploaded Image'}
                    </p>
                    <p className="text-xs text-slate-400 mt-0.5">Resized to 64 x 64 x 3 RGB on backend</p>
                  </div>
                  <p className="text-xs text-blue-600">Click to upload a different image</p>
                </div>
              ) : (
                <div className="space-y-2 py-4">
                  <div className="w-10 h-10 rounded-full bg-slate-100 border border-slate-200 flex items-center justify-center mx-auto">
                    <Upload className="w-5 h-5 text-slate-400" />
                  </div>
                  <div className="text-sm">
                    <span className="font-medium text-blue-600">Click to upload</span>
                    <span className="text-slate-500"> or drag and drop</span>
                  </div>
                  <p className="text-xs text-slate-400">JPG, PNG, or TIFF satellite image</p>
                </div>
              )}
            </div>

            {/* Predict Button */}
            <button
              onClick={runPrediction}
              disabled={loading || !previewUrl}
              className={`w-full py-2.5 px-4 rounded-md font-semibold text-sm flex items-center justify-center gap-2 transition-colors ${
                loading || !previewUrl
                  ? 'bg-slate-100 text-slate-400 cursor-not-allowed border border-slate-200'
                  : 'bg-blue-600 hover:bg-blue-700 text-white'
              }`}
            >
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Running Inference...
                </>
              ) : (
                <>
                  <CheckCircle2 className="w-4 h-4" />
                  Run Classification with {selectedModel === 'resnet18' ? 'ResNet-18' : 'Baseline CNN'}
                </>
              )}
            </button>

            {error && (
              <div className="p-3 rounded-md bg-rose-50 border border-rose-200 text-rose-700 text-xs flex items-center gap-2">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>{error}</span>
              </div>
            )}
          </div>

          {/* Sample Images Picker */}
          <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-3">
            <h3 className="text-sm font-semibold text-slate-700 flex items-center gap-2">
              <ImageIcon className="w-4 h-4 text-slate-400" />
              1-Click Test Samples (EuroSAT)
            </h3>

            {/* Category Filter Pills */}
            <div className="flex gap-1.5 flex-wrap">
              {categories.map((cat) => (
                <button
                  key={cat}
                  onClick={() => setActiveCategoryFilter(cat)}
                  className={`px-2.5 py-1 rounded-full text-xs font-medium border transition-colors ${
                    activeCategoryFilter === cat
                      ? 'bg-blue-600 text-white border-blue-600'
                      : 'bg-white text-slate-600 border-slate-300 hover:border-slate-400'
                  }`}
                >
                  {cat}
                </button>
              ))}
            </div>

            {/* Thumbnail Grid */}
            <div className="grid grid-cols-5 gap-2 max-h-48 overflow-y-auto">
              {displayedSamples.map((sample, idx) => {
                const isSelected = selectedSamplePath === sample.path;
                return (
                  <button
                    key={idx}
                    onClick={() => handleSelectSample(sample)}
                    title={sample.className}
                    className={`relative rounded-md overflow-hidden border-2 transition-all ${
                      isSelected
                        ? 'border-blue-500 ring-2 ring-blue-200'
                        : 'border-slate-200 hover:border-slate-300'
                    }`}
                  >
                    <div className="w-full aspect-square bg-slate-100">
                      <img
                        src={sample.url}
                        alt={sample.filename}
                        className="w-full h-full object-cover"
                        style={{ imageRendering: 'pixelated' }}
                      />
                    </div>
                    <p className="text-[9px] text-slate-500 text-center truncate px-0.5 py-0.5 bg-white">
                      {sample.className.substring(0, 6)}
                    </p>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right Column: Results (7 cols) */}
        <div className="lg:col-span-7">
          {result ? (
            <div className="space-y-4">
              {/* Primary Result Card */}
              <div className="bg-white border border-slate-200 rounded-lg p-6">
                <p className="text-xs font-medium text-slate-500 uppercase tracking-wider mb-1">Predicted Land Cover</p>
                <div className="flex items-center justify-between flex-wrap gap-4">
                  <h3 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
                    <CheckCircle2 className="w-6 h-6 text-green-500" />
                    {result.predicted_class}
                  </h3>
                  <div className="text-right">
                    <p className="text-xs text-slate-500">Confidence</p>
                    <p className="text-2xl font-bold text-blue-700 font-mono">
                      {result.confidence_percentage.toFixed(1)}%
                    </p>
                  </div>
                </div>

                <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-3 pt-4 border-t border-slate-100">
                  <div>
                    <p className="text-xs text-slate-500">Model</p>
                    <p className="text-sm font-semibold text-slate-800 truncate">{result.model_used.name}</p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-500">Parameters</p>
                    <p className="text-sm font-semibold text-slate-800">{result.model_used.parameters.toLocaleString()}</p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-500">Inference Time</p>
                    <p className="text-sm font-semibold text-slate-800">{result.latency.inference_time_ms} ms</p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-500">Class Index</p>
                    <p className="text-sm font-semibold text-slate-800">#{result.class_index}</p>
                  </div>
                </div>
              </div>

              {/* Probability Distribution */}
              <div className="bg-white border border-slate-200 rounded-lg p-6">
                <h4 className="text-sm font-semibold text-slate-700 flex items-center gap-2 mb-4">
                  <BarChart2 className="w-4 h-4 text-slate-400" />
                  Class Probability Distribution (Softmax Output)
                </h4>

                <div className="space-y-3">
                  {result.ranked_probabilities.map((item, rank) => {
                    const isTop = rank === 0;
                    return (
                      <div key={item.className}>
                        <div className="flex items-center justify-between text-xs mb-1">
                          <span className={`font-medium ${isTop ? 'text-blue-700' : 'text-slate-600'}`}>
                            {item.className}
                          </span>
                          <span className={`font-mono font-semibold ${isTop ? 'text-blue-700' : 'text-slate-500'}`}>
                            {item.percentage.toFixed(2)}%
                          </span>
                        </div>
                        <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-500 ${
                              isTop ? 'bg-blue-600' : rank === 1 ? 'bg-blue-300' : 'bg-slate-300'
                            }`}
                            style={{ width: `${Math.max(item.percentage, 0.5)}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          ) : (
            /* Empty State */
            <div className="bg-white border border-slate-200 rounded-lg p-8 flex flex-col items-center justify-center text-center space-y-4 min-h-[420px]">
              <div className="w-14 h-14 rounded-full bg-slate-100 border border-slate-200 flex items-center justify-center">
                <ImageIcon className="w-7 h-7 text-slate-400" />
              </div>

              <div>
                <h4 className="text-base font-semibold text-slate-800">Awaiting Classification</h4>
                <p className="text-sm text-slate-500 mt-1 max-w-sm">
                  Select a sample tile on the left or upload a custom image, then click <strong>Run Classification</strong> to see the result.
                </p>
              </div>

              <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 text-left text-sm text-slate-600 space-y-2 max-w-xs w-full">
                <p className="font-semibold text-slate-700">Quick Test Steps:</p>
                <div className="flex items-center gap-2">
                  <span className="w-5 h-5 rounded-full bg-blue-600 text-white text-xs flex items-center justify-center font-bold shrink-0">1</span>
                  <span>Click any sample tile in the grid</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-5 h-5 rounded-full bg-blue-600 text-white text-xs flex items-center justify-center font-bold shrink-0">2</span>
                  <span>Select ResNet-18 or Baseline CNN</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-5 h-5 rounded-full bg-blue-600 text-white text-xs flex items-center justify-center font-bold shrink-0">3</span>
                  <span>Click Run Classification</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
