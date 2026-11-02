import React, { useState } from 'react';
import { Cpu, Layers, ArrowDown, CheckCircle2, Info } from 'lucide-react';

export const ModelArchitectures: React.FC = () => {
  const [selectedView, setSelectedView] = useState<'both' | 'cnn' | 'resnet18'>('both');

  return (
    <div className="space-y-6 pb-12">
      {/* Top Header */}
      <div className="bg-white border border-slate-200 rounded-lg p-6 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <span className="text-xs font-semibold text-blue-600 uppercase tracking-wider">
            Architecture Specification
          </span>
          <h2 className="text-xl font-bold text-slate-900 mt-1">
            Model Architectures & Parameter Profiling
          </h2>
          <p className="text-xs text-slate-600 max-w-2xl mt-1">
            A comparison between a custom 3-block Baseline CNN (111.9K parameters) and an 18-layer Deep Residual 
            Network (ResNet-18, 11.19M parameters) implemented from scratch in TensorFlow/Keras.
          </p>
        </div>

        {/* View Toggle */}
        <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg border border-slate-200 self-start md:self-auto shrink-0">
          <button
            onClick={() => setSelectedView('both')}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
              selectedView === 'both' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Show Both
          </button>
          <button
            onClick={() => setSelectedView('cnn')}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
              selectedView === 'cnn' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Baseline CNN
          </button>
          <button
            onClick={() => setSelectedView('resnet18')}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
              selectedView === 'resnet18' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            ResNet-18
          </button>
        </div>
      </div>

      {/* Scientific Clarification Note */}
      <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg text-xs text-blue-900 flex items-start gap-2.5">
        <Info className="w-4 h-4 text-blue-600 shrink-0 mt-0.5" />
        <div>
          <strong className="font-semibold">Note on Training Setup:</strong> ResNet-18 was implemented entirely from scratch in Keras and 
          <strong> was NOT initialized with ImageNet pretrained weights</strong>. Both models were trained from scratch under identical conditions 
          (Adam optimizer, learning rate 0.001, 20 epochs, EuroSAT dataset) to ensure a direct, fair architectural comparison.
        </div>
      </div>

      {/* Side-by-Side Model Sections */}
      <div className={`grid grid-cols-1 ${selectedView === 'both' ? 'lg:grid-cols-2' : ''} gap-6`}>
        {/* Model 1: Baseline CNN */}
        {(selectedView === 'both' || selectedView === 'cnn') && (
          <div className="bg-white border border-slate-200 rounded-lg p-6 shadow-sm space-y-5">
            <div className="flex items-center justify-between border-b border-slate-200 pb-3">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-md bg-blue-50 text-blue-600 flex items-center justify-center font-bold">
                  <Cpu className="w-4 h-4" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-slate-900">Custom Baseline CNN</h3>
                  <p className="text-xs text-slate-500">Sequential Convolutional Architecture</p>
                </div>
              </div>
              <span className="text-xs font-mono font-medium px-2 py-0.5 bg-slate-100 text-slate-700 rounded border border-slate-200">
                111,946 params
              </span>
            </div>

            {/* Parameter Counts */}
            <div className="grid grid-cols-3 gap-2 text-center text-xs">
              <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-md">
                <span className="text-[10px] text-slate-500 block uppercase">Total Parameters</span>
                <span className="text-sm font-bold text-slate-900 font-mono">111,946</span>
              </div>
              <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-md">
                <span className="text-[10px] text-slate-500 block uppercase">Trainable</span>
                <span className="text-sm font-bold text-blue-700 font-mono">111,498</span>
              </div>
              <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-md">
                <span className="text-[10px] text-slate-500 block uppercase">Non-Trainable</span>
                <span className="text-sm font-bold text-slate-600 font-mono">448</span>
              </div>
            </div>

            {/* Layer Details List */}
            <div className="space-y-2 text-xs">
              <h4 className="font-semibold text-slate-800 text-xs">Layer Sequence:</h4>

              <div className="p-2.5 bg-slate-50 border border-slate-200 rounded flex justify-between items-center">
                <span className="font-medium text-slate-800">Input Layer</span>
                <span className="font-mono text-slate-600 text-[11px]">(64, 64, 3)</span>
              </div>

              <div className="flex justify-center"><ArrowDown className="w-3.5 h-3.5 text-slate-400" /></div>

              <div className="p-3 bg-slate-50 border border-slate-200 rounded space-y-1">
                <div className="flex justify-between font-medium text-slate-900">
                  <span>Convolutional Block 1</span>
                  <span className="font-mono text-slate-600 text-[11px]">Output: (32, 32, 32)</span>
                </div>
                <p className="text-[11px] text-slate-600">
                  Conv2D (32 filters, 3×3, same) • Batch Normalization • ReLU • MaxPooling (2×2)
                </p>
              </div>

              <div className="flex justify-center"><ArrowDown className="w-3.5 h-3.5 text-slate-400" /></div>

              <div className="p-3 bg-slate-50 border border-slate-200 rounded space-y-1">
                <div className="flex justify-between font-medium text-slate-900">
                  <span>Convolutional Block 2</span>
                  <span className="font-mono text-slate-600 text-[11px]">Output: (16, 16, 64)</span>
                </div>
                <p className="text-[11px] text-slate-600">
                  Conv2D (64 filters, 3×3, same) • Batch Normalization • ReLU • MaxPooling (2×2)
                </p>
              </div>

              <div className="flex justify-center"><ArrowDown className="w-3.5 h-3.5 text-slate-400" /></div>

              <div className="p-3 bg-slate-50 border border-slate-200 rounded space-y-1">
                <div className="flex justify-between font-medium text-slate-900">
                  <span>Convolutional Block 3</span>
                  <span className="font-mono text-slate-600 text-[11px]">Output: (8, 8, 128)</span>
                </div>
                <p className="text-[11px] text-slate-600">
                  Conv2D (128 filters, 3×3, same) • Batch Normalization • ReLU • MaxPooling (2×2)
                </p>
              </div>

              <div className="flex justify-center"><ArrowDown className="w-3.5 h-3.5 text-slate-400" /></div>

              <div className="p-3 bg-blue-50/60 border border-blue-200 rounded space-y-1">
                <div className="flex justify-between font-semibold text-blue-900">
                  <span>Classification Head</span>
                  <span className="font-mono text-blue-700 text-[11px]">Output: 10 Classes</span>
                </div>
                <p className="text-[11px] text-slate-600">
                  Global Average Pooling • Dense (128 units, ReLU) • Dropout (0.30) • Dense (10 units, Softmax)
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Model 2: ResNet-18 */}
        {(selectedView === 'both' || selectedView === 'resnet18') && (
          <div className="bg-white border border-slate-200 rounded-lg p-6 shadow-sm space-y-5">
            <div className="flex items-center justify-between border-b border-slate-200 pb-3">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-md bg-blue-50 text-blue-600 flex items-center justify-center font-bold">
                  <Layers className="w-4 h-4" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-slate-900">ResNet-18</h3>
                  <p className="text-xs text-slate-500">Implemented from scratch in TensorFlow/Keras</p>
                </div>
              </div>
              <span className="text-xs font-mono font-medium px-2 py-0.5 bg-blue-50 text-blue-700 rounded border border-blue-200">
                11,191,242 params
              </span>
            </div>

            {/* Parameter Counts */}
            <div className="grid grid-cols-3 gap-2 text-center text-xs">
              <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-md">
                <span className="text-[10px] text-slate-500 block uppercase">Total Parameters</span>
                <span className="text-sm font-bold text-slate-900 font-mono">11,191,242</span>
              </div>
              <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-md">
                <span className="text-[10px] text-slate-500 block uppercase">Trainable</span>
                <span className="text-sm font-bold text-blue-700 font-mono">11,181,642</span>
              </div>
              <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-md">
                <span className="text-[10px] text-slate-500 block uppercase">Non-Trainable</span>
                <span className="text-sm font-bold text-slate-600 font-mono">9,600</span>
              </div>
            </div>

            {/* Residual Stages Sequence */}
            <div className="space-y-2 text-xs">
              <h4 className="font-semibold text-slate-800 text-xs">Stage Breakdown:</h4>

              <div className="p-3 bg-slate-50 border border-slate-200 rounded space-y-1">
                <div className="flex justify-between font-medium text-slate-900">
                  <span>Stage 0: Convolutional Stem</span>
                  <span className="font-mono text-slate-600 text-[11px]">Output: (16, 16, 64)</span>
                </div>
                <p className="text-[11px] text-slate-600">
                  Conv2D 7×7 (64 filters, stride 2) • Batch Normalization • ReLU • MaxPool 3×3 (stride 2)
                </p>
              </div>

              <div className="flex justify-center"><ArrowDown className="w-3.5 h-3.5 text-slate-400" /></div>

              <div className="p-3 bg-slate-50 border border-slate-200 rounded space-y-1">
                <div className="flex justify-between font-medium text-slate-900">
                  <span>Stage 1: Residual Blocks 1 & 2</span>
                  <span className="font-mono text-slate-600 text-[11px]">Output: (16, 16, 64)</span>
                </div>
                <p className="text-[11px] text-slate-600">
                  64 filters, stride 1 • Identity shortcut F(x) + x without downsampling
                </p>
              </div>

              <div className="flex justify-center"><ArrowDown className="w-3.5 h-3.5 text-slate-400" /></div>

              <div className="p-3 bg-slate-50 border border-slate-200 rounded space-y-1">
                <div className="flex justify-between font-medium text-slate-900">
                  <span>Stage 2: Residual Blocks 3 & 4</span>
                  <span className="font-mono text-slate-600 text-[11px]">Output: (8, 8, 128)</span>
                </div>
                <p className="text-[11px] text-slate-600">
                  128 filters, stride 2 • 1×1 convolution projection shortcut on Block 3
                </p>
              </div>

              <div className="flex justify-center"><ArrowDown className="w-3.5 h-3.5 text-slate-400" /></div>

              <div className="p-3 bg-slate-50 border border-slate-200 rounded space-y-1">
                <div className="flex justify-between font-medium text-slate-900">
                  <span>Stage 3: Residual Blocks 5 & 6</span>
                  <span className="font-mono text-slate-600 text-[11px]">Output: (4, 4, 256)</span>
                </div>
                <p className="text-[11px] text-slate-600">
                  256 filters, stride 2 • 1×1 convolution projection shortcut on Block 5
                </p>
              </div>

              <div className="flex justify-center"><ArrowDown className="w-3.5 h-3.5 text-slate-400" /></div>

              <div className="p-3 bg-slate-50 border border-slate-200 rounded space-y-1">
                <div className="flex justify-between font-medium text-slate-900">
                  <span>Stage 4: Residual Blocks 7 & 8</span>
                  <span className="font-mono text-slate-600 text-[11px]">Output: (2, 2, 512)</span>
                </div>
                <p className="text-[11px] text-slate-600">
                  512 filters, stride 2 • 1×1 convolution projection shortcut on Block 7
                </p>
              </div>

              <div className="flex justify-center"><ArrowDown className="w-3.5 h-3.5 text-slate-400" /></div>

              <div className="p-3 bg-blue-50/60 border border-blue-200 rounded space-y-1">
                <div className="flex justify-between font-semibold text-blue-900">
                  <span>Classifier Head</span>
                  <span className="font-mono text-blue-700 text-[11px]">Output: 10 Classes</span>
                </div>
                <p className="text-[11px] text-slate-600">
                  Global Average Pooling (512-d feature vector) • Dense (10 units, Softmax)
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
