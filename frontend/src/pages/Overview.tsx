import React from 'react';
import { 
  Database, 
  Layers, 
  TrendingUp, 
  Cpu, 
  CheckCircle2, 
  ArrowRight,
  GitFork,
  Target,
  BookOpen
} from 'lucide-react';
import { StatCard } from '../components/StatCard';

interface OverviewProps {
  setActiveTab: (tab: string) => void;
}

export const Overview: React.FC<OverviewProps> = ({ setActiveTab }) => {
  const workflowSteps = [
    { step: 1, title: 'Dataset Acquisition', desc: '27,000 Sentinel-2 images from EuroSAT' },
    { step: 2, title: 'Preprocessing', desc: 'Resized to 64×64 and Z-score normalized' },
    { step: 3, title: 'Augmentation', desc: 'Random flips and ±18° rotations (train-only)' },
    { step: 4, title: 'Model Training', desc: 'Baseline CNN & ResNet-18 from scratch' },
    { step: 5, title: 'Evaluation', desc: 'Tested on held-out 2,700 images' },
    { step: 6, title: 'Demonstration', desc: 'Live inference via FastAPI web interface' }
  ];

  return (
    <div className="space-y-6 pb-12">
      {/* Project Presentation Header */}
      <div className="bg-white border border-slate-200 rounded-lg p-6 shadow-sm">
        <div className="max-w-3xl space-y-3">
          <div className="inline-block px-2.5 py-1 rounded bg-blue-50 text-blue-700 text-xs font-medium border border-blue-100">
            Amity School of Engineering and Technology • Minor Project (ETMN100)
          </div>

          <div>
            <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
              DeepSatVision
            </h1>
            <p className="text-base font-semibold text-slate-700 mt-0.5">
              AI-Based Land Use Classification from Sentinel-2 Images
            </p>
          </div>

          <p className="text-xs text-slate-600 leading-relaxed">
            DeepSatVision is an academic deep learning project developed to classify satellite imagery into 
            10 land-use and land-cover (LULC) categories. Using the EuroSAT dataset (27,000 Sentinel-2 RGB images), 
            we implement and compare a custom Baseline Convolutional Neural Network (CNN) against an 18-layer 
            Deep Residual Network (ResNet-18) implemented from scratch in TensorFlow/Keras.
          </p>

          <div className="flex flex-wrap items-center gap-3 pt-2">
            <button
              onClick={() => setActiveTab('classify')}
              className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-xs font-medium transition-colors shadow-sm"
            >
              <span>Test Live Classification</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => setActiveTab('analytics')}
              className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-medium transition-colors border border-slate-200"
            >
              <span>View Performance Analysis</span>
            </button>
          </div>
        </div>
      </div>

      {/* Four Core Statistic Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          title="Dataset Images"
          value="27,000"
          subtitle="Sentinel-2 RGB patches (64×64)"
          icon={Database}
        />
        <StatCard
          title="Number of Classes"
          value="10"
          subtitle="EuroSAT Land Use / Cover"
          icon={Layers}
        />
        <StatCard
          title="CNN Test Accuracy"
          value="91.11%"
          subtitle="111.9K parameters baseline"
          icon={Cpu}
        />
        <StatCard
          title="ResNet-18 Test Accuracy"
          value="94.04%"
          subtitle="+2.93% gain over baseline"
          icon={TrendingUp}
          badge="Best Model"
        />
      </div>

      {/* Project Introduction & Objectives Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Project Introduction */}
        <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-sm space-y-3">
          <div className="flex items-center gap-2 text-slate-900 font-semibold text-sm">
            <BookOpen className="w-4 h-4 text-blue-600" />
            <span>Project Introduction</span>
          </div>
          <p className="text-xs text-slate-600 leading-relaxed text-justify">
            Monitoring Land Use and Land Cover (LULC) is critical for sustainable urban planning, agricultural yield 
            estimation, environmental conservation, and disaster response. Remote sensing through earth observation satellites 
            like Copernicus Sentinel-2 generates vast amounts of data daily. This project demonstrates how deep learning 
            models can automate the categorization of 64×64 pixel patches with high accuracy and low inference latency.
          </p>
          <div className="pt-2 border-t border-slate-100 text-xs text-slate-500">
            <strong>Key Focus:</strong> Benchmarking a lightweight CNN against ResNet-18 trained from scratch under identical hyperparameter conditions.
          </div>
        </div>

        {/* Project Objectives */}
        <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-sm space-y-3">
          <div className="flex items-center gap-2 text-slate-900 font-semibold text-sm">
            <Target className="w-4 h-4 text-blue-600" />
            <span>Project Objectives</span>
          </div>
          <ul className="space-y-2 text-xs text-slate-600">
            <li className="flex items-start gap-2">
              <CheckCircle2 className="w-3.5 h-3.5 text-blue-600 shrink-0 mt-0.5" />
              <span>Build an efficient data loader with TensorFlow tf.data for EuroSAT.</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2 className="w-3.5 h-3.5 text-blue-600 shrink-0 mt-0.5" />
              <span>Design a 3-block custom baseline CNN tailored for low-resolution satellite patches.</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2 className="w-3.5 h-3.5 text-blue-600 shrink-0 mt-0.5" />
              <span>Implement ResNet-18 from scratch with residual skip connections.</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2 className="w-3.5 h-3.5 text-blue-600 shrink-0 mt-0.5" />
              <span>Compare model convergence, accuracy, confusion matrices, and inference latency.</span>
            </li>
          </ul>
        </div>
      </div>

      {/* Model Comparison Table */}
      <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-sm space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-slate-900 font-semibold text-sm">
            <Cpu className="w-4 h-4 text-blue-600" />
            <span>Model Comparison Overview</span>
          </div>
          <span className="text-xs text-slate-500">Evaluated on 2,700 Test Images</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="bg-slate-50 text-slate-600 border-b border-slate-200">
              <tr>
                <th className="py-2.5 px-3 font-semibold">Architecture</th>
                <th className="py-2.5 px-3 font-semibold">Parameters</th>
                <th className="py-2.5 px-3 font-semibold">Test Accuracy</th>
                <th className="py-2.5 px-3 font-semibold">Test Loss</th>
                <th className="py-2.5 px-3 font-semibold">Macro Precision</th>
                <th className="py-2.5 px-3 font-semibold">Macro Recall</th>
                <th className="py-2.5 px-3 font-semibold">Macro F1</th>
                <th className="py-2.5 px-3 font-semibold">Errors (out of 2,700)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono">
              <tr>
                <td className="py-2.5 px-3 font-sans font-medium text-slate-900">Custom Baseline CNN</td>
                <td className="py-2.5 px-3 text-slate-600">111,946</td>
                <td className="py-2.5 px-3 text-slate-800 font-semibold">91.11%</td>
                <td className="py-2.5 px-3 text-slate-600">0.2627</td>
                <td className="py-2.5 px-3 text-slate-600">91.53%</td>
                <td className="py-2.5 px-3 text-slate-600">90.86%</td>
                <td className="py-2.5 px-3 text-slate-600">90.92%</td>
                <td className="py-2.5 px-3 text-rose-600">240</td>
              </tr>
              <tr className="bg-blue-50/40">
                <td className="py-2.5 px-3 font-sans font-semibold text-blue-900 flex items-center gap-1.5">
                  <span>ResNet-18 (From Scratch)</span>
                  <span className="text-[10px] px-1.5 py-0.2 rounded bg-blue-100 text-blue-700 font-medium">Better</span>
                </td>
                <td className="py-2.5 px-3 text-slate-600">11,191,242</td>
                <td className="py-2.5 px-3 text-blue-700 font-bold">94.04%</td>
                <td className="py-2.5 px-3 text-slate-600">0.1737</td>
                <td className="py-2.5 px-3 text-slate-600">93.89%</td>
                <td className="py-2.5 px-3 text-slate-600">94.00%</td>
                <td className="py-2.5 px-3 text-slate-600">93.92%</td>
                <td className="py-2.5 px-3 text-emerald-700 font-semibold">161 (-79 errors)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Project Workflow Steps */}
      <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-sm space-y-3">
        <div className="flex items-center gap-2 text-slate-900 font-semibold text-sm">
          <GitFork className="w-4 h-4 text-blue-600" />
          <span>Project Workflow</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3 pt-1">
          {workflowSteps.map((s) => (
            <div key={s.step} className="bg-slate-50 border border-slate-200 rounded-lg p-3 space-y-1">
              <span className="text-[10px] font-bold text-blue-600 block">STEP 0{s.step}</span>
              <h5 className="text-xs font-semibold text-slate-900 leading-snug">{s.title}</h5>
              <p className="text-[11px] text-slate-500 leading-tight">{s.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
