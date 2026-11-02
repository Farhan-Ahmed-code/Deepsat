import React from 'react';
import {
  BookOpen,
  Target,
  AlertTriangle,
  Rocket,
  Code
} from 'lucide-react';

export const ProjectInformation: React.FC = () => {
  return (
    <div className="space-y-6 pb-12">
      {/* Academic Header */}
      <div className="bg-white border border-slate-200 rounded-lg p-6">
        <div className="flex flex-wrap gap-2 mb-3">
          <span className="text-xs font-medium px-2.5 py-1 bg-blue-50 text-blue-700 border border-blue-200 rounded">
            Amity University Uttar Pradesh, Noida
          </span>
          <span className="text-xs font-medium px-2.5 py-1 bg-slate-100 text-slate-600 border border-slate-200 rounded">
            Minor Project — ETMN100
          </span>
          <span className="text-xs font-medium px-2.5 py-1 bg-slate-100 text-slate-600 border border-slate-200 rounded">
            B.Tech CSE (Data Science) — Semester VII
          </span>
        </div>

        <h1 className="text-2xl font-bold text-slate-900 leading-snug">
          DeepSatVision: AI-Based Land Use Classification from Sentinel-2 Images
        </h1>
        <p className="text-sm text-slate-500 mt-2 leading-relaxed max-w-3xl">
          Submitted in partial fulfillment of the requirements for the award of the degree of
          Bachelor of Technology in Computer Science and Engineering (Data Science) at
          Amity School of Engineering and Technology, Amity University Uttar Pradesh.
        </p>
      </div>

      {/* People Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Student 1 */}
        <div className="bg-white border border-slate-200 rounded-lg p-5">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-full bg-blue-50 border border-blue-200 flex items-center justify-center text-blue-700 font-bold text-sm">
              FA
            </div>
            <div>
              <h4 className="text-sm font-semibold text-slate-900">Farhan Ahmed Siddique</h4>
              <p className="text-xs text-slate-500">Student Researcher</p>
            </div>
          </div>
          <div className="pt-3 border-t border-slate-100 space-y-1 text-xs text-slate-600">
            <div>Programme: B.Tech CSE (Data Science)</div>
            <div>Semester: VII</div>
            <div>Focus: Model Architectures & Training</div>
          </div>
        </div>

        {/* Student 2 */}
        <div className="bg-white border border-slate-200 rounded-lg p-5">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-full bg-blue-50 border border-blue-200 flex items-center justify-center text-blue-700 font-bold text-sm">
              AP
            </div>
            <div>
              <h4 className="text-sm font-semibold text-slate-900">Adesh Pal</h4>
              <p className="text-xs text-slate-500">Student Researcher</p>
            </div>
          </div>
          <div className="pt-3 border-t border-slate-100 space-y-1 text-xs text-slate-600">
            <div>Programme: B.Tech CSE (Data Science)</div>
            <div>Semester: VII</div>
            <div>Focus: Dataset Profiling & Evaluation</div>
          </div>
        </div>

        {/* Faculty Guide */}
        <div className="bg-white border border-slate-200 rounded-lg p-5">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-full bg-green-50 border border-green-200 flex items-center justify-center text-green-700 font-bold text-sm">
              AG
            </div>
            <div>
              <h4 className="text-sm font-semibold text-slate-900">Dr Aakanshi Gupta</h4>
              <p className="text-xs text-green-700 font-medium">Faculty Guide</p>
            </div>
          </div>
          <div className="pt-3 border-t border-slate-100 space-y-1 text-xs text-slate-600">
            <div>Designation: Assistant Professor-III</div>
            <div>Dept: Computer Science & Engineering</div>
            <div>Amity School of Engineering & Technology</div>
          </div>
        </div>
      </div>

      {/* Abstract & Problem Statement */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-3">
          <div className="flex items-center gap-2">
            <BookOpen className="w-4 h-4 text-slate-400" />
            <h3 className="text-sm font-semibold text-slate-900">Project Abstract</h3>
          </div>
          <p className="text-sm text-slate-600 leading-relaxed text-justify">
            DeepSatVision presents an empirical deep learning computer vision framework for automated
            Land Use and Land Cover (LULC) classification from multi-spectral Sentinel-2 satellite imagery.
            Utilizing the EuroSAT RGB benchmark dataset (27,000 images, 10 categories), the project
            implements and rigorously benchmarks two architectures: a custom Baseline CNN (111.9K parameters)
            and a complete ResNet-18 residual network (11.19M parameters) engineered from scratch in
            TensorFlow/Keras without pretrained ImageNet weights. ResNet-18 achieved 94.04% test accuracy
            versus 91.11% for the Baseline CNN, reducing test classification errors by 32.9% on an independent
            2,700-image test split.
          </p>
        </div>

        <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-3">
          <div className="flex items-center gap-2">
            <Target className="w-4 h-4 text-slate-400" />
            <h3 className="text-sm font-semibold text-slate-900">Problem Statement & Motivation</h3>
          </div>
          <p className="text-sm text-slate-600 leading-relaxed text-justify">
            Timely, accurate, and scalable monitoring of terrestrial land use is paramount for agricultural
            management, urban sprawl assessment, disaster mitigation, and ecological conservation.
            Traditional manual aerial surveys and hand-engineered remote sensing indices are labor-intensive,
            error-prone, and cannot scale across continental satellite swaths. Automated deep learning pipelines
            resolve these limitations, but require rigorous benchmarking to evaluate whether deep residual skip
            architectures provide tangible accuracy gains on low-resolution 64x64 satellite scenes when
            trained from scratch.
          </p>
        </div>
      </div>

      {/* Technology Stack */}
      <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-4">
        <div className="flex items-center gap-2">
          <Code className="w-4 h-4 text-slate-400" />
          <h3 className="text-sm font-semibold text-slate-900">Technology Stack</h3>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
            <p className="text-xs text-slate-500 mb-1">Core ML Engine</p>
            <p className="text-sm font-semibold text-slate-900">TensorFlow 2.18</p>
            <p className="text-xs text-slate-500">Keras 3 + tf.data API</p>
          </div>
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
            <p className="text-xs text-slate-500 mb-1">Backend Server</p>
            <p className="text-sm font-semibold text-slate-900">FastAPI + Uvicorn</p>
            <p className="text-xs text-slate-500">Asynchronous REST API</p>
          </div>
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
            <p className="text-xs text-slate-500 mb-1">Frontend Platform</p>
            <p className="text-sm font-semibold text-slate-900">React 18 + Vite</p>
            <p className="text-xs text-slate-500">TypeScript + Tailwind CSS</p>
          </div>
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
            <p className="text-xs text-slate-500 mb-1">Data & Visualization</p>
            <p className="text-sm font-semibold text-slate-900">NumPy + Recharts</p>
            <p className="text-xs text-slate-500">Pandas, Matplotlib, Scikit-Learn</p>
          </div>
        </div>
      </div>

      {/* Current Limitations */}
      <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-3">
        <div className="flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-amber-500" />
          <h3 className="text-sm font-semibold text-slate-900">Current Project Limitations</h3>
        </div>
        <ul className="space-y-2 text-sm text-slate-600">
          <li className="flex items-start gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400 shrink-0 mt-2" />
            <span>
              <strong className="text-slate-800">RGB-Only Spectral Input:</strong> Only the 3 visual spectral bands were
              evaluated; the remaining 10 Sentinel-2 bands (NIR, SWIR, Red Edge) were not ingested.
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400 shrink-0 mt-2" />
            <span>
              <strong className="text-slate-800">Single-Tile Classification:</strong> Classifies discrete 64x64 patches
              rather than performing dense pixel-wise semantic segmentation across continental landscapes.
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400 shrink-0 mt-2" />
            <span>
              <strong className="text-slate-800">Temporal Invariance:</strong> Models analyze individual static acquisitions
              without integrating multi-temporal seasonal time-series observations.
            </span>
          </li>
        </ul>
      </div>

      {/* Future Scope */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-5 space-y-4">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div className="flex items-center gap-2">
            <Rocket className="w-4 h-4 text-blue-600" />
            <h3 className="text-sm font-semibold text-slate-900">Proposed Future Scope (Major Project Roadmap)</h3>
          </div>
          <span className="text-xs font-medium px-2.5 py-1 bg-white text-blue-700 border border-blue-300 rounded">
            Future Work — Not Claimed in Minor Project
          </span>
        </div>

        <p className="text-sm text-slate-600 leading-relaxed">
          The following features represent conceptual avenues for expansion in the upcoming B.Tech Major Project
          and are <strong>explicitly demarcated as future research directions</strong>, not completed minor project achievements:
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="bg-white border border-blue-200 rounded-lg p-4">
            <h5 className="text-sm font-semibold text-slate-900 mb-1">
              1. Transfer Learning & Vision Transformers (ViT)
            </h5>
            <p className="text-xs text-slate-600 leading-relaxed">
              Fine-tuning modern pretrained foundation models (ConvNeXt, EfficientNet-V2, ViT-B/16) to explore
              whether transfer dynamics from natural images transfer effectively to remote sensing domains.
            </p>
          </div>
          <div className="bg-white border border-blue-200 rounded-lg p-4">
            <h5 className="text-sm font-semibold text-slate-900 mb-1">
              2. Multi-Temporal Change Detection
            </h5>
            <p className="text-xs text-slate-600 leading-relaxed">
              Developing Siamese recurrent neural networks to process paired Sentinel-2 acquisitions over time,
              generating difference heatmaps to detect deforestation, flooding, or urban development.
            </p>
          </div>
          <div className="bg-white border border-blue-200 rounded-lg p-4">
            <h5 className="text-sm font-semibold text-slate-900 mb-1">
              3. Explainable AI (XAI) & Grad-CAM Visualizations
            </h5>
            <p className="text-xs text-slate-600 leading-relaxed">
              Integrating Gradient-weighted Class Activation Mapping (Grad-CAM) to generate spatial saliency heatmaps,
              revealing the exact visual features dictating neural classification decisions.
            </p>
          </div>
          <div className="bg-white border border-blue-200 rounded-lg p-4">
            <h5 className="text-sm font-semibold text-slate-900 mb-1">
              4. Full 13-Band Multispectral Fusion & GIS Mapping
            </h5>
            <p className="text-xs text-slate-600 leading-relaxed">
              Extending the pipeline to all 13 spectral bands (EuroSATallBands) and integrating with Leaflet/Mapbox GIS
              for interactive bounding-box coordinate queries on live satellite maps.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
