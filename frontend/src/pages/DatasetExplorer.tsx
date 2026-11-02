import React, { useState, useEffect } from 'react';
import { 
  Database, 
  Image as ImageIcon, 
  Info,
  CheckCircle2
} from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer 
} from 'recharts';
import { fetchDatasetInfo, fetchSampleImages } from '../services/api';
import { DatasetInfo, SampleImageItem } from '../types';

interface DatasetExplorerProps {
  onSelectSampleForPrediction?: (sample: SampleImageItem) => void;
}

export const DatasetExplorer: React.FC<DatasetExplorerProps> = ({ onSelectSampleForPrediction }) => {
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [samples, setSamples] = useState<SampleImageItem[]>([]);
  const [selectedClass, setSelectedClass] = useState<string>('Forest');
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    async function loadData() {
      try {
        const [info, sampleImgs] = await Promise.all([
          fetchDatasetInfo(),
          fetchSampleImages()
        ]);
        setDatasetInfo(info);
        setSamples(sampleImgs);
      } catch (err) {
        console.error('Error loading dataset info:', err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  const classData = datasetInfo?.classes || [
    { className: 'AnnualCrop', index: 0, train: 2100, val: 600, test: 300, total: 3000, desc: 'Cultivated agricultural land with seasonal crops (wheat, barley, corn).' },
    { className: 'Forest', index: 1, train: 2100, val: 600, test: 300, total: 3000, desc: 'Dense natural woodland and commercial timber forests.' },
    { className: 'HerbaceousVegetation', index: 2, train: 2100, val: 600, test: 300, total: 3000, desc: 'Natural grasslands, moors, meadows, and wild low-height shrublands.' },
    { className: 'Highway', index: 3, train: 1750, val: 500, test: 250, total: 2500, desc: 'High-speed multi-lane roads, motorways, highway junctions, and transport arteries.' },
    { className: 'Industrial', index: 4, train: 1750, val: 500, test: 250, total: 2500, desc: 'Manufacturing plants, storage warehouses, chemical facilities, and industrial complexes.' },
    { className: 'Pasture', index: 5, train: 1400, val: 400, test: 200, total: 2000, desc: 'Grazing land for livestock, fenced green pastures, and open grasslands.' },
    { className: 'PermanentCrop', index: 6, train: 1750, val: 500, test: 250, total: 2500, desc: 'Perennial plantations: orchards, vineyards, fruit groves, and olive trees.' },
    { className: 'Residential', index: 7, train: 2100, val: 600, test: 300, total: 3000, desc: 'Suburban and urban living areas, houses, rooftops, gardens, and residential blocks.' },
    { className: 'River', index: 8, train: 1750, val: 500, test: 250, total: 2500, desc: 'Natural running freshwater waterways, flowing rivers, canals, and riverbanks.' },
    { className: 'SeaLake', index: 9, train: 2100, val: 600, test: 300, total: 3000, desc: 'Open standing water bodies, sea coastlines, ocean bays, lakes, and reservoirs.' }
  ];

  const currentClassInfo = classData.find(c => c.className === selectedClass) || classData[0];
  const filteredSamples = samples.filter(s => s.className === selectedClass);

  return (
    <div className="space-y-6 pb-12">
      {/* Header Info */}
      <div className="bg-white border border-slate-200 rounded-lg p-6 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <span className="text-xs font-semibold text-blue-600 uppercase tracking-wider">
            Dataset Overview
          </span>
          <h2 className="text-xl font-bold text-slate-900 mt-1">
            EuroSAT Land Use & Land Cover Dataset
          </h2>
          <p className="text-xs text-slate-600 max-w-2xl mt-1">
            Contains 27,000 geo-referenced Sentinel-2 satellite images covering 10 distinct LULC classes. 
            All images are 64 × 64 pixels with 3 RGB color channels.
          </p>
        </div>

        <div className="flex items-center gap-3 shrink-0">
          <div className="px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-center">
            <span className="block text-[10px] text-slate-500 uppercase font-medium">Dimensions</span>
            <span className="text-sm font-bold text-slate-900 font-mono">64 × 64 × 3</span>
          </div>
          <div className="px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-center">
            <span className="block text-[10px] text-slate-500 uppercase font-medium">Total Samples</span>
            <span className="text-sm font-bold text-blue-700 font-mono">27,000</span>
          </div>
        </div>
      </div>

      {/* Dataset Splits & Chart Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Class Distribution Chart */}
        <div className="lg:col-span-2 bg-white border border-slate-200 rounded-lg p-5 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-900">
              Class Distribution by Split
            </h3>
            <span className="text-xs text-slate-500 font-mono">70% Train • 20% Val • 10% Test</span>
          </div>

          <div className="h-72 w-full pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={classData} margin={{ top: 10, right: 10, left: -20, bottom: 45 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                <XAxis 
                  dataKey="className" 
                  stroke="#64748b" 
                  fontSize={10} 
                  angle={-45} 
                  textAnchor="end" 
                  interval={0}
                />
                <YAxis stroke="#64748b" fontSize={10} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#ffffff',
                    borderColor: '#cbd5e1',
                    borderRadius: '0.5rem',
                    fontSize: '11px',
                    color: '#0f172a',
                    boxShadow: '0 2px 4px rgba(0,0,0,0.05)'
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                <Bar dataKey="train" name="Train (18,900)" fill="#2563eb" stackId="a" />
                <Bar dataKey="val" name="Validation (5,400)" fill="#0ea5e9" stackId="a" />
                <Bar dataKey="test" name="Test (2,700)" fill="#10b981" stackId="a" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Dataset Splits and Preprocessing Info */}
        <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-sm space-y-4 flex flex-col justify-between">
          <div className="space-y-3">
            <div className="flex items-center gap-1.5 text-slate-900 font-semibold text-sm">
              <Info className="w-4 h-4 text-blue-600" />
              <span>Dataset Details</span>
            </div>

            <div className="space-y-2 text-xs">
              <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                <span className="font-semibold text-slate-800 block mb-1">Stratified Splits</span>
                <div className="space-y-1 text-slate-600 font-mono text-[11px]">
                  <div className="flex justify-between">
                    <span>Training Set:</span>
                    <span className="font-semibold text-slate-900">18,900 (70%)</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Validation Set:</span>
                    <span className="font-semibold text-slate-900">5,400 (20%)</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Testing Set:</span>
                    <span className="font-semibold text-slate-900">2,700 (10%)</span>
                  </div>
                </div>
              </div>

              <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                <span className="font-semibold text-slate-800 block mb-1">Channel Normalization</span>
                <p className="text-[11px] text-slate-600">
                  Per-channel standardization applied: Z = (X - μ) / σ
                </p>
                <div className="mt-1 font-mono text-[10px] text-slate-500">
                  Mean: [0.3484, 0.3824, 0.4085]<br />
                  Std Dev: [0.2102, 0.1413, 0.1186]
                </div>
              </div>
            </div>
          </div>

          <div className="text-[11px] text-slate-500 border-t border-slate-100 pt-2">
            <strong>tf.data Pipeline:</strong> Data augmentation is applied only during training.
          </div>
        </div>
      </div>

      {/* Class Selector & Image Gallery */}
      <div className="bg-white border border-slate-200 rounded-lg p-6 shadow-sm space-y-4">
        <div>
          <h3 className="text-sm font-bold text-slate-900 mb-1">
            EuroSAT Class Gallery
          </h3>
          <p className="text-xs text-slate-500 mb-3">
            Select any class below to inspect genuine satellite image samples:
          </p>

          {/* 10 Class Buttons */}
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
            {classData.map((cls) => {
              const isSelected = cls.className === selectedClass;
              return (
                <button
                  key={cls.className}
                  onClick={() => setSelectedClass(cls.className)}
                  className={`p-2.5 rounded-lg text-left border transition-all text-xs ${
                    isSelected
                      ? 'bg-blue-50 border-blue-600 text-blue-900 font-semibold'
                      : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50 hover:text-slate-900'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="truncate">{cls.className}</span>
                    <span className="text-[10px] font-mono text-slate-400 font-normal">{cls.total}</span>
                  </div>
                  <span className="text-[10px] text-slate-400 font-mono font-normal">Class #{cls.index}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Selected Class Description & Image Grid */}
        <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-200 pb-3">
            <div>
              <h4 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                <span>{currentClassInfo.className}</span>
                <span className="text-xs px-2 py-0.5 rounded bg-blue-100 text-blue-800 font-mono font-normal">
                  {currentClassInfo.total} Images
                </span>
              </h4>
              <p className="text-xs text-slate-600 mt-0.5">
                {currentClassInfo.desc}
              </p>
            </div>

            <div className="text-xs font-mono text-slate-500 flex gap-3">
              <span>Train: <strong className="text-slate-800">{currentClassInfo.train}</strong></span>
              <span>Val: <strong className="text-slate-800">{currentClassInfo.val}</strong></span>
              <span>Test: <strong className="text-slate-800">{currentClassInfo.test}</strong></span>
            </div>
          </div>

          {/* Actual Sample Images Grid */}
          <div>
            <div className="text-xs font-semibold text-slate-700 mb-2">
              Sample Satellite Images (Native 64 × 64 pixels):
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 gap-3">
              {filteredSamples.length > 0 ? (
                filteredSamples.map((sample, idx) => (
                  <div
                    key={idx}
                    className="bg-white border border-slate-200 rounded-lg p-2 flex flex-col items-center shadow-xs"
                  >
                    <div className="w-20 h-20 rounded bg-slate-100 overflow-hidden border border-slate-200">
                      <img
                        src={sample.url}
                        alt={sample.filename}
                        className="w-full h-full object-cover"
                        style={{ imageRendering: 'pixelated' }}
                      />
                    </div>
                    <span className="mt-1.5 text-[10px] font-mono text-slate-600 truncate max-w-full">
                      {sample.filename}
                    </span>
                    {onSelectSampleForPrediction && (
                      <button
                        onClick={() => onSelectSampleForPrediction(sample)}
                        className="mt-1 w-full py-0.5 rounded bg-slate-100 hover:bg-blue-600 hover:text-white text-[10px] font-medium text-slate-700 transition-colors"
                      >
                        Select Tile
                      </button>
                    )}
                  </div>
                ))
              ) : (
                <p className="text-xs text-slate-400">Loading samples...</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
