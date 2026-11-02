import React, { useState, useEffect } from 'react';
import { 
  TrendingUp, 
  BarChart3, 
  AlertCircle,
  CheckCircle2
} from 'lucide-react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  BarChart,
  Bar
} from 'recharts';
import { fetchModelMetrics } from '../services/api';
import { ModelMetricsData } from '../types';

export const PerformanceAnalytics: React.FC = () => {
  const [metrics, setMetrics] = useState<ModelMetricsData | null>(null);
  const [activeChart, setActiveChart] = useState<'accuracy' | 'loss'>('accuracy');
  const [activeCM, setActiveCM] = useState<'resnet18' | 'cnn'>('resnet18');
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    async function loadMetrics() {
      try {
        const data = await fetchModelMetrics();
        setMetrics(data);
      } catch (err) {
        console.error('Failed to load metrics:', err);
      } finally {
        setLoading(false);
      }
    }
    loadMetrics();
  }, []);

  const epochsData = metrics?.epochs_data || [];
  const perClass = metrics?.per_class_comparison || [];
  const cmData = metrics?.confusion_matrices || {};
  const classes = metrics?.classes || [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
  ];

  const summaryRows = [
    { label: 'Test Accuracy', cnn: '91.11%', resnet: '94.04%', diff: '+2.93%' },
    { label: 'Test Loss', cnn: '0.2627', resnet: '0.1737', diff: '-0.089' },
    { label: 'Macro Precision', cnn: '91.53%', resnet: '93.89%', diff: '+2.36%' },
    { label: 'Macro Recall', cnn: '90.86%', resnet: '94.00%', diff: '+3.14%' },
    { label: 'Macro F1-Score', cnn: '90.92%', resnet: '93.92%', diff: '+3.00%' },
    { label: 'Misclassified (of 2,700)', cnn: '240', resnet: '161', diff: '-79 errors' },
  ];

  return (
    <div className="space-y-6 pb-12">
      {/* Header */}
      <div className="bg-white border border-slate-200 rounded-lg p-6 shadow-sm">
        <span className="text-xs font-semibold text-blue-600 uppercase tracking-wider">
          Empirical Evaluation
        </span>
        <h2 className="text-xl font-bold text-slate-900 mt-1">
          Performance Analysis & Benchmarking
        </h2>
        <p className="text-xs text-slate-600 max-w-2xl mt-1">
          Both models were evaluated on the same independently held-out test set of 2,700 images not seen 
          during training or validation. Results confirm ResNet-18 outperforms the baseline CNN across all metrics.
        </p>
      </div>

      {/* Summary Metrics Table */}
      <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-sm space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-bold text-slate-900">
            Summary of Results
          </h3>
          <span className="text-xs text-slate-500">Test Set: 2,700 Images</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="bg-slate-50 border-b border-slate-200 text-slate-600">
              <tr>
                <th className="py-2.5 px-4 font-semibold">Metric</th>
                <th className="py-2.5 px-4 font-semibold text-center">Baseline CNN</th>
                <th className="py-2.5 px-4 font-semibold text-center text-blue-700">ResNet-18</th>
                <th className="py-2.5 px-4 font-semibold text-center">Improvement</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {summaryRows.map((row) => (
                <tr key={row.label} className="hover:bg-slate-50">
                  <td className="py-2.5 px-4 font-medium text-slate-800">{row.label}</td>
                  <td className="py-2.5 px-4 text-center font-mono text-slate-700">{row.cnn}</td>
                  <td className="py-2.5 px-4 text-center font-mono text-blue-700 font-semibold">{row.resnet}</td>
                  <td className="py-2.5 px-4 text-center">
                    <span className="inline-block px-2 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-100 text-[11px] font-mono font-medium">
                      {row.diff}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Training History Charts */}
      <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-bold text-slate-900">
              Training & Validation Curves (20 Epochs)
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Plotted from actual training history files (history_baseline_cnn.json & history_resnet18.json)
            </p>
          </div>

          <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-md border border-slate-200 self-start sm:self-auto">
            <button
              onClick={() => setActiveChart('accuracy')}
              className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                activeChart === 'accuracy' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Accuracy
            </button>
            <button
              onClick={() => setActiveChart('loss')}
              className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                activeChart === 'loss' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Loss
            </button>
          </div>
        </div>

        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            {activeChart === 'accuracy' ? (
              <LineChart data={epochsData} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="epoch" stroke="#64748b" fontSize={11} label={{ value: 'Epoch', position: 'insideBottom', offset: -3, fontSize: 11 }} />
                <YAxis stroke="#64748b" fontSize={11} domain={[60, 100]} label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft', offset: 15, fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#fff',
                    borderColor: '#e2e8f0',
                    borderRadius: '0.5rem',
                    fontSize: '11px',
                    color: '#0f172a',
                    boxShadow: '0 2px 8px rgba(0,0,0,0.08)'
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                <Line type="monotone" dataKey="cnn_accuracy" name="CNN Train" stroke="#93c5fd" strokeWidth={1.5} dot={false} strokeDasharray="4 4" />
                <Line type="monotone" dataKey="cnn_val_accuracy" name="CNN Validation" stroke="#2563eb" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="resnet_accuracy" name="ResNet-18 Train" stroke="#86efac" strokeWidth={1.5} dot={false} strokeDasharray="4 4" />
                <Line type="monotone" dataKey="resnet_val_accuracy" name="ResNet-18 Validation" stroke="#16a34a" strokeWidth={2} dot={false} />
              </LineChart>
            ) : (
              <LineChart data={epochsData} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="epoch" stroke="#64748b" fontSize={11} label={{ value: 'Epoch', position: 'insideBottom', offset: -3, fontSize: 11 }} />
                <YAxis stroke="#64748b" fontSize={11} domain={[0, 1.2]} label={{ value: 'Loss', angle: -90, position: 'insideLeft', offset: 15, fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#fff',
                    borderColor: '#e2e8f0',
                    borderRadius: '0.5rem',
                    fontSize: '11px',
                    boxShadow: '0 2px 8px rgba(0,0,0,0.08)'
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                <Line type="monotone" dataKey="cnn_loss" name="CNN Train" stroke="#fca5a5" strokeWidth={1.5} dot={false} strokeDasharray="4 4" />
                <Line type="monotone" dataKey="cnn_val_loss" name="CNN Validation" stroke="#dc2626" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="resnet_loss" name="ResNet-18 Train" stroke="#93c5fd" strokeWidth={1.5} dot={false} strokeDasharray="4 4" />
                <Line type="monotone" dataKey="resnet_val_loss" name="ResNet-18 Validation" stroke="#2563eb" strokeWidth={2} dot={false} />
              </LineChart>
            )}
          </ResponsiveContainer>
        </div>
      </div>

      {/* Per-Class Results Table */}
      <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-sm space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-bold text-slate-900">Per-Class Classification Report</h3>
            <p className="text-xs text-slate-500 mt-0.5">Comparison across all 10 EuroSAT classes</p>
          </div>
          <span className="text-xs font-medium px-2 py-1 bg-blue-50 text-blue-700 rounded border border-blue-100">
            ResNet-18 leads in 9 of 10 classes (F1)
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="bg-slate-50 border-b border-slate-200 text-slate-600">
              <tr>
                <th className="py-2.5 px-3 font-semibold">Class</th>
                <th className="py-2.5 px-3 font-semibold text-center">Support</th>
                <th className="py-2.5 px-3 font-semibold text-center" colSpan={3}>Baseline CNN</th>
                <th className="py-2.5 px-3 font-semibold text-center text-blue-700" colSpan={3}>ResNet-18</th>
                <th className="py-2.5 px-3 font-semibold text-right">F1 Delta</th>
              </tr>
              <tr className="text-[10px] text-slate-500 bg-slate-50 border-b border-slate-100">
                <th className="py-1.5 px-3"></th>
                <th className="py-1.5 px-3 text-center"></th>
                <th className="py-1.5 px-3 text-center">Prec</th>
                <th className="py-1.5 px-3 text-center">Rec</th>
                <th className="py-1.5 px-3 text-center">F1</th>
                <th className="py-1.5 px-3 text-center">Prec</th>
                <th className="py-1.5 px-3 text-center">Rec</th>
                <th className="py-1.5 px-3 text-center font-semibold text-blue-600">F1</th>
                <th className="py-1.5 px-3 text-right"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono">
              {perClass.map((row) => (
                <tr key={row.className} className="hover:bg-slate-50">
                  <td className="py-2 px-3 font-sans font-medium text-slate-900">{row.className}</td>
                  <td className="py-2 px-3 text-center text-slate-500">{row.support}</td>
                  <td className="py-2 px-3 text-center text-slate-700">{row.cnn_precision.toFixed(1)}%</td>
                  <td className="py-2 px-3 text-center text-slate-700">{row.cnn_recall.toFixed(1)}%</td>
                  <td className="py-2 px-3 text-center text-slate-700">{row.cnn_f1.toFixed(1)}%</td>
                  <td className="py-2 px-3 text-center text-slate-700">{row.resnet_precision.toFixed(1)}%</td>
                  <td className="py-2 px-3 text-center text-slate-700">{row.resnet_recall.toFixed(1)}%</td>
                  <td className="py-2 px-3 text-center font-semibold text-blue-700">{row.resnet_f1.toFixed(1)}%</td>
                  <td className="py-2 px-3 text-right">
                    <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                      row.f1_improvement >= 0
                        ? 'bg-emerald-50 text-emerald-700 border border-emerald-100'
                        : 'bg-slate-100 text-slate-600'
                    }`}>
                      {row.f1_improvement >= 0 ? '+' : ''}{row.f1_improvement.toFixed(2)}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
            <tfoot className="bg-slate-50 border-t-2 border-slate-300 font-mono font-bold text-slate-900">
              <tr>
                <td className="py-2.5 px-3 font-sans">Macro Average</td>
                <td className="py-2.5 px-3 text-center text-slate-500">2,700</td>
                <td className="py-2.5 px-3 text-center">91.53%</td>
                <td className="py-2.5 px-3 text-center">90.86%</td>
                <td className="py-2.5 px-3 text-center">90.92%</td>
                <td className="py-2.5 px-3 text-center">93.89%</td>
                <td className="py-2.5 px-3 text-center">94.00%</td>
                <td className="py-2.5 px-3 text-center text-blue-700">93.92%</td>
                <td className="py-2.5 px-3 text-right">
                  <span className="px-1.5 py-0.5 rounded text-[10px] bg-emerald-50 text-emerald-700 border border-emerald-100">+3.00%</span>
                </td>
              </tr>
            </tfoot>
          </table>
        </div>
      </div>

      {/* Confusion Matrix */}
      <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-bold text-slate-900">
              Confusion Matrix
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Diagonal entries are correct predictions. Off-diagonal entries are misclassifications.
            </p>
          </div>
          <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-md border border-slate-200 self-start sm:self-auto">
            <button
              onClick={() => setActiveCM('resnet18')}
              className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                activeCM === 'resnet18' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              ResNet-18 (161 errors)
            </button>
            <button
              onClick={() => setActiveCM('cnn')}
              className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                activeCM === 'cnn' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Baseline CNN (240 errors)
            </button>
          </div>
        </div>

        {cmData && (
          <div className="overflow-x-auto">
            {activeCM === 'resnet18' && cmData.resnet18 && (
              <table className="w-full text-center border-collapse text-[11px] font-mono">
                <thead>
                  <tr>
                    <th className="p-1.5 text-slate-500 text-left text-[10px] font-normal">True \ Predicted</th>
                    {classes.map(c => (
                      <th key={c} className="p-1.5 text-slate-600 text-[10px] font-semibold" title={c}>
                        {c.substring(0, 4)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {cmData.resnet18.map((row, rIdx) => (
                    <tr key={rIdx}>
                      <td className="p-1.5 text-left font-sans text-slate-700 font-medium text-[10px] whitespace-nowrap pr-4">
                        {classes[rIdx]}
                      </td>
                      {row.map((val, cIdx) => {
                        const isDiag = rIdx === cIdx;
                        return (
                          <td key={cIdx} className={`p-1.5 border border-slate-100 ${
                            isDiag
                              ? 'bg-blue-100 text-blue-900 font-bold'
                              : val > 0
                              ? 'bg-rose-50 text-rose-700'
                              : 'text-slate-300'
                          }`}>
                            {val}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            )}

            {activeCM === 'cnn' && cmData.cnn && (
              <table className="w-full text-center border-collapse text-[11px] font-mono">
                <thead>
                  <tr>
                    <th className="p-1.5 text-slate-500 text-left text-[10px] font-normal">True \ Predicted</th>
                    {classes.map(c => (
                      <th key={c} className="p-1.5 text-slate-600 text-[10px] font-semibold" title={c}>
                        {c.substring(0, 4)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {cmData.cnn.map((row, rIdx) => (
                    <tr key={rIdx}>
                      <td className="p-1.5 text-left font-sans text-slate-700 font-medium text-[10px] whitespace-nowrap pr-4">
                        {classes[rIdx]}
                      </td>
                      {row.map((val, cIdx) => {
                        const isDiag = rIdx === cIdx;
                        return (
                          <td key={cIdx} className={`p-1.5 border border-slate-100 ${
                            isDiag
                              ? 'bg-blue-100 text-blue-900 font-bold'
                              : val > 0
                              ? 'bg-rose-50 text-rose-700'
                              : 'text-slate-300'
                          }`}>
                            {val}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        )}

        <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg flex items-start gap-2 text-xs text-amber-900">
          <AlertCircle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
          <div>
            <strong>Key Observation:</strong> The baseline CNN frequently confuses Highway with River (both exhibit linear spectral patterns) 
            and PermanentCrop with Pasture (both show high green canopy reflectance). ResNet-18's deeper receptive field resolves these 
            spectral ambiguities, reducing highway misclassifications significantly.
          </div>
        </div>
      </div>
    </div>
  );
};
