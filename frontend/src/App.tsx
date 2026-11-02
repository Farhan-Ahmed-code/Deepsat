import React, { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { Navbar } from './components/Navbar';
import { Overview } from './pages/Overview';
import { DatasetExplorer } from './pages/DatasetExplorer';
import { ModelArchitectures } from './pages/ModelArchitectures';
import { PerformanceAnalytics } from './pages/PerformanceAnalytics';
import { LiveClassification } from './pages/LiveClassification';
import { Methodology } from './pages/Methodology';
import { ProjectInformation } from './pages/ProjectInformation';
import { fetchHealth } from './services/api';
import { SampleImageItem } from './types';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('overview');
  const [backendOnline, setBackendOnline] = useState<boolean>(true);
  const [selectedSampleForPrediction, setSelectedSampleForPrediction] = useState<SampleImageItem | null>(null);

  useEffect(() => {
    async function checkHealth() {
      const res = await fetchHealth();
      setBackendOnline(res.status === 'healthy');
    }
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  const handleSelectSampleAndClassify = (sample: SampleImageItem) => {
    setSelectedSampleForPrediction(sample);
    setActiveTab('classify');
  };

  return (
    <div className="flex min-h-screen bg-slate-50 text-slate-800 font-sans">
      {/* Sidebar Navigation */}
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        backendOnline={backendOnline}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        <Navbar
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          backendOnline={backendOnline}
        />

        <main className="flex-1 p-6 md:p-8 max-w-7xl w-full mx-auto">
          {activeTab === 'overview' && (
            <Overview setActiveTab={setActiveTab} />
          )}

          {activeTab === 'dataset' && (
            <DatasetExplorer onSelectSampleForPrediction={handleSelectSampleAndClassify} />
          )}

          {activeTab === 'architectures' && (
            <ModelArchitectures />
          )}

          {activeTab === 'analytics' && (
            <PerformanceAnalytics />
          )}

          {activeTab === 'classify' && (
            <LiveClassification initialSample={selectedSampleForPrediction} />
          )}

          {activeTab === 'methodology' && (
            <Methodology />
          )}

          {activeTab === 'project-info' && (
            <ProjectInformation />
          )}
        </main>

        {/* Footer */}
        <footer className="border-t border-slate-200 bg-white py-4 px-8 text-center text-xs text-slate-400">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-2 max-w-7xl mx-auto">
            <span>DeepSatVision &copy; 2026 &mdash; Amity University Uttar Pradesh, Noida</span>
            <span>Students: Farhan Ahmed Siddique &amp; Adesh Pal &mdash; Guide: Dr Aakanshi Gupta</span>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default App;
