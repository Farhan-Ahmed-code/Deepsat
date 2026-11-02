import React from 'react';
import { Sparkles } from 'lucide-react';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  backendOnline: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab }) => {
  const getTabTitle = () => {
    switch (activeTab) {
      case 'overview':
        return 'Overview';
      case 'dataset':
        return 'Dataset Explorer';
      case 'architectures':
        return 'Model Architectures';
      case 'analytics':
        return 'Performance Analysis';
      case 'classify':
        return 'Live Classification';
      case 'methodology':
        return 'Methodology';
      case 'project-info':
        return 'Project Details';
      default:
        return 'DeepSatVision';
    }
  };

  return (
    <header className="h-16 bg-white border-b border-slate-200 px-6 flex items-center justify-between sticky top-0 z-20">
      <div>
        <h2 className="text-base font-semibold text-slate-900">
          {getTabTitle()}
        </h2>
        <p className="text-xs text-slate-500">
          B.Tech Minor Project • Amity University Uttar Pradesh
        </p>
      </div>

      <div className="flex items-center gap-3">
        {activeTab !== 'classify' && (
          <button
            onClick={() => setActiveTab('classify')}
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-xs font-medium transition-colors shadow-sm"
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Live Classification</span>
          </button>
        )}
      </div>
    </header>
  );
};
