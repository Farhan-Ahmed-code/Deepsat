import React from 'react';
import { 
  LayoutDashboard, 
  Database, 
  Cpu, 
  LineChart, 
  Sparkles, 
  GitFork, 
  FileText, 
  Satellite
} from 'lucide-react';

interface SidebarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  backendOnline: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({ activeTab, setActiveTab, backendOnline }) => {
  const navItems = [
    { id: 'overview', label: 'Overview', icon: LayoutDashboard },
    { id: 'dataset', label: 'Dataset Explorer', icon: Database },
    { id: 'architectures', label: 'Model Architectures', icon: Cpu },
    { id: 'analytics', label: 'Performance Analysis', icon: LineChart },
    { id: 'classify', label: 'Live Classification', icon: Sparkles, isAction: true },
    { id: 'methodology', label: 'Methodology', icon: GitFork },
    { id: 'project-info', label: 'Project Details', icon: FileText },
  ];

  return (
    <aside className="w-64 bg-white border-r border-slate-200 flex flex-col h-screen sticky top-0 shrink-0 z-30 select-none">
      {/* Brand & University Header */}
      <div className="p-5 border-b border-slate-200">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-white shrink-0">
            <Satellite className="w-4 h-4" />
          </div>
          <div>
            <h1 className="text-base font-bold text-slate-900 leading-tight">
              DeepSat<span className="text-blue-600">Vision</span>
            </h1>
            <p className="text-[11px] text-slate-500 font-normal">
              Satellite Land Use AI
            </p>
          </div>
        </div>

        <div className="mt-3 pt-3 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-500">
          <span>Amity University</span>
          <span className="font-medium text-slate-600">ETMN100</span>
        </div>
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        <div className="px-3 pb-2 text-[11px] font-semibold tracking-wider text-slate-400 uppercase">
          Menu
        </div>

        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-xs transition-colors text-left ${
                isActive
                  ? 'bg-blue-50 text-blue-700 font-semibold border-l-4 border-blue-600'
                  : item.isAction
                  ? 'text-blue-600 hover:bg-blue-50/60 font-medium'
                  : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900 font-normal'
              }`}
            >
              <Icon className={`w-4 h-4 shrink-0 ${isActive ? 'text-blue-600' : 'text-slate-400'}`} />
              <span className="truncate">{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Footer Info */}
      <div className="p-4 border-t border-slate-200 text-[11px] text-slate-500 space-y-1">
        <div className="flex items-center justify-between">
          <span>Inference Engine:</span>
          <span className="flex items-center gap-1 font-medium text-slate-700">
            <span className={`w-2 h-2 rounded-full ${backendOnline ? 'bg-emerald-500' : 'bg-amber-500'}`} />
            {backendOnline ? 'Online' : 'Local'}
          </span>
        </div>
        <div className="text-[10px] text-slate-400 pt-1">
          B.Tech CSE (Data Science) • Sem VII
        </div>
      </div>
    </aside>
  );
};
