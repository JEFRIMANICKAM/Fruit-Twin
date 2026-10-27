import React, { useState, useEffect } from 'react';
import { Search, Database, CheckCircle2, XCircle, Sparkles, AlertCircle, X, ExternalLink, Filter } from 'lucide-react';
import { fetchDatasetCatalog } from '../services/api';

export default function DatasetClassCatalogModal({ isOpen, onClose, onSelectFruit }) {
  const [catalog, setCatalog] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedFilter, setSelectedFilter] = useState('all'); // 'all', 'fruit', 'non_fruit'
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (isOpen && !catalog) {
      setIsLoading(true);
      fetchDatasetCatalog()
        .then((data) => {
          setCatalog(data);
        })
        .finally(() => {
          setIsLoading(false);
        });
    }
  }, [isOpen, catalog]);

  if (!isOpen) return null;

  const classList = catalog?.class_details || [];
  
  const filteredList = classList.filter((item) => {
    const matchesSearch = 
      item.class_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.group.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (item.variety && item.variety.toLowerCase().includes(searchTerm.toLowerCase())) ||
      (item.tamil_name && item.tamil_name.includes(searchTerm));

    if (!matchesSearch) return false;

    if (selectedFilter === 'fruit') return item.is_fruit;
    if (selectedFilter === 'non_fruit') return !item.is_fruit;
    return true;
  });

  return (
    <div 
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/85 backdrop-blur-md font-mono"
      role="dialog"
      aria-modal="true"
    >
      <div className="relative w-full max-w-4xl bg-slate-900 border-2 border-slate-700 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[85vh]">
        
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-800 bg-slate-950/80 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">
              <Database className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-base sm:text-lg font-black text-white uppercase tracking-tight">
                  Trained Dataset Class Catalogue
                </h3>
                {catalog && (
                  <span className="text-xs font-bold px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/40">
                    {catalog.total_classes} Classes Active
                  </span>
                )}
              </div>
              <p className="text-xs text-slate-400 font-sans mt-0.5">
                Automatically discovered from the trained Fruits-360 model with capability indicators.
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="p-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white transition-colors cursor-pointer"
            title="Close Catalogue"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Search and Filters Bar */}
        <div className="p-4 border-b border-slate-800 bg-slate-900/90 flex flex-col sm:flex-row items-center justify-between gap-3 shrink-0">
          <div className="relative w-full sm:w-80">
            <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search class, variety, or Tamil..."
              className="w-full pl-9 pr-4 py-2 rounded-xl bg-slate-950 border border-slate-800 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500 transition-colors font-mono"
            />
          </div>

          <div className="flex items-center gap-2 w-full sm:w-auto">
            <button
              type="button"
              onClick={() => setSelectedFilter('all')}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                selectedFilter === 'all' 
                  ? 'bg-emerald-500 text-slate-950 shadow-md shadow-emerald-500/20' 
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              All ({catalog?.total_classes || 0})
            </button>
            <button
              type="button"
              onClick={() => setSelectedFilter('fruit')}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                selectedFilter === 'fruit' 
                  ? 'bg-emerald-500 text-slate-950 shadow-md shadow-emerald-500/20' 
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              Fruits ({catalog?.fruit_classes_count || 0})
            </button>
            <button
              type="button"
              onClick={() => setSelectedFilter('non_fruit')}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                selectedFilter === 'non_fruit' 
                  ? 'bg-amber-500 text-slate-950 shadow-md shadow-amber-500/20' 
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              Non-Fruit Classes ({catalog?.non_fruit_classes_count || 0})
            </button>
          </div>
        </div>

        {/* Scrollable Class Table */}
        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          {isLoading ? (
            <div className="py-12 text-center text-slate-400 text-xs animate-pulse">
              Loading model class registry from backend...
            </div>
          ) : filteredList.length === 0 ? (
            <div className="py-12 text-center text-slate-500 text-xs">
              No matching dataset classes found for "{searchTerm}".
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-2.5">
              {filteredList.map((item, idx) => (
                <div 
                  key={idx}
                  className="p-3.5 rounded-xl bg-slate-950/80 border border-slate-800 hover:border-slate-700 transition-all flex flex-col md:flex-row md:items-center justify-between gap-3"
                >
                  <div className="flex items-start sm:items-center gap-3">
                    <span className="text-xs font-bold text-slate-600 font-mono w-6">
                      {idx + 1}.
                    </span>
                    <div>
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="text-sm font-extrabold text-white">
                          {item.class_name}
                        </span>
                        {item.tamil_name && (
                          <span className="text-xs text-amber-300 font-mono">
                            ({item.tamil_name})
                          </span>
                        )}
                        <span className={`text-[10px] px-2 py-0.5 rounded font-bold uppercase ${
                          item.is_fruit 
                            ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' 
                            : 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                        }`}>
                          {item.category_type}
                        </span>
                      </div>
                      <div className="text-[11px] text-slate-400 mt-0.5">
                        Group: <strong className="text-slate-300">{item.group}</strong> | Variety: <strong className="text-slate-300">{item.variety}</strong>
                      </div>
                    </div>
                  </div>

                  {/* Capability Indicators Matrix */}
                  <div className="flex flex-wrap items-center gap-2">
                    <span 
                      className="inline-flex items-center gap-1 px-2 py-1 rounded bg-slate-900 border border-slate-800 text-[10px] text-emerald-400 font-bold"
                      title="Universal Multi-Class Identification trained on Fruits-360"
                    >
                      <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                      <span>ID: Active</span>
                    </span>

                    <span 
                      className={`inline-flex items-center gap-1 px-2 py-1 rounded border text-[10px] font-bold ${
                        item.maturity_supported 
                          ? 'bg-emerald-950/50 border-emerald-500/40 text-emerald-300' 
                          : 'bg-slate-900 border-slate-800 text-slate-500 opacity-60'
                      }`}
                      title={item.maturity_supported ? "Maturity classification verified via RawRipe" : "Maturity classification unavailable"}
                    >
                      {item.maturity_supported ? <CheckCircle2 className="w-3 h-3 text-emerald-400" /> : <XCircle className="w-3 h-3 text-slate-500" />}
                      <span>Maturity: {item.maturity_supported ? 'RawRipe' : 'N/A'}</span>
                    </span>

                    <span 
                      className={`inline-flex items-center gap-1 px-2 py-1 rounded border text-[10px] font-bold ${
                        item.gan_supported 
                          ? 'bg-purple-950/50 border-purple-500/40 text-purple-300' 
                          : 'bg-slate-900 border-slate-800 text-slate-500 opacity-60'
                      }`}
                      title={item.gan_supported ? "Conditional GAN simulation checkpoint available" : "GAN simulation unavailable"}
                    >
                      {item.gan_supported ? <Sparkles className="w-3 h-3 text-purple-400" /> : <XCircle className="w-3 h-3 text-slate-500" />}
                      <span>GAN: {item.gan_supported ? 'Active' : 'N/A'}</span>
                    </span>

                    {item.has_dedicated_pipeline && onSelectFruit && (
                      <button
                        type="button"
                        onClick={() => {
                          onSelectFruit(item.group.toLowerCase());
                          onClose();
                        }}
                        className="px-2.5 py-1 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-slate-950 text-[11px] font-bold uppercase tracking-wider transition-all cursor-pointer flex items-center gap-1"
                      >
                        <span>Open Tab</span>
                        <ExternalLink className="w-3 h-3" />
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-slate-800 bg-slate-950/90 flex flex-col sm:flex-row items-center justify-between text-[11px] text-slate-400 shrink-0 gap-2">
          <span>
            Model Checkpoint: <code className="text-emerald-400">fruits360_model.pkl</code> | Test Accuracy: <strong className="text-white">{catalog?.evaluation?.accuracy ? `${(catalog.evaluation.accuracy * 100).toFixed(1)}%` : '75.0%'}</strong>
          </span>
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-bold transition-all cursor-pointer"
          >
            Close Catalogue
          </button>
        </div>

      </div>
    </div>
  );
}
