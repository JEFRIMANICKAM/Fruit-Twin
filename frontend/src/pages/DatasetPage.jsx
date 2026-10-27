import React, { useState, useEffect } from 'react';
import { Database, ShieldCheck, Layers, RefreshCw, Server, Zap, Scale, Activity, FileText } from 'lucide-react';
import { fetchDatasetSummary } from '../services/api';
import { INDIAN_MANGO_VARIETIES } from '../services/mangoVarieties';

export default function DatasetPage() {
  const [datasetData, setDatasetData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    async function loadDataset() {
      try {
        const data = await fetchDatasetSummary();
        setDatasetData(data);
      } catch (err) {
        console.error('Failed to load dataset:', err);
      } finally {
        setIsLoading(false);
      }
    }
    loadDataset();
  }, []);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 font-mono">
      
      {/* Header */}
      <div className="text-center max-w-3xl mx-auto mb-10">
        <span className="text-cyan-400 text-xs font-bold uppercase tracking-widest bg-cyan-500/10 px-3.5 py-1.5 rounded-full border border-cyan-500/30">
          Indian & Tamil Mango Research Dataset
        </span>
        <h1 className="text-3xl sm:text-4xl font-black text-white mt-3 font-mono tracking-tight">
          IoT Sensor & Biochemical Dataset
        </h1>
        <p className="text-slate-400 text-xs mt-2 font-sans">
          Experimental dataset evaluating physical, biochemical, and parallel-plate capacitance sensor parameters across famous Indian & Tamil mango varieties (Alphonso, Neelum, Banganapalli, Malgova, Totapuri, Kesar).
        </p>
      </div>

      {/* Indian Varieties Grid */}
      <div className="mb-10">
        <h3 className="text-base font-bold text-white mb-4 flex items-center gap-2">
          <span>🥭</span> Indian & Tamil Mango Varieties Catalog (மாம்பழ வகைகள்)
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {INDIAN_MANGO_VARIETIES.map(v => (
            <div key={v.id} className="glass-card rounded-xl p-4 border border-slate-800 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-bold text-white">{v.name}</span>
                <span className="text-xs font-bold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20 font-sans">
                  {v.tamilName}
                </span>
              </div>
              <p className="text-[11px] text-amber-400 font-bold">{v.origin}</p>
              <p className="text-[11px] text-slate-400 font-sans leading-relaxed">{v.desc}</p>
              <div className="pt-2 border-t border-slate-800 flex justify-between text-[10px] text-slate-400">
                <span>Avg Sweetness: {v.defaultTss} °Brix</span>
                <span>Acidity: {v.defaultTa}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {isLoading ? (
        <div className="text-center py-16 text-slate-400 flex items-center justify-center gap-3">
          <RefreshCw className="w-5 h-5 animate-spin text-emerald-400" />
          <span>Loading dataset & sensor metrics from disk...</span>
        </div>
      ) : (
        <div className="space-y-8">
          
          {/* Summary Metric Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
            <div className="glass-card rounded-xl p-4 border border-slate-800 text-center">
              <span className="text-xs text-slate-400 uppercase font-bold">Varieties</span>
              <p className="text-xl font-bold text-emerald-400 mt-1">Alphonso, Neelum + 4</p>
              <p className="text-[11px] text-slate-500 mt-1">Indian & Tamil Cultivars</p>
            </div>

            <div className="glass-card rounded-xl p-4 border border-slate-800 text-center">
              <span className="text-xs text-slate-400 uppercase font-bold">Total Samples</span>
              <p className="text-2xl font-bold text-white mt-1">
                {datasetData?.total_samples || 200}
              </p>
              <p className="text-[11px] text-slate-500 mt-1">100 Train + 20 Validation</p>
            </div>

            <div className="glass-card rounded-xl p-4 border border-slate-800 text-center">
              <span className="text-xs text-slate-400 uppercase font-bold">Maturation Stages</span>
              <p className="text-xl font-bold text-amber-400 mt-1">
                80 - 110 DAFS
              </p>
              <p className="text-[11px] text-slate-500 mt-1">Days After Fruit Set</p>
            </div>

            <div className="glass-card rounded-xl p-4 border border-slate-800 text-center">
              <span className="text-xs text-slate-400 uppercase font-bold">Avg Sweetness (TSS)</span>
              <p className="text-2xl font-bold text-cyan-400 mt-1">
                {datasetData?.tss_stats?.mean || '10.2'} °Brix
              </p>
              <p className="text-[11px] text-slate-500 mt-1">Range: 6.0 – 18.0 °Brix</p>
            </div>
          </div>


          {/* Dataset Table */}
          <div className="glass-card rounded-2xl p-6 border border-slate-800">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 mb-4">
              <div>
                <h3 className="text-lg font-bold text-white flex items-center gap-2">
                  <Server className="w-5 h-5 text-emerald-400" />
                  Train_Data.csv Inspection Table
                </h3>
                <p className="text-slate-400 text-xs mt-0.5">
                  Location: D:\IOT LAB\GAN FRUIT RIPENESS\MANGO\Train_Data.csv
                </p>
              </div>
              <span className="text-xs text-slate-400 bg-slate-950 px-3 py-1 rounded border border-slate-800">
                Top 15 Records
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 bg-slate-950">
                    <th className="p-3">Storage Time</th>
                    <th className="p-3">DAFS</th>
                    <th className="p-3">No</th>
                    <th className="p-3">W (g)</th>
                    <th className="p-3">V (mV)</th>
                    <th className="p-3">C (pF)</th>
                    <th className="p-3">R, G, B</th>
                    <th className="p-3">TA (%)</th>
                    <th className="p-3">TSS (°Brix)</th>
                    <th className="p-3">Stage</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/80">
                  {datasetData?.sample_train_records?.map((row, idx) => (
                    <tr key={idx} className="hover:bg-slate-900/60 text-slate-300">
                      <td className="p-3 font-bold text-amber-400">Day {row["Storage Time"]}</td>
                      <td className="p-3 text-emerald-400">{row.DAFS} DAFS</td>
                      <td className="p-3 text-slate-400">#{row.No}</td>
                      <td className="p-3 font-bold">{row.W}g</td>
                      <td className="p-3 text-slate-400">{row.V}</td>
                      <td className="p-3 font-bold text-cyan-400">{Number(row.C).toFixed(3)}</td>
                      <td className="p-3 text-slate-400">{row.R}, {row.G}, {row.B}</td>
                      <td className="p-3 text-rose-400">{Number(row.TA).toFixed(2)}%</td>
                      <td className="p-3 font-bold text-yellow-400">{Number(row.TSS).toFixed(1)}°</td>
                      <td className="p-3">
                        <span className={`px-2.5 py-0.5 rounded text-[10px] font-bold ${
                          row.ripeness_stage === 'Unripe' ? 'bg-emerald-950 text-emerald-400 border border-emerald-500/30' :
                          row.ripeness_stage === 'Semi-Ripe' ? 'bg-yellow-950 text-yellow-300 border border-yellow-500/30' :
                          row.ripeness_stage === 'Ripe' ? 'bg-amber-950 text-amber-400 border border-amber-500/30' :
                          'bg-rose-950 text-rose-400 border border-rose-500/30'
                        }`}>
                          {row.ripeness_stage}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

        </div>
      )}

    </div>
  );
}
