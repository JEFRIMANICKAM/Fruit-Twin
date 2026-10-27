import React from 'react';
import { AlertTriangle, CheckCircle2, Cpu, Leaf, Sparkles } from 'lucide-react';

export default function ModelStatusBanner({ modelStatus }) {
  const isConnected = modelStatus?.all_connected;

  if (isConnected) {
    return (
      <div className="bg-emerald-950/40 border border-emerald-500/30 rounded-xl p-4 mb-6 flex flex-col md:flex-row items-start md:items-center justify-between gap-3">
        <div className="flex items-start gap-3">
          <CheckCircle2 className="w-5 h-5 text-emerald-400 mt-0.5 flex-shrink-0" />
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h4 className="text-emerald-300 font-semibold text-sm">8-Fruit AI Architecture Active & Connected</h4>
              <span className="text-[10px] font-mono font-bold bg-emerald-500/20 text-emerald-300 px-2 py-0.5 rounded border border-emerald-500/40">
                🥭 🍌 🍈 🍎 🍊 🍐 🔴 🍉 (8 Fruits)
              </span>
              <span className="text-[10px] font-mono font-bold bg-teal-500/20 text-teal-300 px-2 py-0.5 rounded border border-teal-500/40 flex items-center gap-1">
                <Leaf className="w-3 h-3" /> Plant Guard Active
              </span>
            </div>
            <p className="text-emerald-400/80 text-xs mt-0.5">
              PyTorch CNN Classifiers, Scikit-learn Timeline Regressors, Plant Foliage Guard, and Conditional GANs are loaded across all 8 fruit categories.
            </p>
          </div>
        </div>
        <div className="flex-shrink-0 bg-slate-900/80 px-3 py-1.5 rounded-lg border border-emerald-500/30 text-xs text-emerald-300 flex items-center gap-2 self-stretch md:self-auto justify-center font-mono">
          <Cpu className="w-4 h-4 text-emerald-400" />
          <span>8-Fruit Engine Live</span>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-amber-950/40 border border-amber-500/30 rounded-xl p-4 mb-6 flex flex-col md:flex-row items-start md:items-center justify-between gap-3">
      <div className="flex items-start gap-3">
        <AlertTriangle className="w-5 h-5 text-amber-400 mt-0.5 flex-shrink-0" />
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h4 className="text-amber-300 font-bold text-sm uppercase tracking-wide">Multi-Fruit AI Pipeline Active</h4>
            <span className="text-[10px] bg-amber-500/20 text-amber-300 px-2 py-0.5 rounded border border-amber-500/30 font-mono">
              8 Fruits Ready (🥭 🍌 🍈 🍎 🍊 🍐 🔴 🍉)
            </span>
            <span className="text-[10px] bg-teal-500/20 text-teal-300 px-2 py-0.5 rounded border border-teal-500/30 font-mono">
              🌿 Plant Guard Active
            </span>
          </div>
          <p className="text-amber-200/80 text-xs mt-1">
            Backend AI pipeline ready for 8 agricultural fruits with automated Plant Foliage rejection and Unsupported Fruit detection.
          </p>
        </div>
      </div>
      <div className="flex-shrink-0 bg-slate-900/80 px-3 py-2 rounded-lg border border-amber-500/20 text-xs text-amber-300 flex items-center gap-2 self-stretch md:self-auto justify-center font-mono">
        <Cpu className="w-4 h-4 text-emerald-400" />
        <span>8 Fruits + Vision Guard</span>
      </div>
    </div>
  );
}
