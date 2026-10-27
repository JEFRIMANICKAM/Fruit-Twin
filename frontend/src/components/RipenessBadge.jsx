import React from 'react';

export default function RipenessBadge({ stage, size = 'normal' }) {
  if (!stage) {
    return (
      <span className="px-3 py-1 rounded-full text-xs font-semibold bg-slate-800 text-slate-400 border border-slate-700">
        Unconnected
      </span>
    );
  }

  const styles = {
    'Unripe': 'bg-emerald-950/80 text-emerald-400 border-emerald-500/40 shadow-emerald-500/10',
    'Semi-Ripe': 'bg-yellow-950/80 text-yellow-300 border-yellow-500/40 shadow-yellow-500/10',
    'Ripe': 'bg-amber-950/80 text-amber-400 border-amber-500/40 shadow-amber-500/10',
    'Overripe': 'bg-rose-950/80 text-rose-400 border-rose-500/40 shadow-rose-500/10',
  };

  const currentStyle = styles[stage] || 'bg-slate-800 text-slate-300 border-slate-700';

  const sizeClasses = size === 'large' 
    ? 'px-4 py-1.5 text-base font-bold' 
    : 'px-3 py-1 text-xs font-bold';

  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border shadow-sm ${sizeClasses} ${currentStyle}`}>
      <span className="w-2 h-2 rounded-full bg-current animate-pulse" />
      {stage}
    </span>
  );
}
