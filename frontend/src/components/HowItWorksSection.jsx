import React from 'react';
import { Camera, Clock, Sparkles, ArrowRight, Thermometer, Binary, ShieldCheck } from 'lucide-react';

export default function HowItWorksSection() {
  return (
    <section className="my-16">
      <div className="text-center max-w-3xl mx-auto mb-12">
        <span className="text-emerald-400 font-mono text-xs font-semibold uppercase tracking-wider bg-emerald-500/10 px-3 py-1 rounded-full border border-emerald-500/20">
          Research Architecture
        </span>
        <h2 className="text-3xl font-extrabold text-white mt-3 font-mono">
          How FruitTwin Thinks
        </h2>
        <p className="text-slate-400 text-sm mt-2">
          FruitTwin decouples ripeness classification, timeline regression, and visual synthesis into three distinct AI modules tailored for multi-fruit (Mango, Banana, Papaya, Apple) biochemical dynamics.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        
        {/* Component 1: CV / CNN */}
        <div className="glass-card glass-card-hover rounded-2xl p-6 border border-slate-800 relative overflow-hidden flex flex-col justify-between">
          <div className="absolute top-0 right-0 translate-x-4 -translate-y-4 w-24 h-24 bg-emerald-500/10 rounded-full blur-2xl" />
          <div>
            <div className="w-12 h-12 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400 mb-4">
              <Camera className="w-6 h-6" />
            </div>
            <span className="text-emerald-400 font-mono text-xs uppercase font-bold tracking-wider">Module 1</span>
            <h3 className="text-xl font-bold text-white mt-1">CURRENT STATE</h3>
            <p className="text-xs font-semibold text-emerald-400/90 mb-3">Computer Vision / CNN</p>
            <p className="text-slate-400 text-xs leading-relaxed">
              Analyzes surface texture, hue ratio (green-to-yellow transition), and physical skin indicators to determine current stage.
            </p>
          </div>

          <div className="mt-6 pt-4 border-t border-slate-800/80 bg-slate-900/60 -mx-6 -mb-6 p-4 rounded-b-2xl font-mono text-xs">
            <div className="flex items-center justify-between text-slate-300">
              <span className="bg-slate-800 px-2 py-1 rounded border border-slate-700">IMAGE</span>
              <ArrowRight className="w-4 h-4 text-emerald-400" />
              <span className="bg-emerald-950/80 text-emerald-400 px-2 py-1 rounded border border-emerald-500/30 font-bold">CURRENT RIPENESS</span>
            </div>
          </div>
        </div>

        {/* Component 2: ML Regression */}
        <div className="glass-card glass-card-hover rounded-2xl p-6 border border-slate-800 relative overflow-hidden flex flex-col justify-between">
          <div className="absolute top-0 right-0 translate-x-4 -translate-y-4 w-24 h-24 bg-amber-500/10 rounded-full blur-2xl" />
          <div>
            <div className="w-12 h-12 rounded-xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center text-amber-400 mb-4">
              <Clock className="w-6 h-6" />
            </div>
            <span className="text-amber-400 font-mono text-xs uppercase font-bold tracking-wider">Module 2</span>
            <h3 className="text-xl font-bold text-white mt-1">FUTURE TIME</h3>
            <p className="text-xs font-semibold text-amber-400/90 mb-3">Machine Learning Regressor</p>
            <p className="text-slate-400 text-xs leading-relaxed">
              Predicts estimated remaining days to peak ripeness and total shelf life using stage features combined with room temperature.
            </p>
          </div>

          <div className="mt-6 pt-4 border-t border-slate-800/80 bg-slate-900/60 -mx-6 -mb-6 p-4 rounded-b-2xl font-mono text-xs">
            <div className="flex items-center justify-between text-slate-300">
              <span className="bg-slate-800 px-1.5 py-1 rounded border border-slate-700 text-[10px]">STAGE + TEMP</span>
              <ArrowRight className="w-4 h-4 text-amber-400" />
              <span className="bg-amber-950/80 text-amber-400 px-2 py-1 rounded border border-amber-500/30 font-bold">RIPENING TIME</span>
            </div>
          </div>
        </div>

        {/* Component 3: Conditional GAN */}
        <div className="glass-card glass-card-hover rounded-2xl p-6 border border-slate-800 relative overflow-hidden flex flex-col justify-between">
          <div className="absolute top-0 right-0 translate-x-4 -translate-y-4 w-24 h-24 bg-purple-500/10 rounded-full blur-2xl" />
          <div>
            <div className="w-12 h-12 rounded-xl bg-purple-500/10 border border-purple-500/20 flex items-center justify-center text-purple-400 mb-4">
              <Sparkles className="w-6 h-6" />
            </div>
            <span className="text-purple-400 font-mono text-xs uppercase font-bold tracking-wider">Module 3</span>
            <h3 className="text-xl font-bold text-white mt-1">FUTURE APPEARANCE</h3>
            <p className="text-xs font-semibold text-purple-400/90 mb-3">Conditional GAN</p>
            <p className="text-slate-400 text-xs leading-relaxed">
              Generates a plausible visual projection of how the fruit will look at subsequent target ripening stages, accounting for varietal pigmentation.
            </p>
          </div>

          <div className="mt-6 pt-4 border-t border-slate-800/80 bg-slate-900/60 -mx-6 -mb-6 p-4 rounded-b-2xl font-mono text-xs">
            <div className="flex items-center justify-between text-slate-300">
              <span className="bg-slate-800 px-1.5 py-1 rounded border border-slate-700 text-[10px]">IMAGE + TARGET</span>
              <ArrowRight className="w-4 h-4 text-purple-400" />
              <span className="bg-purple-950/80 text-purple-400 px-2 py-1 rounded border border-purple-500/30 font-bold">FUTURE APPEARANCE</span>
            </div>
          </div>
        </div>

      </div>
    </section>
  );
}
