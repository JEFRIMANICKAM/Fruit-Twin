import React from 'react';
import { Info, Cpu, Layers, Sparkles, CheckCircle2, ShieldCheck, Zap } from 'lucide-react';
import HowItWorksSection from '../components/HowItWorksSection';

export default function AboutPage() {
  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8 font-mono">
      
      {/* Header */}
      <div className="text-center max-w-3xl mx-auto mb-12">
        <span className="text-amber-400 font-mono text-xs font-semibold uppercase tracking-wider bg-amber-500/10 px-3 py-1 rounded-full border border-amber-500/20">
          Research Framework
        </span>
        <h1 className="text-3xl sm:text-5xl font-black text-white mt-3 font-mono">
          About FruitTwin
        </h1>
        <p className="text-amber-300 font-bold text-lg mt-2 font-sans">
          "See the fruit today. Predict its future."
        </p>
      </div>

      <div className="glass-card rounded-3xl p-8 sm:p-12 border border-slate-800 space-y-8 mb-12">
        
        <div>
          <h2 className="text-2xl font-bold text-white font-mono mb-3">
            Project Overview
          </h2>
          <p className="text-slate-300 text-sm sm:text-base leading-relaxed font-sans">
            FruitTwin is an agricultural intelligence framework for multi-fruit ripening prediction and future visual simulation. Conventional computer vision systems stop at simple discrete classification (e.g. labeling a fruit as "ripe" or "unripe"). FruitTwin extends this paradigm across <strong className="text-emerald-400">Mango</strong>, <strong className="text-amber-400">Banana</strong>, <strong className="text-orange-400">Papaya</strong>, and <strong className="text-rose-400">Apple</strong> to answer three essential agricultural & supply-chain questions:
          </p>

          <ul className="mt-4 space-y-2 text-sm text-slate-300 font-mono">
            <li className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>1. What is the fruit's current ripeness stage? (Unripe, Semi-Ripe, Ripe, Overripe)</span>
            </li>
            <li className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-amber-400" />
              <span>2. How long will it take to reach the next stage under ambient storage temperature?</span>
            </li>
            <li className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-purple-400" />
              <span>3. What will the fruit look like visually in future ripening stages?</span>
            </li>
          </ul>
        </div>

        <div className="pt-6 border-t border-slate-800">
          <h3 className="text-xl font-bold text-white font-mono mb-3 flex items-center gap-2">
            <Cpu className="w-5 h-5 text-emerald-400" />
            AI Component Decoupling
          </h3>
          <p className="text-slate-400 text-xs leading-relaxed mb-4 font-sans">
            FruitTwin maintains strict boundaries between visual state analysis, temporal prediction, and visual synthesis:
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-mono">
            <div className="bg-slate-900/90 p-4 rounded-xl border border-slate-800">
              <span className="text-emerald-400 font-bold block mb-1">Computer Vision / CNN</span>
              <p className="text-slate-300 font-sans">Classifies current state from skin RGB/HSV color vectors, carotenoid indices, and surface texture features.</p>
            </div>
            <div className="bg-slate-900/90 p-4 rounded-xl border border-slate-800">
              <span className="text-amber-400 font-bold block mb-1">Machine Learning Regressor</span>
              <p className="text-slate-300 font-sans">Calculates estimated remaining ripening days & shelf life using stage features + Arrhenius thermal dynamics (°C).</p>
            </div>
            <div className="bg-slate-900/90 p-4 rounded-xl border border-slate-800">
              <span className="text-purple-400 font-bold block mb-1">Conditional GAN</span>
              <p className="text-slate-300 font-sans">Generates plausible visual future appearance for target ripening stages across all 4 supported fruits.</p>
            </div>
          </div>
        </div>

        <div className="pt-6 border-t border-slate-800">
          <h3 className="text-xl font-bold text-white font-mono mb-3 flex items-center gap-2">
            <ShieldCheck className="w-5 h-5 text-emerald-400" />
            Multi-Fruit Enterprise Architecture
          </h3>
          <div className="bg-slate-900/80 p-5 rounded-2xl border border-emerald-500/30 space-y-2">
            <p className="text-xs text-slate-300 leading-relaxed font-sans">
              <strong className="text-emerald-400 font-mono">Full 4-Fruit Operational Support:</strong> The system actively supports <strong className="text-emerald-300">Mango (மாம்பழம்)</strong>, <strong className="text-amber-300">Banana (வாழைப்பழம்)</strong>, <strong className="text-orange-300">Papaya (பப்பாளி)</strong>, and <strong className="text-rose-300">Apple (ஆப்பிள்)</strong> across 18+ regional Indian cultivars with cross-fruit safety detection.
            </p>
            <p className="text-xs text-slate-400 leading-relaxed font-sans">
              <strong className="text-slate-300 font-mono">Decoupled Namespaces:</strong> Backend model architecture is modularly namespaced under <code className="bg-slate-800 px-1 text-emerald-400 font-mono">models/mango/</code>, <code className="bg-slate-800 px-1 text-amber-400 font-mono">models/banana/</code>, <code className="bg-slate-800 px-1 text-orange-400 font-mono">models/papaya/</code>, and <code className="bg-slate-800 px-1 text-rose-400 font-mono">models/apple/</code>.
            </p>
          </div>
        </div>

      </div>

      <HowItWorksSection />

    </div>
  );
}
