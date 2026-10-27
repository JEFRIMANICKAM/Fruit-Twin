import React from 'react';
import { Layers, ShieldCheck, Cpu, Leaf } from 'lucide-react';

export default function Footer({ setActiveTab }) {
  return (
    <footer className="glass-card border-t border-slate-800 mt-20 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
          
          <div>
            <div className="flex items-center gap-3 mb-3">
              <div className="w-8 h-8 rounded-lg bg-emerald-500/20 flex items-center justify-center border border-emerald-500/30">
                <Layers className="w-4 h-4 text-emerald-400" />
              </div>
              <span className="font-bold text-lg text-white font-mono">FruitTwin</span>
            </div>
            <p className="text-slate-400 text-sm mb-2 font-medium">
              "See the fruit today. Predict its future."
            </p>
            <p className="text-slate-500 text-xs leading-relaxed">
              AI-Powered Multi-Fruit Ripening Prediction, Plant Foliage Guard & Future Visual Simulation framework across 8 agricultural fruits: Mango, Banana, Papaya, Apple, Orange, Guava, Pomegranate, and Watermelon.
            </p>
          </div>

          <div>
            <h4 className="text-white text-sm font-semibold mb-3 flex items-center gap-2">
              <Cpu className="w-4 h-4 text-emerald-400" /> Research & Architecture
            </h4>
            <ul className="space-y-2 text-xs text-slate-400">
              <li>• Supported Catalog: <strong className="text-emerald-400">8 Fruits (🥭 🍌 🍈 🍎 🍊 🍐 🔴 🍉)</strong></li>
              <li>• Plant Guard: Auto-rejects foliage & prompts for harvested fruit</li>
              <li>• Out-of-Dataset AI: Identifies unsupported exotic fruits (Dragon Fruit, etc.)</li>
              <li>• Generative AI: Conditional GAN Stage Synthesizer across cultivars</li>
            </ul>
          </div>

          <div>
            <h4 className="text-white text-sm font-semibold mb-3 flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-emerald-400" /> Integrity & Disclaimers
            </h4>
            <p className="text-slate-500 text-xs leading-relaxed mb-3">
              FruitTwin enforces automated botanical verification, leaf detection, and variety mismatch correction to ensure reliable post-harvest supply chain decision support.
            </p>
            <div className="flex gap-4 text-xs font-medium text-emerald-400">
              <button onClick={() => setActiveTab('analyze')} className="hover:underline">Analyze Fruit</button>
              <button onClick={() => setActiveTab('simulation')} className="hover:underline">GAN Simulation</button>
              <button onClick={() => setActiveTab('about')} className="hover:underline">About Project</button>
            </div>
          </div>

        </div>

        <div className="pt-6 border-t border-slate-800/80 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-500 gap-4">
          <p>© {new Date().getFullYear()} FruitTwin Multi-Fruit & Botanical Guard AI Platform. 8 Fruits Supported.</p>
          <div className="flex items-center gap-4">
            <span className="bg-slate-800 px-2.5 py-1 rounded-md border border-slate-700 text-slate-300">FastAPI + React 19</span>
            <span className="bg-slate-800 px-2.5 py-1 rounded-md border border-slate-700 text-slate-300">PyTorch & Scikit-Learn</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
