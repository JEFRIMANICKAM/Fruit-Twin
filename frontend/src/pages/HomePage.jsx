import React from 'react';
import { 
  Scan, Sparkles, ArrowRight, ShieldCheck, Thermometer, Camera, 
  Activity, Layers, Cpu, CheckCircle2, Zap, Eye, Compass, RefreshCw, Leaf
} from 'lucide-react';
import ModelStatusBanner from '../components/ModelStatusBanner';
import HowItWorksSection from '../components/HowItWorksSection';
import { ALL_FRUITS } from '../services/allFruitVarieties';
import { MULTI_FRUIT_SAMPLES } from '../services/sampleImages';

export default function HomePage({ setActiveTab, navigateToFruit, modelStatus }) {
  // Quick test specimens from the multi-fruit dataset & diagnostic tests
  const quickTestSpecimens = [
    {
      fruitId: 'mango',
      sample: MULTI_FRUIT_SAMPLES.mango[0],
      name: 'Sindhoora Red Mango',
      tamil: 'சிந்தூரா மாம்பழம்',
      desc: 'Blushed South Indian cultivar with red-orange carotene blush.',
      badge: 'Red Blushed',
      badgeColor: 'border-rose-500/30 text-rose-400 bg-rose-500/10',
      icon: '🥭'
    },
    {
      fruitId: 'banana',
      sample: MULTI_FRUIT_SAMPLES.banana[0],
      name: 'Red Banana (Sevazhai)',
      tamil: 'செவ்வாழை வாழைப்பழம்',
      desc: 'Rare high-antioxidant crimson banana from Kanyakumari.',
      badge: 'Anthocyanin Rich',
      badgeColor: 'border-red-500/30 text-red-400 bg-red-500/10',
      icon: '🍌'
    },
    {
      fruitId: 'orange',
      sample: MULTI_FRUIT_SAMPLES.orange[1],
      name: 'Nagpur Orange',
      tamil: 'நாக்பூர் ஆரஞ்சு',
      desc: 'Vibrant sweet citrus with rich carotenoid & limonene aroma.',
      badge: 'High Vitamin C',
      badgeColor: 'border-orange-500/30 text-orange-400 bg-orange-500/10',
      icon: '🍊'
    },
    {
      fruitId: 'pomegranate',
      sample: MULTI_FRUIT_SAMPLES.pomegranate[1],
      name: 'Bhagwa Pomegranate',
      tamil: 'பக்வா மாதுளை',
      desc: 'Deep ruby crimson leathery rind packed with polyphenols.',
      badge: 'Ellagitannins',
      badgeColor: 'border-red-500/30 text-red-400 bg-red-500/10',
      icon: '🔴'
    },
    {
      fruitId: 'guava',
      sample: MULTI_FRUIT_SAMPLES.guava[1],
      name: 'Safeda Pale Guava',
      tamil: 'சபேடா கொய்யா',
      desc: 'Smooth sweet tropical guava with tender aromatic flesh.',
      badge: 'Pectin Softening',
      badgeColor: 'border-lime-500/30 text-lime-400 bg-lime-500/10',
      icon: '🍐'
    },
    {
      fruitId: 'watermelon',
      sample: MULTI_FRUIT_SAMPLES.watermelon[1],
      name: 'Kiran Watermelon',
      tamil: 'கிரண் தர்பூசணி',
      desc: 'Striped rind with rich golden yellow ground-spot maturation.',
      badge: 'Lycopene Rich',
      badgeColor: 'border-emerald-500/30 text-emerald-400 bg-emerald-500/10',
      icon: '🍉'
    },
    {
      fruitId: 'mango', // Diagnostic test for plant leaf
      sample: MULTI_FRUIT_SAMPLES.diagnostic[0],
      name: 'Plant / Leaf Test Specimen',
      tamil: 'பச்சை இலை சோதனை',
      desc: 'Tests AI vision detection: accurately recognizes vegetative foliage and prompts for real fruit.',
      badge: '🌿 Plant Test',
      badgeColor: 'border-emerald-500/50 text-emerald-300 bg-emerald-500/20',
      icon: '🌿'
    },
    {
      fruitId: 'mango', // Diagnostic test for unsupported fruit
      sample: MULTI_FRUIT_SAMPLES.diagnostic[1],
      name: 'Dragon Fruit (Out of Dataset)',
      tamil: 'டிராகன் பழம் சோதனை',
      desc: 'Tests AI out-of-dataset detection: identifies exotic unsupported fruit and guides to verified database.',
      badge: '🍇 Unsupported Test',
      badgeColor: 'border-purple-500/50 text-purple-300 bg-purple-500/20',
      icon: '🍇'
    }
  ];

  // Specific botanical and biochemical parameters for all 8 fruits
  const fruitProfiles = [
    {
      id: 'mango',
      name: 'Mango',
      tamilName: 'மாம்பழம்',
      latinName: 'Mangifera indica',
      icon: '🥭',
      color: 'emerald',
      accentBorder: 'border-emerald-500/30 hover:border-emerald-500/60',
      badgeColor: 'bg-emerald-500/20 text-emerald-300',
      cultivarsCount: '7 Cultivars',
      cultivarsList: ['Sindhoora', 'Alphonso', 'Neelum', 'Banganapalli', 'Malgova', 'Totapuri', 'Kesar'],
      optimalTemp: '24°C - 30°C',
      tssRange: '14.0 - 22.5° Brix',
      biomarker: 'Beta-Carotene & Chlorophyll Degradation'
    },
    {
      id: 'banana',
      name: 'Banana',
      tamilName: 'வாழைப்பழம்',
      latinName: 'Musa acuminata',
      icon: '🍌',
      color: 'amber',
      accentBorder: 'border-amber-500/30 hover:border-amber-500/60',
      badgeColor: 'bg-amber-500/20 text-amber-300',
      cultivarsCount: '5 Cultivars',
      cultivarsList: ['Red Sevazhai', 'Poovan', 'Nendran', 'Rasthali', 'Grand Naine'],
      optimalTemp: '20°C - 26°C',
      tssRange: '18.0 - 24.0° Brix',
      biomarker: 'Starch-to-Glucose & Sugar Spotting'
    },
    {
      id: 'papaya',
      name: 'Papaya',
      tamilName: 'பப்பாளி',
      latinName: 'Carica papaya',
      icon: '🍈',
      color: 'orange',
      accentBorder: 'border-orange-500/30 hover:border-orange-500/60',
      badgeColor: 'bg-orange-500/20 text-orange-300',
      cultivarsCount: '3 Cultivars',
      cultivarsList: ['Red Lady 786', 'Co-7 / Co-8', 'Washington'],
      optimalTemp: '22°C - 28°C',
      tssRange: '11.5 - 14.5° Brix',
      biomarker: 'Lycopene Accumulation & Pectin Softening'
    },
    {
      id: 'apple',
      name: 'Apple',
      tamilName: 'ஆப்பிள்',
      latinName: 'Malus domestica',
      icon: '🍎',
      color: 'rose',
      accentBorder: 'border-rose-500/30 hover:border-rose-500/60',
      badgeColor: 'bg-rose-500/20 text-rose-300',
      cultivarsCount: '3 Cultivars',
      cultivarsList: ['Royal Delicious', 'Kashmiri Ambri', 'Granny Smith'],
      optimalTemp: '16°C - 22°C',
      tssRange: '12.0 - 16.0° Brix',
      biomarker: 'Anthocyanin Synthesis & Firmness Decay'
    },
    {
      id: 'orange',
      name: 'Orange',
      tamilName: 'ஆரஞ்சு',
      latinName: 'Citrus sinensis',
      icon: '🍊',
      color: 'orange',
      accentBorder: 'border-orange-500/30 hover:border-orange-500/60',
      badgeColor: 'bg-orange-500/20 text-orange-300',
      cultivarsCount: '4 Cultivars',
      cultivarsList: ['Nagpur Orange', 'Sathgudi', 'Coorg Mandarin', 'Kinnow'],
      optimalTemp: '18°C - 24°C',
      tssRange: '9.0 - 13.5° Brix',
      biomarker: 'Carotenoid / Limonene & Acid De-greening'
    },
    {
      id: 'guava',
      name: 'Guava',
      tamilName: 'கொய்யா',
      latinName: 'Psidium guajava',
      icon: '🍐',
      color: 'lime',
      accentBorder: 'border-lime-500/30 hover:border-lime-500/60',
      badgeColor: 'bg-lime-500/20 text-lime-300',
      cultivarsCount: '4 Cultivars',
      cultivarsList: ['Allahabad Safeda', 'Lucknow 49', 'Taiwan Pink', 'Arka Kiran'],
      optimalTemp: '20°C - 26°C',
      tssRange: '8.5 - 12.0° Brix',
      biomarker: 'Ascorbic Acid (Vit C) & Pectin Hydrolysis'
    },
    {
      id: 'pomegranate',
      name: 'Pomegranate',
      tamilName: 'மாதுளை',
      latinName: 'Punica granatum',
      icon: '🔴',
      color: 'red',
      accentBorder: 'border-red-500/30 hover:border-red-500/60',
      badgeColor: 'bg-red-500/20 text-red-300',
      cultivarsCount: '4 Cultivars',
      cultivarsList: ['Bhagwa', 'Ganesh', 'Arakta', 'Ruby'],
      optimalTemp: '18°C - 25°C',
      tssRange: '14.0 - 17.5° Brix',
      biomarker: 'Ellagitannins & Aril Anthocyanin Blush'
    },
    {
      id: 'watermelon',
      name: 'Watermelon',
      tamilName: 'தர்பூசணி',
      latinName: 'Citrullus lanatus',
      icon: '🍉',
      color: 'emerald',
      accentBorder: 'border-emerald-500/30 hover:border-emerald-500/60',
      badgeColor: 'bg-emerald-500/20 text-emerald-300',
      cultivarsCount: '4 Cultivars',
      cultivarsList: ['Sugar Baby', 'Kiran', 'Black Diamond', 'Icebox'],
      optimalTemp: '24°C - 30°C',
      tssRange: '10.0 - 13.0° Brix',
      biomarker: 'Ground Spot Yellowing & Tendril Drying'
    }
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-12">
      
      {/* Model Status Notification */}
      <ModelStatusBanner modelStatus={modelStatus} />

      {/* Hero Section */}
      <section className="relative glass-card rounded-3xl p-8 sm:p-12 lg:p-16 border border-slate-800 text-center overflow-hidden">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[700px] bg-gradient-to-tr from-emerald-500/10 via-amber-500/10 to-purple-500/10 rounded-full blur-3xl pointer-events-none" />
        
        <div className="relative z-10 max-w-4xl mx-auto">
          
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-semibold uppercase tracking-wider mb-6">
            <Sparkles className="w-3.5 h-3.5 text-amber-400" />
            <span>AI Multi-Fruit Ripening, Plant Guard & Future Simulation</span>
          </div>

          <h1 className="text-4xl sm:text-6xl font-black text-white tracking-tight font-mono">
            Fruit<span className="text-emerald-400">Twin</span>
          </h1>
          
          <p className="text-xl sm:text-2xl font-bold text-amber-300 mt-2 font-sans">
            "See the fruit today. Predict its future."
          </p>

          <p className="text-slate-300 text-sm sm:text-base max-w-3xl mx-auto mt-4 leading-relaxed">
            Next-generation agricultural intelligence framework for multi-fruit post-harvest supply chains. 
            Coupling real-time computer vision classification, Plant & Non-Fruit Guard, Out-of-Dataset Fruit Detection, 
            Arrhenius biochemical kinetics forecasting, and Conditional Generative Adversarial Networks (cGAN).
          </p>

          {/* Supported Fruit Switcher Pills (8 Fruits) */}
          <div className="mt-8 flex flex-wrap items-center justify-center gap-2">
            <span className="text-xs text-slate-400 font-mono uppercase tracking-wider mr-1">
              Select Fruit Specimen:
            </span>
            {ALL_FRUITS.map((fruit) => (
              <button
                key={fruit.id}
                onClick={() => navigateToFruit ? navigateToFruit(fruit.id) : setActiveTab('analyze')}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-900/90 hover:bg-slate-800 border border-slate-700/80 hover:border-emerald-500/50 text-slate-200 hover:text-white transition-all text-xs font-mono font-bold shadow-sm group"
                title={`Launch ${fruit.name} scanner`}
              >
                <span className="text-base group-hover:scale-125 transition-transform">{fruit.icon}</span>
                <span>{fruit.name}</span>
                <span className="text-[10px] text-slate-400 font-sans font-normal">({fruit.tamilName})</span>
              </button>
            ))}
          </div>

          {/* Primary CTA Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mt-8">
            <button
              onClick={() => setActiveTab('analyze')}
              className="w-full sm:w-auto px-8 py-4 rounded-xl bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-bold shadow-lg shadow-emerald-500/25 hover:from-emerald-400 hover:to-emerald-500 transition-all flex items-center justify-center gap-2 group font-mono"
            >
              <Scan className="w-5 h-5 group-hover:scale-110 transition-transform" />
              <span>Launch Live Multi-Fruit Scanner</span>
              <ArrowRight className="w-4 h-4 ml-1" />
            </button>

            <button
              onClick={() => setActiveTab('simulation')}
              className="w-full sm:w-auto px-8 py-4 rounded-xl bg-purple-600/20 hover:bg-purple-600/30 text-purple-300 hover:text-white border border-purple-500/30 font-semibold transition-all flex items-center justify-center gap-2 font-mono"
            >
              <Sparkles className="w-4 h-4 text-purple-400" />
              <span>Future Appearance (cGAN)</span>
            </button>

            <button
              onClick={() => setActiveTab('dataset')}
              className="w-full sm:w-auto px-8 py-4 rounded-xl bg-slate-900 hover:bg-slate-800 text-slate-300 hover:text-white border border-slate-800 font-semibold transition-all flex items-center justify-center gap-2 font-mono text-xs"
            >
              <Activity className="w-4 h-4 text-amber-400" />
              <span>IoT & Dataset Telemetry</span>
            </button>
          </div>

        </div>
      </section>

      {/* Live Telemetry & System Metric Counters */}
      <section className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { label: 'SUPPORTED FRUITS', val: '8 Fruits', sub: 'Mango, Banana, Papaya, Apple, Orange, Guava, Pomegranate, Watermelon', icon: Zap, color: 'text-emerald-400' },
          { label: 'CULTIVAR DATABASE', val: '30+ Varieties', sub: 'Regional Indian & Tamil Cultivars', icon: Compass, color: 'text-amber-400' },
          { label: 'PLANT & FOLIAGE GUARD', val: 'Auto-Filter Active', sub: 'Rejects Leaves & Prompts for Fruit', icon: Leaf, color: 'text-teal-400' },
          { label: 'OUT-OF-DATASET AI', val: 'Exotic Fruit Identifier', sub: 'Detects Dragon Fruit, Kiwi, etc.', icon: Cpu, color: 'text-purple-400' }
        ].map((stat, i) => (
          <div key={i} className="glass-card rounded-2xl p-5 border border-slate-800 flex flex-col justify-between">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[11px] font-mono font-bold text-slate-400 tracking-wider uppercase">{stat.label}</span>
              <stat.icon className={`w-4 h-4 ${stat.color}`} />
            </div>
            <div>
              <p className="text-xl sm:text-2xl font-black font-mono text-white">{stat.val}</p>
              <p className="text-xs text-slate-400 font-sans mt-0.5">{stat.sub}</p>
            </div>
          </div>
        ))}
      </section>

      {/* Interactive Fruit Command Center (8 Fruit Cards) */}
      <section className="space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-2 border-b border-slate-800 pb-4">
          <div>
            <div className="inline-flex items-center gap-1.5 text-xs font-mono font-bold uppercase tracking-wider text-emerald-400 mb-1">
              <Layers className="w-3.5 h-3.5" />
              <span>AgriTech Command Cards</span>
            </div>
            <h2 className="text-2xl font-black text-white font-mono">
              8-Fruit Botanical & Biochemical Profiles
            </h2>
          </div>
          <p className="text-xs text-slate-400 font-sans max-w-md">
            Click any fruit to instantly launch dedicated deep classification, cultivar recognition, and shelf-life simulation.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {fruitProfiles.map((fp) => (
            <div 
              key={fp.id}
              className={`glass-card rounded-2xl p-6 border ${fp.accentBorder} flex flex-col justify-between transition-all hover:scale-[1.01] hover:shadow-xl`}
            >
              <div>
                {/* Header */}
                <div className="flex items-center justify-between mb-3">
                  <span className="text-4xl filter drop-shadow">{fp.icon}</span>
                  <span className={`text-[11px] font-mono font-bold px-2.5 py-1 rounded-full ${fp.badgeColor}`}>
                    {fp.cultivarsCount}
                  </span>
                </div>

                <div className="mb-3">
                  <h3 className="text-xl font-black text-white font-mono flex items-baseline gap-2">
                    {fp.name}
                    <span className="text-xs font-normal text-slate-400 font-sans">({fp.tamilName})</span>
                  </h3>
                  <p className="text-[11px] font-mono italic text-slate-400">{fp.latinName}</p>
                </div>

                {/* Cultivars Preview */}
                <div className="mb-4">
                  <span className="text-[10px] uppercase font-mono font-bold text-slate-400 tracking-wider block mb-1.5">
                    Recognized Cultivars
                  </span>
                  <div className="flex flex-wrap gap-1">
                    {fp.cultivarsList.slice(0, 4).map((cult, ci) => (
                      <span key={ci} className="text-[10px] font-mono bg-slate-900 border border-slate-800 text-slate-300 px-2 py-0.5 rounded-md">
                        {cult}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Biomarker Stats */}
                <div className="space-y-1.5 text-xs font-mono bg-slate-950/70 p-3 rounded-xl border border-slate-800/80 mb-5">
                  <div className="flex justify-between items-center text-[11px]">
                    <span className="text-slate-400">Sweetness (TSS):</span>
                    <span className="text-amber-300 font-bold">{fp.tssRange}</span>
                  </div>
                  <div className="flex justify-between items-center text-[11px]">
                    <span className="text-slate-400">Optimal Temp:</span>
                    <span className="text-emerald-400 font-bold">{fp.optimalTemp}</span>
                  </div>
                  <div className="pt-1 border-t border-slate-800/60 text-[10px] text-slate-400 font-sans line-clamp-1">
                    <span className="text-slate-300 font-mono font-bold">Biomarker: </span>{fp.biomarker}
                  </div>
                </div>
              </div>

              {/* Action Button */}
              <button
                onClick={() => navigateToFruit ? navigateToFruit(fp.id) : setActiveTab('analyze')}
                className="w-full py-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700 hover:border-emerald-500 text-white font-mono text-xs font-bold transition-all flex items-center justify-center gap-2 group"
              >
                <span>Launch {fp.name} Scanner</span>
                <ArrowRight className="w-3.5 h-3.5 text-emerald-400 group-hover:translate-x-1 transition-transform" />
              </button>
            </div>
          ))}
        </div>
      </section>

      {/* Interactive Quick-Scan Launcher (Pre-Loaded Specimens + Diagnostic Tests) */}
      <section className="glass-card rounded-3xl p-6 sm:p-8 border border-slate-800">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
          <div>
            <div className="inline-flex items-center gap-1.5 text-xs font-mono font-bold uppercase tracking-wider text-amber-400 mb-1">
              <Zap className="w-3.5 h-3.5" />
              <span>Instant Test Specimens & AI Diagnostic Benchmarks</span>
            </div>
            <h3 className="text-xl sm:text-2xl font-black text-white font-mono">
              Live Test Specimens: 1-Click Scan & Validation
            </h3>
            <p className="text-xs text-slate-400 font-sans mt-0.5">
              Click any specimen to load it directly into the AI scanner. Includes tests for Plant/Foliage Rejection and Unsupported Fruit Detection.
            </p>
          </div>
          <button
            onClick={() => setActiveTab('analyze')}
            className="self-start sm:self-auto px-4 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700 text-xs font-mono font-bold text-slate-200 hover:text-white transition-all flex items-center gap-2"
          >
            <span>Custom Photo Upload</span>
            <ArrowRight className="w-3.5 h-3.5 text-emerald-400" />
          </button>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
          {quickTestSpecimens.map((item, idx) => (
            <div 
              key={idx}
              className="bg-slate-950/90 rounded-2xl p-4 border border-slate-800/80 hover:border-slate-700 flex flex-col justify-between group transition-all"
            >
              <div>
                {/* Image Preview Container */}
                <div className="w-full h-36 rounded-xl bg-slate-900 border border-slate-800 overflow-hidden flex items-center justify-center p-2 mb-3 relative group-hover:border-emerald-500/40 transition-colors">
                  <img 
                    src={item.sample.img} 
                    alt={item.name} 
                    className="h-full object-contain filter drop-shadow group-hover:scale-105 transition-transform"
                  />
                  <div className="absolute top-2 right-2">
                    <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded-full border ${item.badgeColor}`}>
                      {item.badge}
                    </span>
                  </div>
                </div>

                <div className="flex items-center gap-2 mb-1">
                  <span className="text-lg">{item.icon}</span>
                  <h4 className="text-sm font-bold text-white font-mono">{item.name}</h4>
                </div>
                <p className="text-xs font-sans text-amber-300/80 mb-2">{item.tamil}</p>
                <p className="text-[11px] text-slate-400 font-sans leading-relaxed mb-4">
                  {item.desc}
                </p>
              </div>

              <button
                onClick={() => navigateToFruit ? navigateToFruit(item.fruitId, item.sample) : setActiveTab('analyze')}
                className="w-full py-2.5 rounded-xl bg-emerald-500/10 hover:bg-emerald-500 text-emerald-400 hover:text-white border border-emerald-500/30 font-mono text-xs font-bold transition-all flex items-center justify-center gap-2 shadow-sm"
              >
                <Scan className="w-3.5 h-3.5" />
                <span>Quick Scan Specimen</span>
              </button>
            </div>
          ))}
        </div>
      </section>

      {/* Visual Process Flow (Pipeline) */}
      <section className="mb-4">
        <div className="text-center mb-8">
          <span className="text-xs font-bold uppercase tracking-widest text-slate-400 font-mono bg-slate-900 px-3.5 py-1 rounded-full border border-slate-800">
            Multi-Fruit AI Pipeline Architecture
          </span>
          <h3 className="text-2xl font-black text-white font-mono mt-3">
            How FruitTwin Decouples Analysis, Botanical Verification & Prediction
          </h3>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { step: '01', title: 'SPECIMEN INGESTION & PLANT GUARD', desc: 'Ingest specimen; AI vision checks if image is a plant leaf, unsupported fruit, or one of the 8 supported fruits.', color: 'emerald' },
            { step: '02', title: 'VISION CHROMATOGRAPHY & MORPHOMETRY', desc: 'Deep vision neural net maps skin hue distribution, carotenoid/anthocyanin shift, and cross-fruit boundaries.', color: 'teal' },
            { step: '03', title: 'ARRHENIUS BIOCHEMICAL REGRESSION', desc: 'Chemical kinetics engine calculates shelf-life days, TSS sweetness decay, and optimal export consumption windows.', color: 'amber' },
            { step: '04', title: 'cGAN APPEARANCE SYNTHESIS', desc: 'Conditional Generative Adversarial Network visualizes photo-realistic future ripening appearance.', color: 'purple' },
          ].map((item, idx) => (
            <div key={idx} className="glass-card rounded-2xl p-5 border border-slate-800 relative group">
              <span className="text-xs font-mono font-bold text-slate-500 block mb-2">{item.step}</span>
              <h4 className="text-base font-bold text-white font-mono flex items-center gap-2 mb-2">
                {item.title}
              </h4>
              <p className="text-slate-400 text-xs leading-relaxed">{item.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Ripening Dynamics Reference Table */}
      <section className="glass-card rounded-2xl p-6 sm:p-8 border border-slate-800 font-mono">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
          <div>
            <h3 className="text-lg font-bold text-white flex items-center gap-2">
              <Activity className="w-5 h-5 text-emerald-400" />
              Ripening Dynamics & Commercial Maturity Standards across 8 Fruits
            </h3>
            <p className="text-xs text-slate-400 font-sans mt-0.5">
              Reference physical and biochemical criteria recognized by the FruitTwin model suite.
            </p>
          </div>
          <span className="text-xs text-slate-400 bg-slate-900 border border-slate-800 px-3 py-1 rounded-lg">
            ISO / Codex Alimentarius Standards
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="pb-3 font-bold uppercase">Maturity Stage</th>
                <th className="pb-3 font-bold uppercase">Skin Hue Indicator</th>
                <th className="pb-3 font-bold uppercase">Flesh Firmness</th>
                <th className="pb-3 font-bold uppercase">Typical TSS (°Brix)</th>
                <th className="pb-3 font-bold uppercase">Supply Chain Recommendation</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-300">
              <tr>
                <td className="py-3 font-bold text-emerald-400">Unripe (Stage 1)</td>
                <td className="py-3 font-sans">Dark green skin, high chlorophyll dominance</td>
                <td className="py-3 font-sans">Very firm (&gt;12 N/cm²)</td>
                <td className="py-3">6.0 - 10.0°</td>
                <td className="py-3 font-sans text-emerald-300">Safe for long-distance export and cold storage</td>
              </tr>
              <tr>
                <td className="py-3 font-bold text-amber-400">Semi-Ripe (Stage 2)</td>
                <td className="py-3 font-sans">Color-break stage; green-to-yellow/red gradient</td>
                <td className="py-3 font-sans">Yielding to pressure (6-10 N/cm²)</td>
                <td className="py-3">9.0 - 14.5°</td>
                <td className="py-3 font-sans text-amber-300">Ideal for retail distribution & transit to store</td>
              </tr>
              <tr>
                <td className="py-3 font-bold text-yellow-400">Ripe (Stage 3)</td>
                <td className="py-3 font-sans">Golden yellow, orange, or deep crimson flush</td>
                <td className="py-3 font-sans">Soft, aromatic flesh (3-5 N/cm²)</td>
                <td className="py-3">12.0 - 23.0°</td>
                <td className="py-3 font-sans text-yellow-300">Peak consumer sweetness; sell immediately</td>
              </tr>
              <tr>
                <td className="py-3 font-bold text-rose-400">Overripe (Stage 4)</td>
                <td className="py-3 font-sans">Senescence, dark sugar spotting, wrinkling</td>
                <td className="py-3 font-sans">Very soft (&lt;2 N/cm²)</td>
                <td className="py-3">14.0 - 24.0° (fermenting)</td>
                <td className="py-3 font-sans text-rose-300">Immediate discount or transfer to processing/pulping</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* How FruitTwin Thinks Section */}
      <HowItWorksSection />

    </div>
  );
}
