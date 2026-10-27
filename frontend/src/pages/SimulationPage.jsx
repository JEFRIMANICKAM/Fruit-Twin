import React, { useState, useRef } from 'react';
import { Sparkles, Upload, ArrowRight, AlertTriangle, Layers, Eye, RefreshCw, CheckCircle2, Zap } from 'lucide-react';
import { simulateGANStage } from '../services/api';
import { ALL_FRUITS } from '../services/allFruitVarieties';
import { MULTI_FRUIT_SAMPLES } from '../services/sampleImages';
import ModelStatusBanner from '../components/ModelStatusBanner';

export default function SimulationPage({ modelStatus }) {
  const [selectedFruit, setSelectedFruit] = useState(ALL_FRUITS[0]); // Default: Mango
  const [selectedVariety, setSelectedVariety] = useState(ALL_FRUITS[0].varieties[0]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [targetStage, setTargetStage] = useState('Ripe');
  const [temperature, setTemperature] = useState(28);

  const [isSimulating, setIsSimulating] = useState(false);
  const [simulationResult, setSimulationResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  const fileInputRef = useRef(null);

  const handleFruitChange = (fruit) => {
    setSelectedFruit(fruit);
    setSelectedVariety(fruit.varieties[0]);
    setSelectedFile(null);
    setImagePreview(null);
    setSimulationResult(null);
    setErrorMsg(null);
  };

  const handleLoadSample = (sample) => {
    setSelectedFile(null);
    setImagePreview(sample.img);
    setTemperature(sample.temp || 28);
    setSimulationResult(null);
    setErrorMsg(null);
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setImagePreview(URL.createObjectURL(file));
      setSimulationResult(null);
      setErrorMsg(null);
    }
  };

  const handleRunSimulation = async (stageVal = targetStage) => {
    if (!selectedFile && !imagePreview) {
      setErrorMsg(`Please upload or select a source ${selectedFruit.name.toLowerCase()} image first.`);
      return;
    }

    setIsSimulating(true);
    setErrorMsg(null);
    setTargetStage(stageVal);

    try {
      let fileToUpload = selectedFile;
      if (!fileToUpload && imagePreview) {
        const res = await fetch(imagePreview);
        const blob = await res.blob();
        fileToUpload = new File([blob], `${selectedFruit.id}_input.svg`, { type: 'image/svg+xml' });
      }

      const data = await simulateGANStage(fileToUpload, temperature, stageVal, selectedFruit.id);
      setSimulationResult(data);
    } catch (err) {
      setErrorMsg(err.message || 'GAN simulation failed.');
    } finally {
      setIsSimulating(false);
    }
  };

  const currentFruitSamples = MULTI_FRUIT_SAMPLES[selectedFruit.id] || [];

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      
      {/* Page Header */}
      <div className="text-center max-w-2xl mx-auto mb-4 font-mono">
        <span className="text-purple-400 text-xs font-bold uppercase tracking-wider bg-purple-500/10 px-3.5 py-1 rounded-full border border-purple-500/20">
          Generative AI Multi-Fruit Visualizer
        </span>
        <h1 className="text-3xl sm:text-4xl font-black text-white mt-2 font-mono">
          Future Ripening Simulation (cGAN)
        </h1>
        <p className="text-slate-400 text-xs font-sans mt-1">
          Synthesize photo-realistic future visual ripening appearances of Indian & Tamil fruit cultivars across target maturity stages.
        </p>
      </div>

      <ModelStatusBanner modelStatus={modelStatus} />

      {/* Fruit Switcher Tabs */}
      <div className="glass-card rounded-2xl p-3 border border-slate-800 flex flex-wrap items-center justify-center gap-2">
        <span className="text-xs font-mono font-bold text-slate-400 uppercase tracking-wider mr-2 hidden sm:inline">
          Target Fruit Specimen:
        </span>
        {ALL_FRUITS.map((fruit) => (
          <button
            key={fruit.id}
            onClick={() => handleFruitChange(fruit)}
            className={`px-4 py-2 rounded-xl text-xs font-mono font-bold transition-all flex items-center gap-2 border ${
              selectedFruit.id === fruit.id
                ? 'bg-purple-600 text-white border-purple-400 shadow-md shadow-purple-500/20'
                : 'bg-slate-900 text-slate-300 border-slate-800 hover:bg-slate-800 hover:text-white'
            }`}
          >
            <span className="text-base">{fruit.icon}</span>
            <span>{fruit.name}</span>
            <span className="text-[11px] opacity-80">({fruit.tamilName})</span>
          </button>
        ))}
      </div>

      {/* Control Panel */}
      <div className="glass-card rounded-2xl p-6 sm:p-8 border border-slate-800 shadow-xl">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-start font-mono">
          
          {/* Upload & Sample Selector Input */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <label className="block text-xs font-bold text-slate-300 uppercase tracking-wider">
                Source {selectedFruit.name} Specimen
              </label>
              <span className="text-[11px] text-purple-400 font-sans">
                {selectedVariety.name}
              </span>
            </div>

            <div 
              onClick={() => fileInputRef.current?.click()}
              className="border border-dashed border-slate-800 hover:border-purple-500/50 rounded-xl p-4 text-center cursor-pointer transition-colors bg-slate-950/80 min-h-[200px] flex flex-col items-center justify-center"
            >
              {imagePreview ? (
                <div className="flex flex-col items-center">
                  <img src={imagePreview} alt="Source Preview" className="h-36 object-contain rounded-lg border border-slate-800 mb-2" />
                  <span className="text-xs text-purple-400 font-bold">Click to replace photo</span>
                </div>
              ) : (
                <div className="flex flex-col items-center">
                  <Upload className="w-8 h-8 text-purple-400 mb-2" />
                  <p className="text-xs font-bold text-white">Upload {selectedFruit.name} Photo</p>
                  <p className="text-[11px] text-slate-400 mt-1 font-sans">JPG / PNG / SVG image</p>
                </div>
              )}
            </div>
            <input type="file" ref={fileInputRef} accept="image/*" className="hidden" onChange={handleFileChange} />

            {/* Quick Sample Selector for Fruit */}
            {currentFruitSamples.length > 0 && (
              <div>
                <span className="text-[10px] font-mono font-bold text-slate-400 uppercase tracking-wider block mb-2">
                  Or Load Pre-Loaded {selectedFruit.name} Test Specimen:
                </span>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                  {currentFruitSamples.map((sample) => (
                    <button
                      key={sample.id}
                      type="button"
                      onClick={() => handleLoadSample(sample)}
                      className="p-2 rounded-xl bg-slate-900 border border-slate-800 hover:border-purple-500/50 text-left transition-all group"
                    >
                      <div className="h-14 rounded-lg bg-slate-950 flex items-center justify-center overflow-hidden mb-1 p-1">
                        <img src={sample.img} alt={sample.label} className="h-full object-contain group-hover:scale-105 transition-transform" />
                      </div>
                      <p className="text-[10px] font-mono font-bold text-slate-200 truncate">{sample.label}</p>
                      <p className="text-[9px] text-purple-300 font-sans">{sample.stage}</p>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Simulation Stage Options */}
          <div className="space-y-4">
            
            {/* Variety Selector */}
            <div>
              <label className="block text-xs font-bold text-slate-300 uppercase tracking-wider mb-2">
                {selectedFruit.name} Cultivar (வகை)
              </label>
              <select
                value={selectedVariety.id}
                onChange={(e) => {
                  const found = selectedFruit.varieties.find(v => v.id === e.target.value);
                  if (found) setSelectedVariety(found);
                }}
                className="w-full bg-slate-950 border border-slate-800 text-slate-200 text-xs font-mono rounded-lg px-3 py-2.5 outline-none"
              >
                {selectedFruit.varieties.map(v => (
                  <option key={v.id} value={v.id}>
                    {v.name} ({v.tamilName}) - {v.origin}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-xs font-bold text-slate-300 uppercase tracking-wider mb-2">
                Target Ripening Stage
              </label>
              <div className="grid grid-cols-3 gap-2">
                {['Semi-Ripe', 'Ripe', 'Overripe'].map((stg) => (
                  <button
                    key={stg}
                    type="button"
                    onClick={() => setTargetStage(stg)}
                    className={`py-2.5 px-2 rounded-lg text-xs font-bold transition-all border ${
                      targetStage === stg
                        ? 'bg-purple-600 text-white border-purple-400 shadow-md shadow-purple-500/20'
                        : 'bg-slate-950 text-slate-300 border-slate-800 hover:bg-slate-900'
                    }`}
                  >
                    {stg}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">
                Storage Temperature (°C)
              </label>
              <input
                type="number"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value) || 28)}
                className="w-full bg-slate-950 border border-slate-800 text-white font-mono text-xs rounded-lg px-3 py-2 outline-none"
              />
            </div>

            {errorMsg && (
              <p className="text-xs text-rose-400 font-medium font-sans">{errorMsg}</p>
            )}

            <button
              type="button"
              disabled={isSimulating || (!selectedFile && !imagePreview)}
              onClick={() => handleRunSimulation(targetStage)}
              className={`w-full py-3.5 rounded-xl font-extrabold text-xs uppercase tracking-wider transition-all flex items-center justify-center gap-2 shadow-lg ${
                isSimulating || (!selectedFile && !imagePreview)
                  ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700'
                  : 'bg-purple-600 hover:bg-purple-500 text-white shadow-purple-600/20'
              }`}
            >
              {isSimulating ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Synthesizing cGAN {selectedFruit.name} Image...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>Synthesize {targetStage} Stage ({selectedFruit.name})</span>
                </>
              )}
            </button>
          </div>

        </div>
      </div>

      {/* Side-by-Side Visual Comparison Display */}
      {imagePreview && (
        <div className="glass-card rounded-2xl p-6 sm:p-8 border border-slate-800 font-mono">
          <h3 className="text-lg font-bold text-white mb-6 flex items-center gap-2">
            <Eye className="w-5 h-5 text-purple-400" />
            Visual Stage Comparison: {selectedFruit.name} ({selectedVariety.name} - {selectedVariety.tamilName})
          </h3>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-8 items-center">
            
            {/* Input Image */}
            <div className="bg-slate-950 p-5 rounded-xl border border-slate-800 flex flex-col items-center text-center">
              <span className="text-xs font-bold text-slate-400 uppercase mb-3">
                Source Specimen (Input)
              </span>
              <div className="w-full h-64 rounded-lg bg-slate-900 flex items-center justify-center overflow-hidden border border-slate-800 mb-3">
                <img src={imagePreview} alt="Input Fruit" className="h-full object-contain p-2" />
              </div>
              <span className="text-xs text-slate-300 font-bold">{selectedVariety.name} ({selectedVariety.tamilName})</span>
            </div>

            {/* Generated GAN Output */}
            <div className="bg-slate-950 p-5 rounded-xl border border-purple-500/30 flex flex-col items-center text-center relative">
              <span className="text-xs font-bold text-purple-400 uppercase mb-3 flex items-center gap-1.5">
                <Sparkles className="w-4 h-4 text-purple-400" />
                Target: {targetStage} Stage (Conditional GAN)
              </span>

              <div className="w-full h-64 rounded-lg bg-slate-900 flex flex-col items-center justify-center overflow-hidden border border-slate-800 p-2 mb-3">
                {simulationResult && simulationResult.generated_image ? (
                  <img src={simulationResult.generated_image} alt="GAN Generated" className="h-full object-contain rounded-lg" />
                ) : (
                  <div className="text-center px-4">
                    <Sparkles className="w-8 h-8 text-purple-400/60 mx-auto mb-2" />
                    <p className="text-xs text-slate-400 leading-relaxed font-sans">
                      Click "Synthesize {targetStage} Stage" to generate predicted future appearance.
                    </p>
                  </div>
                )}
              </div>

              <span className="text-xs text-slate-400">
                {simulationResult?.metadata?.method ? simulationResult.metadata.method : 'Conditional GAN Neural Synthesis Engine'}
              </span>
            </div>

          </div>
        </div>
      )}

    </div>
  );
}
