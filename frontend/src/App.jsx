import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import HomePage from './pages/HomePage';
import AnalysisPage from './pages/AnalysisPage';
import SimulationPage from './pages/SimulationPage';
import DatasetPage from './pages/DatasetPage';
import AboutPage from './pages/AboutPage';
import { fetchModelStatus } from './services/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('home');
  const [modelStatus, setModelStatus] = useState(null);
  const [selectedFruitId, setSelectedFruitId] = useState('mango');
  const [activeSample, setActiveSample] = useState(null);

  useEffect(() => {
    async function checkStatus() {
      const status = await fetchModelStatus();
      setModelStatus(status);
    }
    checkStatus();
  }, []);

  const navigateToFruit = (fruitId = 'mango', sample = null) => {
    if (fruitId) setSelectedFruitId(fruitId);
    if (sample) setActiveSample(sample);
    setActiveTab('analyze');
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100 font-sans selection:bg-emerald-500 selection:text-white">
      
      {/* Navigation Bar */}
      <Navbar 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        modelStatus={modelStatus} 
      />

      {/* Main Content Area */}
      <main className="flex-1">
        {activeTab === 'home' && (
          <HomePage 
            setActiveTab={setActiveTab} 
            navigateToFruit={navigateToFruit}
            modelStatus={modelStatus} 
          />
        )}
        {activeTab === 'analyze' && (
          <AnalysisPage 
            modelStatus={modelStatus} 
            initialFruitId={selectedFruitId}
            initialSample={activeSample}
            onClearSample={() => setActiveSample(null)}
          />
        )}
        {activeTab === 'simulation' && (
          <SimulationPage modelStatus={modelStatus} />
        )}
        {activeTab === 'dataset' && (
          <DatasetPage />
        )}
        {activeTab === 'about' && (
          <AboutPage />
        )}
      </main>

      {/* Footer */}
      <Footer setActiveTab={setActiveTab} />

    </div>
  );
}
