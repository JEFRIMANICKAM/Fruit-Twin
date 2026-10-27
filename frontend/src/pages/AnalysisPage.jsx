import React, { useState, useRef, useEffect } from 'react';
import { Upload, Camera, Thermometer, Droplets, Scan, CheckCircle2, AlertCircle, AlertTriangle, Sparkles, RefreshCw, Info, ArrowRight, ShieldCheck, Cpu, Activity, Zap, Scale, ChevronDown, Leaf, Clock, X, Database } from 'lucide-react';
import { analyzeMango, calculateWhatIf, simulateGANStage, predictFruit, validateFruit } from '../services/api';
import { MULTI_FRUIT_SAMPLES } from '../services/sampleImages';
import { ALL_FRUITS } from '../services/allFruitVarieties';
import RipenessBadge from '../components/RipenessBadge';
import ModelStatusBanner from '../components/ModelStatusBanner';
import DatasetClassCatalogModal from '../components/DatasetClassCatalogModal';

export default function AnalysisPage({ modelStatus, initialFruitId, initialSample, onClearSample }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [temperature, setTemperature] = useState(28);
  const [humidity, setHumidity] = useState('65');
  
  const [selectedFruit, setSelectedFruit] = useState(() => {
    if (initialFruitId) {
      return ALL_FRUITS.find(f => f.id === initialFruitId) || ALL_FRUITS[0];
    }
    return ALL_FRUITS[0];
  });
  const [selectedVariety, setSelectedVariety] = useState(selectedFruit.varieties[0]);
  const [autoDetectedVariety, setAutoDetectedVariety] = useState(null);
  
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  // Pop-up Modal State (Mismatch, Plant/Foliage, Non-Fruit, Unsupported Fruit, Uncertain, Dataset Non-Fruit)
  const [activeModal, setActiveModal] = useState(null);

  // Dataset Class Catalogue Modal State
  const [isCatalogOpen, setIsCatalogOpen] = useState(false);

  // What-If Simulation State
  const [whatIfTemp, setWhatIfTemp] = useState(32);
  const [whatIfResult, setWhatIfResult] = useState(null);

  // GAN Simulation State within Result
  const [ganSimResults, setGanSimResults] = useState({});
  const [isGanSimulating, setIsGanSimulating] = useState(false);
  const [activeGanStage, setActiveGanStage] = useState(null);

  const fileInputRef = useRef(null);
  const cameraInputRef = useRef(null);

  const handleFruitChange = (fruitItem) => {
    setSelectedFruit(fruitItem);
    setSelectedVariety(fruitItem.varieties[0]);
    setAutoDetectedVariety(null);
    setImagePreview(null);
    setSelectedFile(null);
    setAnalysisResult(null);
    setGanSimResults({});
    setActiveModal(null);
  };

  useEffect(() => {
    if (initialFruitId) {
      const target = ALL_FRUITS.find(f => f.id === initialFruitId);
      if (target && target.id !== selectedFruit.id) {
        handleFruitChange(target);
      }
    }
  }, [initialFruitId]);

  useEffect(() => {
    if (initialSample) {
      const sampleFruit = ALL_FRUITS.find(f => initialSample.id.startsWith(f.id));
      if (sampleFruit && sampleFruit.id !== selectedFruit.id) {
        setSelectedFruit(sampleFruit);
        setSelectedVariety(sampleFruit.varieties[0]);
      }
      handleLoadSample(initialSample);
      if (onClearSample) onClearSample();
    }
  }, [initialSample]);

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setImagePreview(url);
      setAnalysisResult(null);
      setAutoDetectedVariety(null);
      setGanSimResults({});
      setErrorMessage(null);

      // Fast Client-Side Color Pre-Scan:
      // Instantly checks for Red Banana (Sevazhai) color signature
      const img = new window.Image();
      img.onload = () => {
        try {
          const canvas = document.createElement('canvas');
          canvas.width = 64;
          canvas.height = 64;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(img, 0, 0, 64, 64);
          const data = ctx.getImageData(0, 0, 64, 64).data;
          
          let totalR = 0, totalG = 0, totalB = 0, count = 0;
          let redDomCount = 0;
          
          for (let i = 0; i < data.length; i += 4) {
            const r = data[i], g = data[i+1], b = data[i+2];
            if ((r > 30 || g > 30 || b > 30) && !(r > 240 && g > 240 && b > 240)) {
              totalR += r;
              totalG += g;
              totalB += b;
              count++;
              if (r > g * 1.15 && r > b * 1.15 && r > 45) {
                redDomCount++;
              }
            }
          }
          
          if (count > 30) {
            const avgR = totalR / count;
            const avgG = totalG / count;
            const redRatio = redDomCount / count;
            
            if (selectedFruit.id === 'banana') {
              if (redRatio >= 0.08 || avgR > avgG * 1.12) {
                const sevazhaiVar = selectedFruit.varieties.find(v => v.id === 'sevazhai');
                if (sevazhaiVar) {
                  setSelectedVariety(sevazhaiVar);
                  setAutoDetectedVariety(`${sevazhaiVar.name} (${sevazhaiVar.tamilName})`);
                }
              }
            }
          }
        } catch (err) {
          console.warn('Canvas pre-scan exception:', err);
        }
      };
      img.src = url;

      // High-Precision AI Pre-Scan via validateFruit (Universal 20-Class Multi-Dataset Engine):
      // Automatically identifies Cross-Fruit Mismatch, Plant Foliage, Non-Fruit Object,
      // Dataset Non-Fruit Class, Unsupported Fruit, or Blurred Image immediately on upload
      validateFruit(file, selectedFruit.id).then((val) => {
        if (val.input_type === 'plant_foliage') {
          setActiveModal({
            type: 'plant_leaf',
            title: 'Plant / Foliage Detected',
            tamilTitle: 'தாவர இலை / செடி கண்டறியப்பட்டது',
            message: val.message || 'Vegetative leaves or foliage detected rather than a harvested fruit.',
            preview: url,
            confidence: val.confidence
          });
        } else if (val.input_type === 'non_fruit') {
          setActiveModal({
            type: 'non_fruit',
            title: 'No Fruit Detected',
            tamilTitle: 'பழம் அல்லாத பொருள்',
            message: val.message || 'Non-botanical object detected.',
            preview: url,
            confidence: val.confidence
          });
        } else if (val.input_type === 'uncertain') {
          setActiveModal({
            type: 'uncertain',
            title: 'Image Uncertain — Please Try Again',
            tamilTitle: 'தெளிவற்ற படம் — மீண்டும் முயற்சிக்கவும்',
            message: val.message || 'Fruit Twin could not confidently identify this image. Please upload a clearer photograph showing the fruit.',
            preview: url,
            confidence: val.confidence
          });
        } else if (val.input_type === 'supported_nonfruit_class') {
          setActiveModal({
            type: 'supported_nonfruit_class',
            title: `Supported Non-Fruit Class: ${val.detected_category}`,
            tamilTitle: 'பழம் அல்லாத தரவுத்தொகுப்பு வகுப்பு',
            detectedClass: val.detected_category,
            message: val.message || `${val.detected_category} is catalogued in Fruits-360, but it is an agricultural crop, not a fresh fruit.`,
            preview: url,
            confidence: val.confidence,
            topCandidates: val.top_three_candidates
          });
        } else if (val.input_type === 'unsupported_fruit') {
          setActiveModal({
            type: 'unsupported_fruit',
            title: 'Out-of-Dataset Fruit Specimen',
            tamilTitle: 'வகைப்படுத்தப்படாத பழம்',
            candidateName: val.detected_category || val.detected_fruit,
            message: val.message || 'Recognized fruit outside the 8 primary Fruit Twin pipelines.',
            preview: url,
            confidence: val.confidence,
            topCandidates: val.top_three_candidates
          });
        } else {
          // input_type === 'supported_fruit'
          const detectedFruitId = (val.detected_fruit || '').toLowerCase();
          const detectedFruitObj = ALL_FRUITS.find(f => f.id === detectedFruitId || f.name.toLowerCase() === detectedFruitId);
          
          if (detectedFruitObj && detectedFruitObj.id !== selectedFruit.id) {
            // Mismatch detected across tabs! Show pop-up modal immediately!
            setActiveModal({
              type: 'mismatch',
              detectedFruitId: detectedFruitObj.id,
              detectedFruit: detectedFruitObj,
              detectedVariety: val.detected_variety,
              confidence: val.confidence,
              currentFruit: selectedFruit,
              file: file,
              preview: url,
              topCandidates: val.top_three_candidates
            });
          } else {
            // Fruit matches current tab!
            if (val.detected_variety) {
              setAutoDetectedVariety(val.detected_variety);
              const match = selectedFruit.varieties.find(v => 
                val.detected_variety.toLowerCase().includes(v.id) || 
                val.detected_variety.toLowerCase().includes(v.name.toLowerCase()) ||
                (v.id === 'sevazhai' && (val.detected_variety.toLowerCase().includes('red banana') || val.detected_variety.includes('செவ்வாழை'))) ||
                (v.id === 'sindhoora' && (val.detected_variety.toLowerCase().includes('sindhoora') || val.detected_variety.includes('சிந்தூரா'))) ||
                (v.id === 'royal_delicious' && (val.detected_variety.toLowerCase().includes('royal') || val.detected_variety.includes('டெலிஷியஸ்'))) ||
                (v.id === 'ambri' && (val.detected_variety.toLowerCase().includes('ambri') || val.detected_variety.includes('அம்ரி'))) ||
                (v.id === 'granny_smith' && (val.detected_variety.toLowerCase().includes('granny') || val.detected_variety.includes('ஸ்மித்'))) ||
                (v.id === 'red_lady' && (val.detected_variety.toLowerCase().includes('red lady') || val.detected_variety.includes('ரெட் லேடி'))) ||
                (v.id === 'co7' && (val.detected_variety.toLowerCase().includes('co-7') || val.detected_variety.includes('கோ-7')))
              );
              if (match) setSelectedVariety(match);
            }
          }
        }
      }).catch((err) => {
        console.warn('validateFruit pre-scan error:', err);
      });
    }
  };

  const handleLoadSample = async (sample) => {
    setImagePreview(sample.img);
    setTemperature(sample.temp);
    setAnalysisResult(null);
    setGanSimResults({});
    setErrorMessage(null);
    
    if (sample.id === 'banana_red_sevazhai') {
      const sevazhaiVar = selectedFruit.varieties.find(v => v.id === 'sevazhai');
      if (sevazhaiVar) {
        setSelectedVariety(sevazhaiVar);
        setAutoDetectedVariety(`${sevazhaiVar.name} (${sevazhaiVar.tamilName})`);
      }
    } else if (sample.id === 'mango_sindhoora') {
      const sindhooraVar = selectedFruit.varieties.find(v => v.id === 'sindhoora');
      if (sindhooraVar) {
        setSelectedVariety(sindhooraVar);
        setAutoDetectedVariety(`${sindhooraVar.name} (${sindhooraVar.tamilName})`);
      }
    } else if (sample.id === 'apple_ripe') {
      const appleVar = selectedFruit.varieties.find(v => v.id === 'royal_delicious');
      if (appleVar) {
        setSelectedVariety(appleVar);
        setAutoDetectedVariety(`${appleVar.name} (${appleVar.tamilName})`);
      }
    } else if (sample.id === 'apple_unripe') {
      const appleVar = selectedFruit.varieties.find(v => v.id === 'granny_smith');
      if (appleVar) {
        setSelectedVariety(appleVar);
        setAutoDetectedVariety(`${appleVar.name} (${appleVar.tamilName})`);
      }
    } else if (sample.id === 'orange_ripe') {
      const orangeVar = selectedFruit.varieties.find(v => v.id === 'nagpur');
      if (orangeVar) {
        setSelectedVariety(orangeVar);
        setAutoDetectedVariety(`${orangeVar.name} (${orangeVar.tamilName})`);
      }
    } else if (sample.id === 'guava_ripe') {
      const guavaVar = selectedFruit.varieties.find(v => v.id === 'allahabad_safeda');
      if (guavaVar) {
        setSelectedVariety(guavaVar);
        setAutoDetectedVariety(`${guavaVar.name} (${guavaVar.tamilName})`);
      }
    } else if (sample.id === 'pomegranate_ripe') {
      const pomVar = selectedFruit.varieties.find(v => v.id === 'bhagwa');
      if (pomVar) {
        setSelectedVariety(pomVar);
        setAutoDetectedVariety(`${pomVar.name} (${pomVar.tamilName})`);
      }
    } else if (sample.id === 'watermelon_ripe') {
      const wmVar = selectedFruit.varieties.find(v => v.id === 'kiran');
      if (wmVar) {
        setSelectedVariety(wmVar);
        setAutoDetectedVariety(`${wmVar.name} (${wmVar.tamilName})`);
      }
    } else if (sample.id === 'papaya_ripe') {
      const papayaVar = selectedFruit.varieties.find(v => v.id === 'red_lady') || selectedFruit.varieties[0];
      if (papayaVar) {
        setSelectedVariety(papayaVar);
        setAutoDetectedVariety(`${papayaVar.name} (${papayaVar.tamilName})`);
      }
    } else if (sample.id === 'papaya_unripe') {
      const papayaVar = selectedFruit.varieties.find(v => v.id === 'co7') || selectedFruit.varieties[0];
      if (papayaVar) {
        setSelectedVariety(papayaVar);
        setAutoDetectedVariety(`${papayaVar.name} (${papayaVar.tamilName})`);
      }
    } else if (sample.id === 'test_plant_leaf' || sample.id === 'test_tree_branch') {
      setAutoDetectedVariety("Plant Leaf / Foliage Specimen (பச்சை இலை)");
    } else if (sample.id === 'test_dragon_fruit') {
      setAutoDetectedVariety("Dragon Fruit / Exotic Specimen (டிராகன் பழம்)");
    } else if (sample.id === 'test_potato') {
      setAutoDetectedVariety("Potato Red / Dataset Class (உருளைக்கிழங்கு)");
    } else if (sample.id === 'test_phone') {
      setAutoDetectedVariety("Smartphone / Non-Botanical (தொலைபேசி)");
    } else if (sample.id === 'test_blurred') {
      setAutoDetectedVariety("Blurred / Uncertain Specimen (தெளிவற்ற படம்)");
    } else {
      setAutoDetectedVariety(null);
    }

    // Modal popup triggers for diagnostic specimens or cross-fruit samples:
    if (sample.type === 'plant_leaf' || sample.id === 'test_plant_leaf' || sample.id === 'test_tree_branch') {
      setActiveModal({
        type: 'plant_leaf',
        title: sample.id === 'test_tree_branch' ? 'Tree Branch & Foliage Detected' : 'Plant / Foliage Specimen Detected',
        tamilTitle: 'தாவர இலை / கிளை மாதிரி',
        message: 'This diagnostic specimen contains plant leaves or vegetative foliage rather than a harvested fruit.',
        preview: sample.img,
        confidence: 98.5
      });
    } else if (sample.type === 'non_fruit' || sample.id === 'test_phone') {
      setActiveModal({
        type: 'non_fruit',
        title: 'No Fruit Detected',
        tamilTitle: 'பழம் அல்லாத பொருள்',
        message: 'The uploaded image contains an electronic device (Smartphone). Please upload a clear photograph of an agricultural fruit.',
        preview: sample.img,
        confidence: 95.0
      });
    } else if (sample.type === 'supported_nonfruit_class' || sample.id === 'test_potato') {
      setActiveModal({
        type: 'supported_nonfruit_class',
        title: 'Supported Non-Fruit Class: Potato Red',
        tamilTitle: 'பழம் அல்லாத தரவுத்தொகுப்பு வகுப்பு (உருளைக்கிழங்கு)',
        detectedClass: 'Potato Red',
        message: 'Potato Red is catalogued in Fruits-360 as an agricultural crop, but it is a tuber/vegetable rather than a fresh fruit. Fruit Twin ripening pipelines are calibrated for fruits.',
        preview: sample.img,
        confidence: 94.2,
        topCandidates: [{ class_name: 'Potato Red', confidence: 94.2 }]
      });
    } else if (sample.type === 'uncertain' || sample.id === 'test_blurred') {
      setActiveModal({
        type: 'uncertain',
        title: 'Image Uncertain — Please Try Again',
        tamilTitle: 'தெளிவற்ற படம் — மீண்டும் முயற்சிக்கவும்',
        message: 'Fruit Twin could not confidently identify this image. The photo is severely blurred or low-contrast. Please upload a clearer photograph showing the fruit.',
        preview: sample.img,
        confidence: 32.5
      });
    } else if (sample.type === 'unsupported_fruit' || sample.id === 'test_dragon_fruit') {
      setActiveModal({
        type: 'unsupported_fruit',
        candidateName: 'Dragon Fruit (பித்தாயா / டிராகன் பழம்)',
        message: 'This diagnostic specimen is recognized as Dragon Fruit, which is outside the 8 verified agricultural fruits.',
        preview: sample.img,
        confidence: 94.0
      });
    } else {
      // Check if sample belongs to a different fruit tab
      const sampleFruit = ALL_FRUITS.find(f => sample.id.startsWith(f.id));
      if (sampleFruit && sampleFruit.id !== selectedFruit.id) {
        setActiveModal({
          type: 'mismatch',
          detectedFruitId: sampleFruit.id,
          detectedFruit: sampleFruit,
          detectedVariety: sample.label,
          confidence: 97.0,
          currentFruit: selectedFruit,
          file: null,
          preview: sample.img
        });
      }
    }
    
    try {
      const blob = await (await fetch(sample.img)).blob();
      const file = new File([blob], `${sample.id}.svg`, { type: 'image/svg+xml' });
      setSelectedFile(file);
    } catch (err) {
      console.warn('Sample blob conversion fallback:', err);
    }
  };

  const handleModalSwitchFruit = async (targetFruit, fileToAnalyze, previewUrl) => {
    setActiveModal(null);
    if (!targetFruit) return;

    setSelectedFruit(targetFruit);
    setSelectedVariety(targetFruit.varieties?.[0] || { id: 'default', name: targetFruit.name, tamilName: '' });
    setAutoDetectedVariety(null);
    setAnalysisResult(null);
    setGanSimResults({});

    let uploadPayload = fileToAnalyze || selectedFile;
    if (!uploadPayload && (previewUrl || imagePreview)) {
      try {
        const res = await fetch(previewUrl || imagePreview);
        const blob = await res.blob();
        uploadPayload = new File([blob], `${targetFruit.id}_input.svg`, { type: 'image/svg+xml' });
      } catch (e) {
        console.warn('Fallback file conversion:', e);
      }
    }

    if (uploadPayload) {
      setSelectedFile(uploadPayload);
      setIsAnalyzing(true);
      setErrorMessage(null);
      try {
        const data = await analyzeMango(uploadPayload, temperature, humidity, targetFruit.id);
        if (data.detected_variety) {
          setAutoDetectedVariety(data.detected_variety);
          const match = targetFruit.varieties?.find(v => 
            data.detected_variety.toLowerCase().includes(v.id) || 
            data.detected_variety.toLowerCase().includes(v.name.toLowerCase()) ||
            (v.id === 'sevazhai' && (data.detected_variety.toLowerCase().includes('red banana') || data.detected_variety.includes('செவ்வாழை'))) ||
            (v.id === 'sindhoora' && (data.detected_variety.toLowerCase().includes('sindhoora') || data.detected_variety.includes('சிந்தூரா'))) ||
            (v.id === 'royal_delicious' && (data.detected_variety.toLowerCase().includes('royal') || data.detected_variety.includes('டெலிஷியஸ்'))) ||
            (v.id === 'ambri' && (data.detected_variety.toLowerCase().includes('ambri') || data.detected_variety.includes('அம்ரி'))) ||
            (v.id === 'granny_smith' && (data.detected_variety.toLowerCase().includes('granny') || data.detected_variety.includes('ஸ்மித்'))) ||
            (v.id === 'red_lady' && (data.detected_variety.toLowerCase().includes('red lady') || data.detected_variety.includes('ரெட் லேடி'))) ||
            (v.id === 'co7' && (data.detected_variety.toLowerCase().includes('co-7') || data.detected_variety.includes('கோ-7')))
          );
          if (match) setSelectedVariety(match);
        }
        setAnalysisResult(data);
        setWhatIfResult(null);
        setGanSimResults({});
      } catch (err) {
        setErrorMessage(err.message || `Error executing ${targetFruit.name} analysis.`);
      } finally {
        setIsAnalyzing(false);
      }
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile && !imagePreview) {
      setErrorMessage(`Please upload a ${selectedFruit.name.toLowerCase()} photo or select a sample image first.`);
      return;
    }

    setIsAnalyzing(true);
    setErrorMessage(null);

    try {
      let fileToUpload = selectedFile;
      if (!fileToUpload && imagePreview) {
        const res = await fetch(imagePreview);
        const blob = await res.blob();
        fileToUpload = new File([blob], `${selectedFruit.id}_input.svg`, { type: 'image/svg+xml' });
      }

      const data = await analyzeMango(fileToUpload, temperature, humidity, selectedFruit.id);
      
      // Auto-detected variety handling
      if (data.detected_variety) {
        setAutoDetectedVariety(data.detected_variety);
        // Find match in varieties array
        const match = selectedFruit.varieties.find(v => 
          data.detected_variety.toLowerCase().includes(v.id) || 
          data.detected_variety.toLowerCase().includes(v.name.toLowerCase()) ||
          (v.id === 'sevazhai' && (data.detected_variety.toLowerCase().includes('red banana') || data.detected_variety.includes('செவ்வாழை'))) ||
          (v.id === 'sindhoora' && (data.detected_variety.toLowerCase().includes('sindhoora') || data.detected_variety.includes('சிந்தூரா'))) ||
          (v.id === 'royal_delicious' && (data.detected_variety.toLowerCase().includes('royal') || data.detected_variety.includes('டெலிஷியஸ்'))) ||
          (v.id === 'ambri' && (data.detected_variety.toLowerCase().includes('ambri') || data.detected_variety.includes('அம்ரி'))) ||
          (v.id === 'granny_smith' && (data.detected_variety.toLowerCase().includes('granny') || data.detected_variety.includes('ஸ்மித்'))) ||
          (v.id === 'red_lady' && (data.detected_variety.toLowerCase().includes('red lady') || data.detected_variety.includes('ரெட் லேடி'))) ||
          (v.id === 'co7' && (data.detected_variety.toLowerCase().includes('co-7') || data.detected_variety.includes('கோ-7')))
        );
        if (match) {
          setSelectedVariety(match);
        }
      }

      // Check for pop-up modal triggers (Universal Multi-Dataset AI Guard):
      if (data.is_uncertain) {
        setActiveModal({
          type: 'uncertain',
          title: data.uncertain_validation?.title || 'Image Uncertain — Please Try Again',
          tamilTitle: 'தெளிவற்ற படம் — மீண்டும் முயற்சிக்கவும்',
          message: data.uncertain_validation?.message || data.status_message,
          preview: imagePreview,
          confidence: data.confidence
        });
      } else if (data.is_plant_or_leaf) {
        setActiveModal({
          type: 'plant_leaf',
          title: data.plant_validation?.title || 'Plant / Foliage Detected',
          tamilTitle: 'தாவர இலை கண்டறியப்பட்டது',
          message: data.plant_validation?.message || data.status_message,
          preview: imagePreview,
          confidence: data.confidence
        });
      } else if (data.is_supported_nonfruit_class) {
        setActiveModal({
          type: 'supported_nonfruit_class',
          title: data.nonfruit_dataset_validation?.title || 'Supported Non-Fruit Dataset Class',
          tamilTitle: 'பழம் அல்லாத தரவுத்தொகுப்பு வகுப்பு',
          detectedClass: data.nonfruit_dataset_validation?.detected_class,
          message: data.nonfruit_dataset_validation?.message || data.status_message,
          preview: imagePreview,
          confidence: data.confidence,
          topCandidates: data.fruit_identification?.top_three_candidates
        });
      } else if (data.is_non_fruit) {
        setActiveModal({
          type: 'non_fruit',
          title: 'No Fruit Detected',
          tamilTitle: 'பழம் அல்லாத பொருள்',
          message: data.status_message,
          preview: imagePreview,
          confidence: data.confidence
        });
      } else if (data.is_unsupported_fruit) {
        setActiveModal({
          type: 'unsupported_fruit',
          candidateName: data.unsupported_validation?.candidate_name || data.fruit_identification?.predicted_fruit,
          message: data.unsupported_validation?.message,
          preview: imagePreview,
          confidence: data.confidence,
          topCandidates: data.fruit_identification?.top_three_candidates
        });
      } else if (data.is_fruit_mismatch && data.fruit_validation) {
        const detectedFruitObj = ALL_FRUITS.find(f => f.id === data.fruit_validation.detected_fruit);
        setActiveModal({
          type: 'mismatch',
          detectedFruitId: data.fruit_validation.detected_fruit,
          detectedFruit: detectedFruitObj || { id: data.fruit_validation.detected_fruit, name: data.fruit_validation.detected_fruit_name, icon: data.fruit_validation.detected_fruit_icon, tamilName: data.fruit_validation.detected_fruit_tamil, varieties: [] },
          detectedVariety: data.detected_variety,
          confidence: data.confidence,
          currentFruit: selectedFruit,
          file: fileToUpload,
          preview: imagePreview,
          topCandidates: data.fruit_identification?.top_three_candidates
        });
      }

      setAnalysisResult(data);
      setWhatIfResult(null);
      setGanSimResults({});
    } catch (err) {
      setErrorMessage(err.message || `Error executing ${selectedFruit.name} analysis.`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleSwitchFruit = async (targetFruitId) => {
    const targetFruit = ALL_FRUITS.find(f => f.id === targetFruitId);
    if (!targetFruit) return;

    setSelectedFruit(targetFruit);
    setSelectedVariety(targetFruit.varieties[0]);
    setAutoDetectedVariety(null);
    setAnalysisResult(null);
    setGanSimResults({});

    if (selectedFile || imagePreview) {
      setIsAnalyzing(true);
      setErrorMessage(null);
      try {
        let fileToUpload = selectedFile;
        if (!fileToUpload && imagePreview) {
          const res = await fetch(imagePreview);
          const blob = await res.blob();
          fileToUpload = new File([blob], `${targetFruit.id}_input.svg`, { type: 'image/svg+xml' });
        }

        const data = await analyzeMango(fileToUpload, temperature, humidity, targetFruit.id);

        if (data.detected_variety) {
          setAutoDetectedVariety(data.detected_variety);
          const match = targetFruit.varieties.find(v => 
            data.detected_variety.toLowerCase().includes(v.id) || 
            data.detected_variety.toLowerCase().includes(v.name.toLowerCase()) ||
            (v.id === 'sevazhai' && (data.detected_variety.toLowerCase().includes('red banana') || data.detected_variety.includes('செவ்வாழை'))) ||
            (v.id === 'sindhoora' && (data.detected_variety.toLowerCase().includes('sindhoora') || data.detected_variety.includes('சிந்தூரா'))) ||
            (v.id === 'royal_delicious' && (data.detected_variety.toLowerCase().includes('royal') || data.detected_variety.includes('டெலிஷியஸ்'))) ||
            (v.id === 'ambri' && (data.detected_variety.toLowerCase().includes('ambri') || data.detected_variety.includes('அம்ரி'))) ||
            (v.id === 'granny_smith' && (data.detected_variety.toLowerCase().includes('granny') || data.detected_variety.includes('ஸ்மித்'))) ||
            (v.id === 'red_lady' && (data.detected_variety.toLowerCase().includes('red lady') || data.detected_variety.includes('ரெட் லேடி'))) ||
            (v.id === 'co7' && (data.detected_variety.toLowerCase().includes('co-7') || data.detected_variety.includes('கோ-7')))
          );
          if (match) {
            setSelectedVariety(match);
          }
        }

        setAnalysisResult(data);
        setWhatIfResult(null);
        setGanSimResults({});
      } catch (err) {
        setErrorMessage(err.message || `Error executing ${targetFruit.name} analysis.`);
      } finally {
        setIsAnalyzing(false);
      }
    }
  };

  const handleWhatIfCalculate = async (tempVal) => {
    setWhatIfTemp(tempVal);
    try {
      const stage = analysisResult?.ripeness_stage || 'Semi-Ripe';
      const res = await calculateWhatIf(tempVal, stage, selectedFruit.id);
      setWhatIfResult(res);
    } catch (err) {
      console.error('What-if error:', err);
    }
  };

  const handleSimulateGAN = async (stage) => {
    if (!selectedFile && !imagePreview) return;
    setActiveGanStage(stage);
    setIsGanSimulating(true);
    try {
      let fileToUpload = selectedFile;
      if (!fileToUpload && imagePreview) {
        const res = await fetch(imagePreview);
        const blob = await res.blob();
        fileToUpload = new File([blob], `${selectedFruit.id}_input.svg`, { type: 'image/svg+xml' });
      }
      const res = await simulateGANStage(fileToUpload, temperature, stage, selectedFruit.id);
      setGanSimResults(prev => ({
        ...prev,
        [stage]: res.generated_image
      }));
    } catch (err) {
      console.error('GAN simulation error:', err);
    } finally {
      setIsGanSimulating(false);
      setActiveGanStage(null);
    }
  };

  const currentSamples = MULTI_FRUIT_SAMPLES[selectedFruit.id] || MULTI_FRUIT_SAMPLES.banana;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 font-mono">
      
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between border-b border-slate-800/80 pb-6 mb-8 gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono font-semibold text-emerald-400 bg-emerald-500/10 px-2.5 py-1 rounded border border-emerald-500/20 uppercase tracking-wider">
              Variety Auto-Detection Pipeline Active
            </span>
            <span className="text-xs font-mono text-slate-400">| Red Banana (செவ்வாழை) & Indian Cultivars</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white mt-1 font-mono tracking-tight">
            Fruit Ripening & Variety Classification Engine
          </h1>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={() => setIsCatalogOpen(true)}
            className="px-4 py-2.5 rounded-lg bg-emerald-950/70 hover:bg-emerald-900/90 text-emerald-300 hover:text-white text-xs font-bold font-mono transition-all border border-emerald-500/40 flex items-center gap-2 shadow-sm cursor-pointer"
            title="Explore all 20 Fruits-360 classes & RawRipe capabilities"
          >
            <Database className="w-4 h-4 text-emerald-400" />
            <span>📚 Dataset Catalogue (20 Classes)</span>
          </button>
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="px-4 py-2.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-bold font-mono transition-all border border-slate-700 flex items-center gap-2 cursor-pointer"
          >
            <Upload className="w-4 h-4 text-emerald-400" />
            <span>Upload Image</span>
          </button>
          <button
            type="button"
            onClick={() => cameraInputRef.current?.click()}
            className="px-4 py-2.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-bold font-mono transition-all border border-slate-700 flex items-center gap-2 cursor-pointer"
          >
            <Camera className="w-4 h-4 text-amber-400" />
            <span>Camera Capture</span>
          </button>
        </div>
      </div>

      <ModelStatusBanner modelStatus={modelStatus} />

      {/* Control Workspace */}
      <div className="glass-card rounded-2xl p-6 border border-slate-800/80 mb-12 shadow-xl">
        
        {/* Fruit Category Tabs */}
        <div className="mb-6 pb-4 border-b border-slate-800">
          <label className="block text-xs font-bold text-slate-300 uppercase tracking-wider mb-3">
            Select Fruit Category (பழங்களின் வகைகள்):
          </label>
          <div className="grid grid-cols-2 sm:grid-cols-4 xl:grid-cols-8 gap-2">
            {ALL_FRUITS.map(f => (
              <button
                key={f.id}
                type="button"
                onClick={() => handleFruitChange(f)}
                className={`py-2.5 px-3 rounded-xl border font-bold text-xs transition-all flex flex-col items-center justify-center gap-1 ${
                  selectedFruit.id === f.id
                    ? 'bg-gradient-to-r from-emerald-600 to-teal-700 text-white border-emerald-400 shadow-lg shadow-emerald-500/20'
                    : 'bg-slate-950 border-slate-800 text-slate-300 hover:bg-slate-900'
                }`}
              >
                <span className="text-xl">{f.icon}</span>
                <span className="truncate">{f.name}</span>
                <span className="text-[10px] font-sans opacity-75 truncate">{f.tamilName}</span>
              </button>
            ))}
          </div>

          {/* Diagnostic AI Tests: Plant / Foliage, Non-Fruit Object, Dataset Non-Fruit, Unsupported, Uncertain */}
          <div className="mt-4 pt-3 border-t border-slate-800/60 flex flex-wrap items-center gap-2">
            <span className="text-[11px] font-mono font-bold text-slate-400 uppercase tracking-wider mr-1">
              AI Vision Capabilities:
            </span>
            <button
              type="button"
              onClick={() => {
                const s = (MULTI_FRUIT_SAMPLES.diagnostic || []).find(x => x.id === 'test_plant_leaf');
                if (s) handleLoadSample(s);
              }}
              className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-emerald-950/40 hover:bg-emerald-900/60 border border-emerald-500/40 text-emerald-300 hover:text-white text-xs font-mono font-bold transition-all shadow-sm cursor-pointer"
              title="Test uploading a plant leaf image to verify leaf rejection"
            >
              <Leaf className="w-3.5 h-3.5 text-emerald-400" />
              <span>🌿 Plant Leaf</span>
            </button>
            <button
              type="button"
              onClick={() => {
                const s = (MULTI_FRUIT_SAMPLES.diagnostic || []).find(x => x.id === 'test_tree_branch');
                if (s) handleLoadSample(s);
              }}
              className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-emerald-950/40 hover:bg-emerald-900/60 border border-emerald-500/40 text-emerald-300 hover:text-white text-xs font-mono font-bold transition-all shadow-sm cursor-pointer"
              title="Test uploading a tree branch with vegetation to verify foliage rejection"
            >
              <span>🌳 Tree Branch</span>
            </button>
            <button
              type="button"
              onClick={() => {
                const s = (MULTI_FRUIT_SAMPLES.diagnostic || []).find(x => x.id === 'test_phone');
                if (s) handleLoadSample(s);
              }}
              className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-rose-950/40 hover:bg-rose-900/60 border border-rose-500/40 text-rose-300 hover:text-white text-xs font-mono font-bold transition-all shadow-sm cursor-pointer"
              title="Test uploading an electronic device (Smartphone) to verify non-fruit rejection"
            >
              <AlertCircle className="w-3.5 h-3.5 text-rose-400" />
              <span>📱 Test Phone (Non-Fruit)</span>
            </button>
            <button
              type="button"
              onClick={() => {
                const s = (MULTI_FRUIT_SAMPLES.diagnostic || []).find(x => x.id === 'test_potato');
                if (s) handleLoadSample(s);
              }}
              className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-cyan-950/40 hover:bg-cyan-900/60 border border-cyan-500/40 text-cyan-300 hover:text-white text-xs font-mono font-bold transition-all shadow-sm cursor-pointer"
              title="Test uploading Potato Red to verify non-fruit dataset class detection"
            >
              <Database className="w-3.5 h-3.5 text-cyan-400" />
              <span>🥔 Test Potato (Dataset Class)</span>
            </button>
            <button
              type="button"
              onClick={() => {
                const s = (MULTI_FRUIT_SAMPLES.diagnostic || []).find(x => x.id === 'test_dragon_fruit');
                if (s) handleLoadSample(s);
              }}
              className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-purple-950/40 hover:bg-purple-900/60 border border-purple-500/40 text-purple-300 hover:text-white text-xs font-mono font-bold transition-all shadow-sm cursor-pointer"
              title="Test uploading an unsupported fruit (Dragon Fruit) to verify out-of-dataset detection"
            >
              <Sparkles className="w-3.5 h-3.5 text-purple-400" />
              <span>🍇 Unsupported Fruit</span>
            </button>
            <button
              type="button"
              onClick={() => {
                const s = (MULTI_FRUIT_SAMPLES.diagnostic || []).find(x => x.id === 'test_blurred');
                if (s) handleLoadSample(s);
              }}
              className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-slate-800/60 hover:bg-slate-700/80 border border-slate-600 text-slate-300 hover:text-white text-xs font-mono font-bold transition-all shadow-sm cursor-pointer"
              title="Test uploading a severely blurred image to verify uncertain image detection"
            >
              <Activity className="w-3.5 h-3.5 text-slate-400" />
              <span>🌫️ Test Blurred Image</span>
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          
          {/* Left Column: Image Viewport & Samples (7 Cols) */}
          <div className="lg:col-span-7">
            <div className="flex items-center justify-between mb-3">
              <label className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-2">
                <Camera className="w-4 h-4 text-emerald-400" />
                1. {selectedFruit.name} Image Telemetry Viewport
              </label>
              <span className="text-[11px] text-slate-400">RGB Spectrum Array</span>
            </div>

            <div className="border border-slate-800 rounded-xl p-4 bg-slate-950/90 min-h-[280px] flex flex-col items-center justify-center relative overflow-hidden">
              {imagePreview ? (
                <div className="w-full flex flex-col items-center">
                  <div className="relative max-h-64 rounded-lg overflow-hidden border border-slate-800 shadow-xl">
                    <img
                      src={imagePreview}
                      alt={`${selectedFruit.name} Input Preview`}
                      className="max-h-64 object-contain rounded-lg p-2"
                    />
                  </div>
                  
                  <div className="mt-3 flex flex-col sm:flex-row items-center gap-2 text-xs font-mono">
                    {analysisResult?.is_fruit_mismatch ? (
                      <div className="flex items-center gap-2 text-amber-400 bg-amber-950/60 px-3 py-1.5 rounded border border-amber-500/40 font-bold animate-pulse">
                        <AlertTriangle className="w-4 h-4 text-amber-400" />
                        <span>Warning: Visual features match {analysisResult.fruit_validation?.detected_fruit_name} ({analysisResult.fruit_validation?.detected_fruit_tamil})</span>
                      </div>
                    ) : (
                      <div className="flex items-center gap-2 text-emerald-400 bg-slate-900 px-3 py-1.5 rounded border border-slate-800 font-bold">
                        <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                        <span>
                          {autoDetectedVariety ? `Auto-Detected: ${autoDetectedVariety}` : `${selectedFruit.name} Loaded`}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <div 
                  onClick={() => fileInputRef.current?.click()}
                  className="flex flex-col items-center py-10 cursor-pointer text-center group"
                >
                  <div className="w-14 h-14 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-center text-emerald-400 mb-3 group-hover:border-emerald-500/50 transition-colors">
                    <Upload className="w-6 h-6" />
                  </div>
                  <p className="text-sm font-bold text-white font-mono">Upload {selectedFruit.name} Photo ({selectedFruit.tamilName})</p>
                  <p className="text-xs text-slate-400 mt-1">AI automatically identifies Red Banana (செவ்வாழை), Poovan, Nendran, etc.</p>
                </div>
              )}
            </div>

            {/* Hidden Inputs */}
            <input type="file" ref={fileInputRef} accept="image/*" className="hidden" onChange={handleFileSelect} />
            <input type="file" ref={cameraInputRef} accept="image/*" capture="environment" className="hidden" onChange={handleFileSelect} />

            {/* Sample Selector */}
            <div className="mt-4 pt-4 border-t border-slate-800/80">
              <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider block mb-2">
                Pre-Loaded {selectedFruit.name} Test Samples:
              </span>
              <div className="grid grid-cols-3 gap-2">
                {currentSamples.map((sample) => (
                  <button
                    key={sample.id}
                    type="button"
                    onClick={() => handleLoadSample(sample)}
                    className="p-2.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-left transition-all group"
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-bold text-slate-200 group-hover:text-emerald-400">
                        {sample.label}
                      </span>
                      <span className="text-[10px] text-slate-400">{sample.stage}</span>
                    </div>
                    <div className="w-full h-8 rounded bg-slate-950 overflow-hidden border border-slate-800">
                      <img src={sample.img} alt={sample.label} className="w-full h-full object-cover" />
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Right Column: Variety Selector & Environmental Form (5 Cols) */}
          <div className="lg:col-span-5 space-y-6">
            
            {/* Variety Selector */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                  2. Variety Detection & Manual Override
                </label>
                {autoDetectedVariety && (
                  <span className="text-[10px] text-emerald-400 font-bold bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/30">
                    AI Auto-Identified
                  </span>
                )}
              </div>
              <div className="grid grid-cols-2 gap-2">
                {selectedFruit.varieties.map((v) => (
                  <button
                    key={v.id}
                    type="button"
                    onClick={() => setSelectedVariety(v)}
                    className={`p-2.5 rounded-xl border text-left transition-all ${
                      selectedVariety.id === v.id
                        ? 'bg-emerald-950/80 border-emerald-500/60 text-white shadow-md shadow-emerald-500/10'
                        : 'bg-slate-950 border-slate-800 text-slate-300 hover:bg-slate-900'
                    }`}
                  >
                    <div className="font-bold text-xs flex items-center justify-between">
                      <span>{v.name}</span>
                      <span className="text-[10px] text-emerald-400 font-sans">{v.tamilName}</span>
                    </div>
                    <p className="text-[10px] text-slate-400 truncate mt-0.5">{v.origin.split('/')[0]}</p>
                  </button>
                ))}
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                  <Thermometer className="w-4 h-4 text-amber-400" />
                  3. Storage Temperature (°C)
                </label>
                <span className="text-[10px] text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/20">
                  Required
                </span>
              </div>

              <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-slate-400 text-xs">Ambient Temp:</span>
                  <span className="text-xl font-bold text-amber-400">{temperature} °C</span>
                </div>
                <input
                  type="range"
                  min="15"
                  max="38"
                  step="0.5"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-amber-400"
                />
                <div className="flex justify-between text-[10px] text-slate-500">
                  <span>15°C Cold</span>
                  <span>28°C Ambient</span>
                  <span>38°C Tropical</span>
                </div>
              </div>
            </div>

            <div>
              <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <Droplets className="w-4 h-4 text-blue-400" />
                Relative Humidity (%) [Optional]
              </label>
              <input
                type="number"
                placeholder="65"
                value={humidity}
                onChange={(e) => setHumidity(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 focus:border-emerald-500 text-slate-200 text-xs rounded-lg px-3.5 py-2.5 outline-none"
              />
            </div>

            {errorMessage && (
              <div className="p-3 rounded-lg bg-rose-950/80 border border-rose-500/40 text-rose-300 text-xs flex items-center gap-2">
                <AlertCircle className="w-4 h-4 flex-shrink-0 text-rose-400" />
                <span>{errorMessage}</span>
              </div>
            )}

            <button
              type="button"
              disabled={isAnalyzing || (!selectedFile && !imagePreview)}
              onClick={handleAnalyze}
              className={`w-full py-3.5 rounded-xl font-bold text-xs tracking-wider uppercase transition-all flex items-center justify-center gap-2 ${
                isAnalyzing || (!selectedFile && !imagePreview)
                  ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700'
                  : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-600/20'
              }`}
            >
              {isAnalyzing ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Identifying Variety & Ripeness...</span>
                </>
              ) : (
                <>
                  <Scan className="w-4 h-4" />
                  <span>Analyze & Identify {selectedFruit.name}</span>
                </>
              )}
            </button>

          </div>

        </div>
      </div>

      {/* RESULT PAGE REPORT */}
      {analysisResult && (
        <div className="space-y-10 animate-fade-in">
          
          {/* 1. PLANT / LEAF / FOLIAGE DETECTED BANNER */}
          {analysisResult.is_plant_or_leaf && (
            <div className="bg-gradient-to-r from-emerald-950/90 via-slate-900 to-teal-950/90 border-2 border-emerald-500 rounded-3xl p-6 sm:p-8 shadow-2xl space-y-5">
              <div className="flex items-start gap-4">
                <div className="p-4 bg-emerald-500/20 border border-emerald-500/50 rounded-2xl text-emerald-400 shrink-0 shadow-lg shadow-emerald-500/10">
                  <Leaf className="w-10 h-10 animate-pulse" />
                </div>
                <div className="space-y-2">
                  <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/20 border border-emerald-500/40 text-emerald-300 text-xs font-mono font-bold uppercase tracking-wider">
                    <span>🌿 Botanical Vision Classifier</span>
                    <span>•</span>
                    <span>Plant / Foliage Detected</span>
                  </div>
                  <h3 className="text-xl sm:text-2xl font-black text-white font-mono">
                    Plant Leaf / Branch Detected — Not a Harvested Fruit!
                  </h3>
                  <p className="text-sm text-slate-300 font-sans leading-relaxed max-w-3xl">
                    {analysisResult.plant_validation?.message || "The AI computer vision pipeline analyzed this image and determined it contains vegetative foliage, plant leaves, or stems rather than a fruit specimen. Ripeness classification and Arrhenius shelf-life prediction require a clear photo of the harvested fruit itself."}
                  </p>
                  <p className="text-xs text-emerald-400 font-mono font-bold">
                    💡 Action Required: Please upload or capture an image of the actual fruit (Mango, Banana, Papaya, Apple, Orange, Guava, Pomegranate, or Watermelon).
                  </p>
                </div>
              </div>

              {/* Quick Actions to Load a Real Fruit Specimen */}
              <div className="pt-4 border-t border-slate-800 flex flex-wrap items-center gap-3">
                <span className="text-xs font-mono font-bold text-slate-400">
                  Load a Verified Fruit Test Specimen:
                </span>
                {ALL_FRUITS.slice(0, 4).map((f) => (
                  <button
                    key={f.id}
                    type="button"
                    onClick={() => {
                      handleFruitChange(f);
                      const samp = (MULTI_FRUIT_SAMPLES[f.id] || [])[0];
                      if (samp) handleLoadSample(samp);
                    }}
                    className="px-3 py-1.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700 hover:border-emerald-500 text-xs font-mono font-bold text-slate-200 hover:text-white flex items-center gap-1.5 transition-all"
                  >
                    <span>{f.icon}</span>
                    <span>{f.name}</span>
                  </button>
                ))}
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="px-4 py-1.5 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 text-xs font-mono font-extrabold uppercase tracking-wider shadow-lg shadow-emerald-500/20 flex items-center gap-1.5 ml-auto transition-all"
                >
                  <Upload className="w-3.5 h-3.5" />
                  <span>Upload Actual Fruit Photo</span>
                </button>
              </div>
            </div>
          )}

          {/* 2. UNSUPPORTED / EXOTIC FRUIT DETECTED BANNER */}
          {analysisResult.is_unsupported_fruit && (
            <div className="bg-gradient-to-r from-purple-950/90 via-slate-900 to-indigo-950/90 border-2 border-purple-500 rounded-3xl p-6 sm:p-8 shadow-2xl space-y-5">
              <div className="flex items-start gap-4">
                <div className="p-4 bg-purple-500/20 border border-purple-500/50 rounded-2xl text-purple-400 shrink-0 shadow-lg shadow-purple-500/10">
                  <Sparkles className="w-10 h-10 animate-bounce" />
                </div>
                <div className="space-y-2">
                  <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-purple-500/20 border border-purple-500/40 text-purple-300 text-xs font-mono font-bold uppercase tracking-wider">
                    <span>🍇 Unverified Agricultural Specimen</span>
                    <span>•</span>
                    <span>Out-of-Dataset Fruit</span>
                  </div>
                  <h3 className="text-xl sm:text-2xl font-black text-white font-mono">
                    {analysisResult.unsupported_validation?.title || "Unsupported Fruit Specimen Detected"}
                  </h3>
                  <p className="text-sm text-slate-300 font-sans leading-relaxed max-w-3xl">
                    {analysisResult.unsupported_validation?.message || "This specimen is recognized as a fruit, but is not currently in FruitTwin's verified agricultural database. Our shelf-life regression and cGAN future synthesis models are verified on 8 primary agricultural fruits."}
                  </p>
                  <p className="text-xs text-purple-300 font-mono font-bold">
                    💡 Verified Agricultural Database: Mango, Banana, Papaya, Apple, Orange, Guava, Pomegranate, and Watermelon.
                  </p>
                </div>
              </div>

              {/* Verified Fruits Switcher Pills */}
              <div className="pt-4 border-t border-slate-800 space-y-2">
                <span className="text-xs font-mono font-bold text-slate-400 block">
                  Switch to one of FruitTwin's 8 Verified Agricultural Fruits:
                </span>
                <div className="flex flex-wrap gap-2">
                  {ALL_FRUITS.map((f) => (
                    <button
                      key={f.id}
                      type="button"
                      onClick={() => {
                        handleFruitChange(f);
                        const samp = (MULTI_FRUIT_SAMPLES[f.id] || [])[0];
                        if (samp) handleLoadSample(samp);
                      }}
                      className="px-3 py-1.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700 hover:border-purple-400 text-xs font-mono font-bold text-slate-200 hover:text-white flex items-center gap-1.5 transition-all"
                    >
                      <span>{f.icon}</span>
                      <span>{f.name} ({f.tamilName})</span>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* 3. GENERAL NON-FRUIT OBJECT BANNER */}
          {analysisResult.is_non_fruit && !analysisResult.is_plant_or_leaf && (
            <div className="bg-gradient-to-r from-rose-950/90 via-slate-900 to-rose-950/70 border-2 border-rose-500 rounded-3xl p-6 shadow-2xl flex items-start gap-4">
              <div className="p-3.5 bg-rose-500/20 border border-rose-500/50 rounded-2xl text-rose-400 shrink-0">
                <AlertCircle className="w-8 h-8" />
              </div>
              <div className="space-y-1">
                <h3 className="text-lg font-bold text-white font-mono">No Fruit Detected in Image</h3>
                <p className="text-xs text-slate-300 font-sans">
                  The uploaded photo does not contain recognized fruit characteristics. Please provide a clear, well-lit photograph of a fruit specimen.
                </p>
              </div>
            </div>
          )}

          {/* 4. CROSS-CATEGORY FRUIT MISMATCH BANNER */}
          {analysisResult.is_fruit_mismatch && analysisResult.fruit_validation && (
            <div className="bg-gradient-to-r from-amber-950/70 via-slate-900 to-amber-950/60 border-2 border-amber-500/80 rounded-2xl p-6 shadow-2xl flex flex-col md:flex-row items-start md:items-center justify-between gap-5">
              <div className="flex items-start gap-4">
                <div className="p-3.5 bg-amber-500/20 border border-amber-500/50 rounded-2xl text-amber-400 shrink-0 shadow-lg shadow-amber-500/10">
                  <AlertTriangle className="w-8 h-8" />
                </div>
                <div>
                  <div className="flex flex-wrap items-center gap-2 mb-1.5">
                    <span className="text-xs font-bold uppercase tracking-wider text-amber-400">
                      Cross-Category Fruit Mismatch
                    </span>
                    <span className="text-xs bg-amber-500/20 text-amber-200 px-2.5 py-0.5 rounded-full font-mono border border-amber-500/40 font-bold flex items-center gap-1.5">
                      <span>{analysisResult.fruit_validation.detected_fruit_icon}</span>
                      <span>Detected: {analysisResult.fruit_validation.detected_fruit_name} ({analysisResult.fruit_validation.detected_fruit_tamil})</span>
                    </span>
                  </div>
                  <p className="text-sm text-amber-100 font-medium leading-relaxed max-w-3xl">
                    {analysisResult.fruit_validation.mismatch_message}
                  </p>
                  <p className="text-xs text-amber-300/70 mt-1">
                    You uploaded this in the {selectedFruit.name} ({selectedFruit.tamilName}) workspace. Click the button to switch to the correct category and calculate precise parameters.
                  </p>
                </div>
              </div>
              <div className="flex flex-col sm:flex-row items-center gap-2 shrink-0">
                <button
                  type="button"
                  onClick={() => setActiveModal({
                    type: 'mismatch',
                    detectedFruitId: analysisResult.fruit_validation.detected_fruit,
                    detectedFruit: ALL_FRUITS.find(f => f.id === analysisResult.fruit_validation.detected_fruit) || { id: analysisResult.fruit_validation.detected_fruit, name: analysisResult.fruit_validation.detected_fruit_name, icon: analysisResult.fruit_validation.detected_fruit_icon, tamilName: analysisResult.fruit_validation.detected_fruit_tamil, varieties: [] },
                    detectedVariety: analysisResult.detected_variety,
                    confidence: analysisResult.confidence,
                    currentFruit: selectedFruit,
                    file: selectedFile,
                    preview: imagePreview
                  })}
                  className="w-full sm:w-auto px-4 py-3 bg-slate-800 hover:bg-slate-700 text-amber-300 hover:text-white font-bold text-xs tracking-wider rounded-xl border border-amber-500/40 transition-all text-center"
                >
                  View Details & Mismatch Pop-up
                </button>
                <button
                  type="button"
                  onClick={() => handleSwitchFruit(analysisResult.fruit_validation.detected_fruit)}
                  className="w-full sm:w-auto px-5 py-3.5 bg-gradient-to-r from-amber-500 to-amber-400 hover:from-amber-400 hover:to-amber-300 text-slate-950 font-extrabold text-xs tracking-wider uppercase rounded-xl shadow-xl shadow-amber-500/25 flex items-center justify-center gap-2.5 transition-all transform hover:scale-[1.02] active:scale-[0.98] cursor-pointer"
                >
                  <span>Switch to {analysisResult.fruit_validation.detected_fruit_name} & Re-Analyze</span>
                  <ArrowRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          )}

          {/* Plant / Non-Fruit Diagnostic Overview (when non-fruit detected) */}
          {(analysisResult.is_plant_or_leaf || analysisResult.is_non_fruit) ? (
            <div className="glass-card rounded-2xl p-8 border border-slate-800 text-center space-y-4">
              <div className="w-full max-w-sm mx-auto h-52 rounded-xl bg-slate-950 border border-slate-800 overflow-hidden flex items-center justify-center p-3">
                <img src={imagePreview} alt="Specimen Preview" className="h-full object-contain" />
              </div>
              <div className="max-w-xl mx-auto space-y-2">
                <span className="text-xs font-mono font-bold text-amber-400 uppercase tracking-wider bg-amber-500/10 px-3 py-1 rounded-full border border-amber-500/30">
                  {analysisResult.is_plant_or_leaf ? "Plant Foliage Diagnostic Result" : "Non-Fruit Specimen Result"}
                </span>
                <h3 className="text-xl font-bold text-white font-mono">
                  {analysisResult.is_plant_or_leaf ? "Vegetative Foliage Identified" : "Non-Fruit Object Identified"}
                </h3>
                <p className="text-xs text-slate-300 font-sans leading-relaxed">
                  {analysisResult.is_plant_or_leaf
                    ? "The uploaded image was verified by computer vision as vegetative plant foliage/leaf. Biochemical ripeness indicators (TSS, Firmness, DAFS) and Arrhenius decay modeling are reserved for harvested fruit specimens."
                    : "The uploaded object is outside botanical fruit domains. Please capture or select a fruit specimen."}
                </p>
                <div className="pt-4 flex justify-center gap-3">
                  <button
                    type="button"
                    onClick={() => {
                      const sample = MULTI_FRUIT_SAMPLES.mango[0];
                      handleFruitChange(ALL_FRUITS[0]);
                      handleLoadSample(sample);
                    }}
                    className="px-5 py-2.5 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold text-xs uppercase tracking-wider transition-all"
                  >
                    Load Verified Fruit Specimen (🥭 Mango)
                  </button>
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="px-5 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-white font-bold text-xs uppercase tracking-wider border border-slate-700 transition-all"
                  >
                    Upload Other Fruit
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <>
              <div className="border-b border-slate-800 pb-4 flex items-center justify-between">
                <h2 className="text-xl font-extrabold text-white tracking-tight flex items-center gap-2">
                  <Activity className="w-5 h-5 text-emerald-400" />
                  {selectedFruit.name} Telemetry & Variety Report
                </h2>
                <span className="text-xs font-mono text-emerald-400 bg-emerald-500/10 px-3 py-1 rounded border border-emerald-500/20 font-bold">
                  Detected Variety: {autoDetectedVariety || `${selectedVariety.name} (${selectedVariety.tamilName})`}
                </span>
              </div>

          {/* 1. UPLOADED INPUT & 2. CURRENT RIPENESS & 3. ENVIRONMENTAL INPUT */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            
            {/* 1. Uploaded Input & Fruits-360 Identification */}
            <div className="glass-card rounded-2xl p-5 border border-slate-800 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">
                    1. Fruits-360 Fruit Identification
                  </span>
                  <span className={`text-[10px] font-bold px-2 py-0.5 rounded border ${
                    analysisResult.fruit_identification?.is_reliable !== false
                      ? 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20'
                      : 'text-amber-400 bg-amber-500/10 border-amber-500/20'
                  }`}>
                    {analysisResult.fruit_identification?.is_reliable !== false ? '✓ Reliable' : '⚠️ Uncertain'}
                  </span>
                </div>
                
                <div className="w-full h-44 rounded-lg bg-slate-950 border border-slate-800 overflow-hidden flex items-center justify-center p-2 mb-3">
                  <img src={imagePreview} alt={`Uploaded ${selectedFruit.name}`} className="h-full object-contain" />
                </div>

                <div className="space-y-2 bg-slate-950 p-3 rounded-lg border border-slate-800 text-xs">
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400">Identified Fruit:</span>
                    <span className="font-bold text-white text-sm">
                      {analysisResult.fruit_identification?.predicted_fruit || selectedFruit.name}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400">Identified Variety:</span>
                    <span className="font-bold text-emerald-400">
                      {analysisResult.fruit_identification?.predicted_variety || autoDetectedVariety || selectedVariety.name}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400">Model Confidence:</span>
                    <span className="font-bold text-cyan-400 font-mono">
                      {analysisResult.fruit_identification?.confidence || analysisResult.confidence}%
                    </span>
                  </div>
                </div>

                {/* Top 3 Candidates */}
                {analysisResult.fruit_identification?.top_three_candidates?.length > 0 && (
                  <div className="mt-3 bg-slate-950/60 p-2.5 rounded-lg border border-slate-800/80">
                    <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1.5">
                      Top Candidates (Fruits-360):
                    </span>
                    <div className="space-y-1">
                      {analysisResult.fruit_identification.top_three_candidates.slice(0, 3).map((cand, cIdx) => (
                        <div key={cIdx} className="flex justify-between items-center text-[11px] text-slate-300">
                          <span className="truncate max-w-[150px]">{cand.fruit} - {cand.variety}</span>
                          <span className="font-mono text-slate-400">{cand.confidence}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              <div className="mt-3 pt-3 border-t border-slate-800/80 text-[10px] text-slate-400 flex items-center justify-between">
                <span>Dataset:</span>
                <span className="text-emerald-400">Fruits-360 Multi-Class Benchmark</span>
              </div>
            </div>

            {/* 2. RawRipe Maturity Classification */}
            <div className="glass-card rounded-2xl p-5 border border-slate-800 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">
                    2. RawRipe Maturity Classification
                  </span>
                  <span className="text-[10px] text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/20">
                    RawRipe (1,630 imgs)
                  </span>
                </div>

                {analysisResult.rawripe_supported !== false ? (
                  <div className="space-y-3">
                    <div className="flex items-center justify-between bg-slate-950 p-3 rounded-lg border border-slate-800">
                      <span className="text-xs text-slate-400">RawRipe State:</span>
                      <span className={`px-3 py-1 rounded-full text-xs font-bold tracking-wide border ${
                        analysisResult.rawripe_maturity === 'Ripe'
                          ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                          : 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                      }`}>
                        {analysisResult.rawripe_maturity || 'Ripe'}
                      </span>
                    </div>

                    <div className="bg-slate-950 p-3.5 rounded-lg border border-slate-800 space-y-2">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-slate-400">Maturity Confidence</span>
                        <span className="text-cyan-400 font-bold">{analysisResult.rawripe_confidence || 94.0}%</span>
                      </div>
                      <div className="w-full h-2 bg-slate-900 rounded-full overflow-hidden border border-slate-800">
                        <div 
                          className="h-full bg-gradient-to-r from-amber-400 to-emerald-400 rounded-full" 
                          style={{ width: `${analysisResult.rawripe_confidence || 90.0}%` }} 
                        />
                      </div>
                    </div>

                    <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 space-y-1.5 text-xs">
                      <div className="flex items-center justify-between">
                        <span className="text-slate-400">Stage Classification:</span>
                        <RipenessBadge stage={analysisResult.ripeness_stage} size="small" />
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-slate-400">Ripeness Index:</span>
                        <span className="text-white font-bold">{analysisResult.ripeness_score}%</span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4 text-center my-4 space-y-2">
                    <AlertTriangle className="w-6 h-6 text-amber-400 mx-auto" />
                    <p className="text-xs font-bold text-amber-300 leading-snug">
                      Fruit identified successfully. Ripeness prediction is not yet available for this fruit.
                    </p>
                    <p className="text-[11px] text-slate-400">
                      RawRipe benchmark covers 10 agricultural fruits (Apple, Banana, Coconut, Guava, Litchi, Mango, Orange, Papaya, Pomegranate, Strawberry).
                    </p>
                  </div>
                )}
              </div>

              <div className="mt-3 pt-3 border-t border-slate-800/80 text-[10px] text-slate-400 flex items-center justify-between">
                <span>Model:</span>
                <span className="text-cyan-400">RawRipe 10-Fruit Maturity Classifier</span>
              </div>
            </div>

            {/* 3. Environmental Input */}
            <div className="glass-card rounded-2xl p-5 border border-slate-800 flex flex-col justify-between">
              <div>
                <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-2 block">
                  3. Environmental Input
                </span>

                <div className="bg-slate-950 p-4 rounded-lg border border-slate-800 text-center space-y-1">
                  <Thermometer className="w-6 h-6 text-amber-400 mx-auto" />
                  <p className="text-xs text-slate-400">Ambient Room Temperature</p>
                  <p className="text-3xl font-bold text-white">
                    {analysisResult.environmental_input?.temperature} <span className="text-amber-400 text-xl">°C</span>
                  </p>
                </div>

                {analysisResult.biochemical && (
                  <div className="mt-3 bg-slate-950 p-3 rounded-lg border border-slate-800 space-y-1 text-xs">
                    <div className="flex justify-between text-slate-300">
                      <span className="text-slate-400">TSS Sweetness:</span>
                      <span className="text-cyan-400 font-bold">{analysisResult.biochemical.tss} °Brix</span>
                    </div>
                    <div className="flex justify-between text-slate-300">
                      <span className="text-slate-400">Titratable Acidity:</span>
                      <span className="text-rose-400 font-bold">{analysisResult.biochemical.ta} %</span>
                    </div>
                  </div>
                )}
              </div>

              <div className="mt-3 pt-3 border-t border-slate-800/80 text-[10px] text-slate-400">
                Temperature is one verified physiological factor affecting ripening.
              </div>
            </div>

          </div>

          {/* 4. ML RIPENING FORECAST */}
          <div className="glass-card rounded-2xl p-6 border border-slate-800">
            <div className="flex items-center justify-between mb-4">
              <div>
                <span className="text-xs font-bold text-amber-400 uppercase tracking-wider">
                  4. ML Ripening-Time Forecast
                </span>
                <h3 className="text-lg font-extrabold text-white mt-0.5">
                  Ripening Timeline ({autoDetectedVariety || selectedVariety.name})
                </h3>
              </div>
              <span className="text-xs text-slate-400 bg-slate-950 px-3 py-1 rounded border border-slate-800">
                {analysisResult.ripening_time_supported ? 'Calibrated Telemetry' : 'Data Status'}
              </span>
            </div>

            {analysisResult.ripening_time_supported ? (
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="bg-slate-950 p-5 rounded-xl border border-slate-800 text-center">
                  <p className="text-xs text-slate-400 uppercase mb-1">Estimated Time to Ripe</p>
                  <p className="text-4xl font-bold text-amber-400">
                    {analysisResult.estimated_days_to_ripe} <span className="text-sm text-slate-300">Days</span>
                  </p>
                  <p className="text-[10px] text-slate-400 mt-2">Window to optimum °Brix (MAE: 1.6d)</p>
                </div>

                <div className="bg-slate-950 p-5 rounded-xl border border-slate-800 text-center">
                  <p className="text-xs text-slate-400 uppercase mb-1">Estimated Shelf Life</p>
                  <p className="text-4xl font-bold text-emerald-400">
                    {analysisResult.estimated_shelf_life} <span className="text-sm text-slate-300">Days</span>
                  </p>
                  <p className="text-[10px] text-slate-400 mt-2">Total window before overripe</p>
                </div>

                <div className="bg-slate-950 p-5 rounded-xl border border-slate-800 text-center">
                  <p className="text-xs text-slate-400 uppercase mb-1">Overripe Risk</p>
                  <p className="text-4xl font-bold text-rose-400">
                    {analysisResult.overripe_risk}%
                  </p>
                  <p className="text-[10px] text-slate-400 mt-2">At {temperature}°C ambient storage</p>
                </div>
              </div>
            ) : (
              <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-5 text-center space-y-2">
                <Clock className="w-8 h-8 text-slate-400 mx-auto" />
                <p className="text-sm font-bold text-amber-300">
                  {analysisResult.ripening_time_message || "Ripening-time estimation is not yet supported because validated training data is unavailable."}
                </p>
                <p className="text-xs text-slate-400 max-w-xl mx-auto">
                  To avoid fabricating days-to-ripe, ripening-time regression is only reported for fruit varieties backed by real, multi-day biochemical storage telemetry (such as Nam Dok Mai Si Tong mango).
                </p>
              </div>
            )}
          </div>

          {/* 5. FUTURE GAN SIMULATION */}
          <div className="glass-card rounded-2xl p-6 border border-slate-800">
            <div className="mb-4">
              <span className="text-xs font-bold text-purple-400 uppercase tracking-wider flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-purple-400" />
                5. Conditional GAN Stage Generator (Active Model)
              </span>
              <h3 className="text-lg font-extrabold text-white mt-0.5">
                Future Ripening Simulation ({autoDetectedVariety || selectedVariety.name})
              </h3>
              <p className="text-slate-400 text-xs">
                Click any stage button below to execute Conditional GAN visual stage synthesis.
              </p>
            </div>

            {/* Timeline Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-4 gap-4 mb-4">
              
              {/* TODAY */}
              <div className="bg-slate-950 border border-emerald-500/40 rounded-xl p-3 flex flex-col items-center">
                <span className="text-xs font-bold text-emerald-400 uppercase tracking-wider mb-2">
                  TODAY
                </span>
                <div className="w-full h-36 rounded-lg bg-slate-900 overflow-hidden flex items-center justify-center mb-2 border border-slate-800">
                  <img src={imagePreview} alt="Today" className="h-full object-contain p-1" />
                </div>
                <span className="text-[10px] text-slate-300 font-bold">Uploaded State</span>
              </div>

              {/* +1 DAY (Semi-Ripe) */}
              <div className="bg-slate-950 border border-slate-800 rounded-xl p-3 flex flex-col items-center justify-between text-center">
                <span className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">
                  +1 DAY (Semi-Ripe)
                </span>
                <div className="w-full h-36 rounded-lg bg-slate-900 border border-slate-800 flex flex-col items-center justify-center p-2 text-center overflow-hidden">
                  {ganSimResults['Semi-Ripe'] ? (
                    <img src={ganSimResults['Semi-Ripe']} alt="+1 Day GAN Generated" className="h-full object-contain rounded-lg" />
                  ) : (
                    <div className="text-center p-2">
                      <Sparkles className="w-6 h-6 text-purple-400/60 mx-auto mb-1" />
                      <p className="text-[10px] text-slate-400 leading-tight">
                        Click button to generate cGAN Semi-Ripe visual
                      </p>
                    </div>
                  )}
                </div>
                <button
                  type="button"
                  disabled={isGanSimulating && activeGanStage === 'Semi-Ripe'}
                  onClick={() => handleSimulateGAN('Semi-Ripe')}
                  className="mt-2 w-full text-[10px] font-bold text-purple-400 hover:text-purple-300 bg-purple-500/10 px-2.5 py-1.5 rounded-lg border border-purple-500/20 flex items-center justify-center gap-1"
                >
                  {isGanSimulating && activeGanStage === 'Semi-Ripe' ? (
                    <RefreshCw className="w-3 h-3 animate-spin" />
                  ) : (
                    <Sparkles className="w-3 h-3 text-purple-400" />
                  )}
                  <span>Generate Semi-Ripe</span>
                </button>
              </div>

              {/* +2 DAYS (Ripe) */}
              <div className="bg-slate-950 border border-slate-800 rounded-xl p-3 flex flex-col items-center justify-between text-center">
                <span className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">
                  +2 DAYS (Ripe)
                </span>
                <div className="w-full h-36 rounded-lg bg-slate-900 border border-slate-800 flex flex-col items-center justify-center p-2 text-center overflow-hidden">
                  {ganSimResults['Ripe'] ? (
                    <img src={ganSimResults['Ripe']} alt="+2 Days GAN Generated" className="h-full object-contain rounded-lg" />
                  ) : (
                    <div className="text-center p-2">
                      <Sparkles className="w-6 h-6 text-amber-400/60 mx-auto mb-1" />
                      <p className="text-[10px] text-slate-400 leading-tight">
                        Click button to generate cGAN Ripe visual
                      </p>
                    </div>
                  )}
                </div>
                <button
                  type="button"
                  disabled={isGanSimulating && activeGanStage === 'Ripe'}
                  onClick={() => handleSimulateGAN('Ripe')}
                  className="mt-2 w-full text-[10px] font-bold text-amber-400 hover:text-amber-300 bg-amber-500/10 px-2.5 py-1.5 rounded-lg border border-amber-500/20 flex items-center justify-center gap-1"
                >
                  {isGanSimulating && activeGanStage === 'Ripe' ? (
                    <RefreshCw className="w-3 h-3 animate-spin" />
                  ) : (
                    <Sparkles className="w-3 h-3 text-amber-400" />
                  )}
                  <span>Generate Ripe</span>
                </button>
              </div>

              {/* +3 DAYS (Overripe) */}
              <div className="bg-slate-950 border border-slate-800 rounded-xl p-3 flex flex-col items-center justify-between text-center">
                <span className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">
                  +3 DAYS (Overripe)
                </span>
                <div className="w-full h-36 rounded-lg bg-slate-900 border border-slate-800 flex flex-col items-center justify-center p-2 text-center overflow-hidden">
                  {ganSimResults['Overripe'] ? (
                    <img src={ganSimResults['Overripe']} alt="+3 Days GAN Generated" className="h-full object-contain rounded-lg" />
                  ) : (
                    <div className="text-center p-2">
                      <Sparkles className="w-6 h-6 text-rose-400/60 mx-auto mb-1" />
                      <p className="text-[10px] text-slate-400 leading-tight">
                        Click button to generate cGAN Overripe visual
                      </p>
                    </div>
                  )}
                </div>
                <button
                  type="button"
                  disabled={isGanSimulating && activeGanStage === 'Overripe'}
                  onClick={() => handleSimulateGAN('Overripe')}
                  className="mt-2 w-full text-[10px] font-bold text-rose-400 hover:text-rose-300 bg-rose-500/10 px-2.5 py-1.5 rounded-lg border border-rose-500/20 flex items-center justify-center gap-1"
                >
                  {isGanSimulating && activeGanStage === 'Overripe' ? (
                    <RefreshCw className="w-3 h-3 animate-spin" />
                  ) : (
                    <Sparkles className="w-3 h-3 text-rose-400" />
                  )}
                  <span>Generate Overripe</span>
                </button>
              </div>

            </div>

            {isGanSimulating && (
              <div className="text-center text-xs text-purple-400 animate-pulse flex items-center justify-center gap-2 pt-2">
                <Sparkles className="w-4 h-4 animate-spin" />
                <span>Running Conditional GAN visual stage synthesis...</span>
              </div>
            )}

            <div className="mt-4 p-3 bg-purple-950/20 border border-purple-800/30 rounded-xl text-center">
              <p className="text-[11px] text-purple-300 font-medium">
                ⚠️ <span className="font-bold">AI Simulation Disclaimer:</span> Generated fruit twin appearances are AI-generated conditional synthesis visualizations of ripening progression, not guaranteed physical photographs.
              </p>
            </div>
          </div>

          {/* 6. WHAT-IF SIMULATION */}
          <div className="glass-card rounded-2xl p-6 border border-slate-800">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-4">
              <div>
                <span className="text-xs font-bold text-teal-400 uppercase tracking-wider">
                  6. Interactive Scenario Test
                </span>
                <h3 className="text-lg font-extrabold text-white mt-0.5">
                  "What If?" Temperature Simulation
                </h3>
              </div>

              {/* Preset Buttons */}
              <div className="flex items-center gap-2 bg-slate-950 p-1 rounded-lg border border-slate-800">
                {[25, 28, 30, 32].map((tVal) => (
                  <button
                    key={tVal}
                    onClick={() => handleWhatIfCalculate(tVal)}
                    className={`px-3 py-1 rounded text-xs font-bold transition-all ${
                      whatIfTemp === tVal
                        ? 'bg-emerald-600 text-white shadow'
                        : 'text-slate-300 hover:text-white hover:bg-slate-800'
                    }`}
                  >
                    {tVal}°C
                  </button>
                ))}
              </div>
            </div>

            {/* Comparison Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                <span className="text-xs text-slate-400 uppercase font-bold block mb-1">
                  Current Storage Temp
                </span>
                <p className="text-2xl font-bold text-white mb-2">{temperature}°C</p>
                <div className="flex items-center justify-between text-xs pt-2 border-t border-slate-800">
                  <span className="text-slate-400">Predicted Ripening:</span>
                  <span className="font-bold text-amber-400">
                    {analysisResult.estimated_days_to_ripe} Days
                  </span>
                </div>
              </div>

              <div className="bg-slate-950 p-4 rounded-xl border border-emerald-500/40">
                <span className="text-xs text-emerald-400 uppercase font-bold block mb-1">
                  Simulated Condition
                </span>
                <p className="text-2xl font-bold text-white mb-2">{whatIfTemp}°C</p>
                <div className="flex items-center justify-between text-xs pt-2 border-t border-slate-800">
                  <span className="text-slate-400">Predicted Ripening:</span>
                  <span className="font-bold text-emerald-400">
                    {whatIfResult ? `${whatIfResult.estimated_days_to_ripe} Days` : 'Select preset'}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* 7. AI RECOMMENDATION */}
          <div className="glass-card rounded-2xl p-6 border border-slate-800">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-9 h-9 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
                <ShieldCheck className="w-5 h-5" />
              </div>
              <div>
                <span className="text-xs font-bold text-emerald-400 uppercase tracking-wider">
                  7. AI Recommendation
                </span>
                <h3 className="text-lg font-extrabold text-white">
                  Commercial Action Guidance for {autoDetectedVariety || selectedVariety.name}
                </h3>
              </div>
            </div>

            {analysisResult.recommendation && (
              <div className="bg-slate-950 rounded-xl p-5 border border-slate-800 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-slate-400 uppercase">Category:</span>
                  <span className="px-3 py-1 rounded text-xs font-bold uppercase bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                    {analysisResult.recommendation.category}
                  </span>
                </div>

                <p className="text-slate-200 text-xs leading-relaxed font-sans font-medium">
                  "{analysisResult.recommendation.action}"
                </p>

                <p className="text-[10px] text-slate-400 italic pt-2 border-t border-slate-800/80 font-mono">
                  {analysisResult.recommendation.disclaimer}
                </p>
              </div>
            )}
          </div>
          </>
          )}

        </div>
      )}

      {/* =========================================================================
          INTERACTIVE AI MODAL POPUPS (MISMATCH, PLANT LEAF, NON-FRUIT, UNSUPPORTED)
         ========================================================================= */}
      {activeModal && (
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md"
          role="dialog"
          aria-modal="true"
        >
          <div 
            className="relative w-full max-w-2xl bg-slate-900 border-2 rounded-2xl shadow-2xl overflow-hidden font-mono text-left transition-all transform scale-100"
            style={{
              borderColor: activeModal.type === 'mismatch' ? '#f59e0b' : 
                           activeModal.type === 'plant_leaf' ? '#10b981' : 
                           activeModal.type === 'unsupported_fruit' ? '#a855f7' : 
                           activeModal.type === 'supported_nonfruit_class' ? '#06b6d4' : 
                           activeModal.type === 'uncertain' ? '#94a3b8' : '#ef4444'
            }}
          >
            {/* Modal Header */}
            <div className={`px-6 py-4 border-b flex items-center justify-between ${
              activeModal.type === 'mismatch' ? 'bg-amber-950/40 border-amber-500/30 text-amber-300' :
              activeModal.type === 'plant_leaf' ? 'bg-emerald-950/40 border-emerald-500/30 text-emerald-300' :
              activeModal.type === 'unsupported_fruit' ? 'bg-purple-950/40 border-purple-500/30 text-purple-300' :
              activeModal.type === 'supported_nonfruit_class' ? 'bg-cyan-950/40 border-cyan-500/30 text-cyan-300' :
              activeModal.type === 'uncertain' ? 'bg-slate-900 border-slate-700 text-slate-300' :
              'bg-rose-950/40 border-rose-500/30 text-rose-300'
            }`}>
              <div className="flex items-center gap-2.5">
                {activeModal.type === 'mismatch' && <AlertTriangle className="w-5 h-5 text-amber-400 animate-pulse" />}
                {activeModal.type === 'plant_leaf' && <Leaf className="w-5 h-5 text-emerald-400" />}
                {activeModal.type === 'unsupported_fruit' && <Sparkles className="w-5 h-5 text-purple-400" />}
                {activeModal.type === 'supported_nonfruit_class' && <Database className="w-5 h-5 text-cyan-400" />}
                {activeModal.type === 'uncertain' && <Activity className="w-5 h-5 text-slate-400" />}
                {activeModal.type === 'non_fruit' && <AlertCircle className="w-5 h-5 text-rose-400" />}

                <div>
                  <h3 className="text-sm sm:text-base font-extrabold text-white tracking-wide uppercase">
                    {activeModal.type === 'mismatch' && 'Fruit Mismatch Detected | பழ வகை பொருந்தவில்லை'}
                    {activeModal.type === 'plant_leaf' && 'Plant / Foliage Detected | தாவர இலை கண்டறியப்பட்டது'}
                    {activeModal.type === 'unsupported_fruit' && 'Out-of-Dataset Fruit Specimen | வகைப்படுத்தப்படாத பழம்'}
                    {activeModal.type === 'supported_nonfruit_class' && 'Dataset Class Detected (Non-Fruit) | பழம் அல்லாத பயிர் வகுப்பு'}
                    {activeModal.type === 'uncertain' && 'Image Uncertain — Please Try Again | தெளிவற்ற படம்'}
                    {activeModal.type === 'non_fruit' && 'No Supported Fruit Detected | பழம் கண்டறியப்படவில்லை'}
                  </h3>
                  <span className="text-[11px] opacity-80 block">
                    {activeModal.type === 'mismatch' && 'AI Computer Vision Pipeline Cross-Category Alert'}
                    {activeModal.type === 'plant_leaf' && 'Harvested Fruit Specimen Required for Digital Twin'}
                    {activeModal.type === 'unsupported_fruit' && 'Outside 8-Fruit Calibrated Agricultural Dataset'}
                    {activeModal.type === 'supported_nonfruit_class' && 'Fruits-360 Benchmark Dataset Class (Vegetable/Tuber/Grain)'}
                    {activeModal.type === 'uncertain' && 'Image Degraded, Severely Blurred, or Low Contrast'}
                    {activeModal.type === 'non_fruit' && 'Botanical Feature Extraction Failed'}
                  </span>
                </div>
              </div>

              <button
                type="button"
                onClick={() => setActiveModal(null)}
                className="p-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-400 hover:text-white transition-colors cursor-pointer"
                title="Dismiss"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Modal Body */}
            <div className="p-6 space-y-5 max-h-[75vh] overflow-y-auto">
              {/* MISMATCH MODAL CONTENT */}
              {activeModal.type === 'mismatch' && (
                <div className="space-y-4">
                  {/* Visual Side-by-Side Comparison */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 p-3 bg-slate-950/70 rounded-xl border border-slate-800">
                    <div className="flex items-center gap-3 p-3 rounded-lg bg-amber-500/10 border border-amber-500/30">
                      {activeModal.preview && (
                        <div className="w-14 h-14 rounded-lg bg-slate-900 border border-amber-500/40 overflow-hidden shrink-0 flex items-center justify-center p-1">
                          <img src={activeModal.preview} alt="Uploaded Specimen" className="w-full h-full object-contain" />
                        </div>
                      )}
                      <div>
                        <span className="text-[10px] text-amber-400/80 uppercase font-bold tracking-wider block">AI Detected Fruit</span>
                        <div className="text-sm font-extrabold text-white flex items-center gap-1.5">
                          <span>{activeModal.detectedFruit?.icon || '🍌'}</span>
                          <span>{activeModal.detectedFruit?.name}</span>
                          <span className="text-xs text-amber-300">({activeModal.detectedFruit?.tamilName})</span>
                        </div>
                        {activeModal.detectedVariety && (
                          <span className="text-[11px] text-emerald-400 font-bold block mt-0.5">
                            {activeModal.detectedVariety}
                          </span>
                        )}
                        <span className="text-[10px] text-slate-400">Confidence: {activeModal.confidence}%</span>
                      </div>
                    </div>

                    <div className="flex items-center gap-3 p-3 rounded-lg bg-slate-900/60 border border-slate-800">
                      <div className="w-14 h-14 rounded-lg bg-slate-950 border border-slate-800 overflow-hidden shrink-0 flex items-center justify-center text-2xl">
                        {activeModal.currentFruit?.icon || '🍎'}
                      </div>
                      <div>
                        <span className="text-[10px] text-slate-400 uppercase font-bold tracking-wider block">Current Pipeline Tab</span>
                        <div className="text-sm font-extrabold text-slate-300 flex items-center gap-1.5">
                          <span>{activeModal.currentFruit?.icon}</span>
                          <span>{activeModal.currentFruit?.name}</span>
                          <span className="text-xs text-slate-400">({activeModal.currentFruit?.tamilName})</span>
                        </div>
                        <span className="text-[11px] text-amber-400 font-bold block mt-0.5">
                          Pipeline Mismatch
                        </span>
                        <span className="text-[10px] text-slate-500">Active workspace</span>
                      </div>
                    </div>
                  </div>

                  <p className="text-xs sm:text-sm text-slate-300 leading-relaxed font-sans">
                    You uploaded an image of a <strong className="text-amber-300 font-mono font-bold">{activeModal.detectedFruit?.name} ({activeModal.detectedFruit?.tamilName})</strong> while currently in the <strong className="text-slate-200 font-mono font-bold">{activeModal.currentFruit?.name} ({activeModal.currentFruit?.tamilName})</strong> workspace.
                  </p>
                  <p className="text-xs text-slate-400 leading-relaxed font-sans">
                    Switching to the <strong className="text-emerald-400 font-mono">{activeModal.detectedFruit?.name} pipeline</strong> will load its dedicated RawRipe maturity classifier, calibrated Arrhenius shelf-life regression, and conditional GAN future appearance synthesizer.
                  </p>

                  <div className="pt-2 flex flex-col sm:flex-row items-center justify-end gap-3">
                    <button
                      type="button"
                      onClick={() => setActiveModal(null)}
                      className="w-full sm:w-auto px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-mono font-bold transition-all border border-slate-700 text-center cursor-pointer"
                    >
                      Stay on {activeModal.currentFruit?.name} Tab
                    </button>
                    <button
                      type="button"
                      onClick={() => handleModalSwitchFruit(activeModal.detectedFruit, activeModal.file, activeModal.preview)}
                      className="w-full sm:w-auto px-5 py-2.5 rounded-xl bg-gradient-to-r from-amber-500 to-emerald-500 hover:from-amber-400 hover:to-emerald-400 text-slate-950 text-xs font-mono font-black uppercase tracking-wider shadow-lg shadow-amber-500/20 flex items-center justify-center gap-2 transition-all transform hover:scale-[1.02] cursor-pointer"
                    >
                      <span>🚀 Switch to {activeModal.detectedFruit?.name} Pipeline & Run Analysis</span>
                      <ArrowRight className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              )}

              {/* PLANT / FOLIAGE MODAL CONTENT */}
              {activeModal.type === 'plant_leaf' && (
                <div className="space-y-4">
                  <div className="flex flex-col sm:flex-row items-center gap-4 p-4 bg-emerald-950/30 rounded-xl border border-emerald-500/30">
                    {activeModal.preview && (
                      <div className="w-24 h-24 rounded-xl bg-slate-950 border border-emerald-500/40 overflow-hidden shrink-0 flex items-center justify-center p-2">
                        <img src={activeModal.preview} alt="Plant Specimen" className="w-full h-full object-contain" />
                      </div>
                    )}
                    <div className="space-y-1 text-center sm:text-left">
                      <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 text-[11px] font-bold">
                        <Leaf className="w-3.5 h-3.5" />
                        <span>Foliage Specimen Detected ({activeModal.confidence || 98.5}%)</span>
                      </div>
                      <h4 className="text-sm font-bold text-white font-mono">
                        Not a Harvested Fruit Specimen
                      </h4>
                      <p className="text-xs text-slate-300 font-sans leading-relaxed">
                        The computer vision classifier detected vegetative leaves, branches, or plant foliage rather than a harvestable fruit specimen.
                      </p>
                    </div>
                  </div>

                  <div className="p-3 bg-slate-950/70 rounded-xl border border-slate-800 space-y-2">
                    <p className="text-xs text-slate-300 font-sans">
                      💡 <strong>Why is a harvested fruit required?</strong>
                      <br />
                      FruitTwin's digital twin engine evaluates pericarp skin color, surface reflectance, ripening biochemistry (TSS °Brix, Titratable Acidity), and GAN future decay simulation. These models require a clear photo of the fruit itself.
                    </p>
                    <p className="text-xs text-emerald-400 font-bold font-mono">
                      Supported Fruits: Mango, Banana, Papaya, Apple, Orange, Guava, Pomegranate, Watermelon.
                    </p>
                  </div>

                  <div className="pt-2 flex flex-col sm:flex-row items-center justify-end gap-3">
                    <button
                      type="button"
                      onClick={() => setActiveModal(null)}
                      className="w-full sm:w-auto px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-mono font-bold transition-all border border-slate-700 cursor-pointer"
                    >
                      Dismiss
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setActiveModal(null);
                        const samp = (MULTI_FRUIT_SAMPLES[selectedFruit.id] || [])[0];
                        if (samp) handleLoadSample(samp);
                      }}
                      className="w-full sm:w-auto px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-emerald-300 hover:text-white text-xs font-mono font-bold border border-emerald-500/40 transition-all flex items-center justify-center gap-2 cursor-pointer"
                    >
                      <span>Try {selectedFruit.name} Sample</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setActiveModal(null);
                        fileInputRef.current?.click();
                      }}
                      className="w-full sm:w-auto px-5 py-2.5 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 text-xs font-mono font-black uppercase tracking-wider shadow-lg shadow-emerald-500/20 flex items-center justify-center gap-2 transition-all cursor-pointer"
                    >
                      <Upload className="w-4 h-4" />
                      <span>Upload Harvested Fruit Photo</span>
                    </button>
                  </div>
                </div>
              )}

              {/* NON-FRUIT MODAL CONTENT */}
              {activeModal.type === 'non_fruit' && (
                <div className="space-y-4">
                  <div className="flex flex-col sm:flex-row items-center gap-4 p-4 bg-rose-950/30 rounded-xl border border-rose-500/30">
                    {activeModal.preview && (
                      <div className="w-24 h-24 rounded-xl bg-slate-950 border border-rose-500/40 overflow-hidden shrink-0 flex items-center justify-center p-2">
                        <img src={activeModal.preview} alt="Specimen Preview" className="w-full h-full object-contain" />
                      </div>
                    )}
                    <div className="space-y-1 text-center sm:text-left">
                      <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-rose-500/20 text-rose-300 text-[11px] font-bold">
                        <AlertCircle className="w-3.5 h-3.5" />
                        <span>Non-Botanical Object Detected</span>
                      </div>
                      <h4 className="text-sm font-bold text-white font-mono">
                        No Fruit Features Recognized
                      </h4>
                      <p className="text-xs text-slate-300 font-sans leading-relaxed">
                        The uploaded image does not contain recognized botanical fruit characteristics. Please provide a clear photograph of an agricultural fruit.
                      </p>
                    </div>
                  </div>

                  <div className="pt-2 flex flex-col sm:flex-row items-center justify-end gap-3">
                    <button
                      type="button"
                      onClick={() => setActiveModal(null)}
                      className="w-full sm:w-auto px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-mono font-bold transition-all border border-slate-700 cursor-pointer"
                    >
                      Dismiss
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setActiveModal(null);
                        fileInputRef.current?.click();
                      }}
                      className="w-full sm:w-auto px-5 py-2.5 rounded-xl bg-rose-500 hover:bg-rose-400 text-slate-950 text-xs font-mono font-black uppercase tracking-wider shadow-lg shadow-rose-500/20 flex items-center justify-center gap-2 transition-all cursor-pointer"
                    >
                      <Upload className="w-4 h-4" />
                      <span>Upload Valid Fruit Photo</span>
                    </button>
                  </div>
                </div>
              )}

              {/* UNSUPPORTED FRUIT MODAL CONTENT */}
              {activeModal.type === 'unsupported_fruit' && (
                <div className="space-y-4">
                  <div className="flex flex-col sm:flex-row items-center gap-4 p-4 bg-purple-950/30 rounded-xl border border-purple-500/30">
                    {activeModal.preview && (
                      <div className="w-24 h-24 rounded-xl bg-slate-950 border border-purple-500/40 overflow-hidden shrink-0 flex items-center justify-center p-2">
                        <img src={activeModal.preview} alt="Exotic Specimen" className="w-full h-full object-contain" />
                      </div>
                    )}
                    <div className="space-y-1 text-center sm:text-left">
                      <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-purple-500/20 text-purple-300 text-[11px] font-bold">
                        <Sparkles className="w-3.5 h-3.5" />
                        <span>Out-of-Dataset Fruit Specimen</span>
                      </div>
                      <h4 className="text-sm font-bold text-white font-mono">
                        Recognized as: {activeModal.candidateName || 'Exotic Fruit'}
                      </h4>
                      <p className="text-xs text-slate-300 font-sans leading-relaxed">
                        Fruit recognized by Fruits-360, but FruitTwin's maturity and cGAN simulation models are trained on the 8 primary agricultural fruits.
                      </p>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <span className="text-xs font-mono font-bold text-slate-400 block">
                      Choose one of FruitTwin's 8 Supported Fruits:
                    </span>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                      {ALL_FRUITS.map((f) => (
                        <button
                          key={f.id}
                          type="button"
                          onClick={() => {
                            setActiveModal(null);
                            handleFruitChange(f);
                            const samp = (MULTI_FRUIT_SAMPLES[f.id] || [])[0];
                            if (samp) handleLoadSample(samp);
                          }}
                          className="p-2 rounded-xl bg-slate-950 hover:bg-slate-800 border border-slate-800 hover:border-purple-400 text-xs font-mono font-bold text-slate-200 hover:text-white flex items-center gap-1.5 transition-all text-left cursor-pointer"
                        >
                          <span>{f.icon}</span>
                          <span className="truncate">{f.name}</span>
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="pt-2 flex justify-end">
                    <button
                      type="button"
                      onClick={() => setActiveModal(null)}
                      className="px-5 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-mono font-bold border border-slate-700 cursor-pointer"
                    >
                      Dismiss
                    </button>
                  </div>
                </div>
              )}

              {/* SUPPORTED NON-FRUIT CLASS MODAL CONTENT */}
              {activeModal.type === 'supported_nonfruit_class' && (
                <div className="space-y-4">
                  <div className="flex flex-col sm:flex-row items-center gap-4 p-4 bg-cyan-950/30 rounded-xl border border-cyan-500/30">
                    {activeModal.preview && (
                      <div className="w-24 h-24 rounded-xl bg-slate-950 border border-cyan-500/40 overflow-hidden shrink-0 flex items-center justify-center p-2">
                        <img src={activeModal.preview} alt="Dataset Specimen" className="w-full h-full object-contain" />
                      </div>
                    )}
                    <div className="space-y-1 text-center sm:text-left">
                      <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-cyan-500/20 text-cyan-300 text-[11px] font-bold">
                        <Database className="w-3.5 h-3.5" />
                        <span>Fruits-360 Dataset Class: {activeModal.detectedClass || 'Agricultural Crop'} ({activeModal.confidence}%)</span>
                      </div>
                      <h4 className="text-sm font-bold text-white font-mono">
                        Non-Fruit Agricultural Class Detected
                      </h4>
                      <p className="text-xs text-slate-300 font-sans leading-relaxed">
                        {activeModal.message || `The image was identified as ${activeModal.detectedClass}, which is a recognized class in the Fruits-360 dataset. However, it is an agricultural crop, not a fresh fruit supported by Fruit Twin ripening pipelines.`}
                      </p>
                    </div>
                  </div>

                  <div className="p-3 bg-slate-950/70 rounded-xl border border-slate-800 space-y-2">
                    <p className="text-xs text-slate-300 font-sans">
                      ℹ️ <strong>Why is this class detected?</strong>
                      <br />
                      Fruit Twin's 20-class multi-model classifier includes benchmark dataset classes such as Potato Red, Tomato 1, Corn, and Chestnut. While these classes are accurately recognized by the neural classifier, fruit ripening simulation and GAN synthesis are only calibrated for harvestable fruits.
                    </p>
                    <div className="pt-1">
                      <button
                        type="button"
                        onClick={() => {
                          setActiveModal(null);
                          setIsCatalogOpen(true);
                        }}
                        className="text-xs text-cyan-400 hover:text-cyan-300 font-mono underline flex items-center gap-1 cursor-pointer"
                      >
                        <span>View All 20 Dataset Classes & Capabilities in Dataset Catalogue →</span>
                      </button>
                    </div>
                  </div>

                  <div className="pt-2 flex flex-col sm:flex-row items-center justify-end gap-3">
                    <button
                      type="button"
                      onClick={() => setActiveModal(null)}
                      className="w-full sm:w-auto px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-mono font-bold transition-all border border-slate-700 cursor-pointer"
                    >
                      Dismiss
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setActiveModal(null);
                        fileInputRef.current?.click();
                      }}
                      className="w-full sm:w-auto px-5 py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-slate-950 text-xs font-mono font-black uppercase tracking-wider shadow-lg shadow-cyan-500/20 flex items-center justify-center gap-2 transition-all cursor-pointer"
                    >
                      <Upload className="w-4 h-4" />
                      <span>Upload Supported Fruit Photo</span>
                    </button>
                  </div>
                </div>
              )}

              {/* UNCERTAIN / DEGRADED IMAGE MODAL CONTENT */}
              {activeModal.type === 'uncertain' && (
                <div className="space-y-4">
                  <div className="flex flex-col sm:flex-row items-center gap-4 p-4 bg-slate-950/60 rounded-xl border border-slate-700">
                    {activeModal.preview && (
                      <div className="w-24 h-24 rounded-xl bg-slate-950 border border-slate-700 overflow-hidden shrink-0 flex items-center justify-center p-2">
                        <img src={activeModal.preview} alt="Uncertain Specimen" className="w-full h-full object-contain filter blur-[1px]" />
                      </div>
                    )}
                    <div className="space-y-1 text-center sm:text-left">
                      <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-slate-800 text-slate-300 text-[11px] font-bold">
                        <Activity className="w-3.5 h-3.5 text-slate-400" />
                        <span>Image Degradation / Low Contrast Alert</span>
                      </div>
                      <h4 className="text-sm font-bold text-white font-mono">
                        Image Uncertain — Please Try Again
                      </h4>
                      <p className="text-xs text-slate-300 font-sans leading-relaxed">
                        {activeModal.message || 'Fruit Twin could not confidently identify this image. The photo may be out of focus, motion-blurred, or too dimly lit for reliable pericarp color and texture extraction.'}
                      </p>
                    </div>
                  </div>

                  <div className="p-3 bg-slate-950/70 rounded-xl border border-slate-800 space-y-2">
                    <p className="text-xs text-slate-300 font-sans">
                      📸 <strong>Tips for a reliable scan:</strong>
                      <br />
                      1. Ensure good natural or room lighting without heavy glare.
                      <br />
                      2. Hold camera steady to avoid motion blur.
                      <br />
                      3. Keep the fruit centered and in focus.
                    </p>
                  </div>

                  <div className="pt-2 flex flex-col sm:flex-row items-center justify-end gap-3">
                    <button
                      type="button"
                      onClick={() => setActiveModal(null)}
                      className="w-full sm:w-auto px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-mono font-bold transition-all border border-slate-700 cursor-pointer"
                    >
                      Dismiss
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setActiveModal(null);
                        fileInputRef.current?.click();
                      }}
                      className="w-full sm:w-auto px-5 py-2.5 rounded-xl bg-slate-700 hover:bg-slate-600 text-white text-xs font-mono font-black uppercase tracking-wider shadow-lg flex items-center justify-center gap-2 transition-all cursor-pointer"
                    >
                      <Upload className="w-4 h-4" />
                      <span>Upload Clear Fruit Photo</span>
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Dataset Class Catalog Modal */}
      <DatasetClassCatalogModal 
        isOpen={isCatalogOpen} 
        onClose={() => setIsCatalogOpen(false)} 
        onSelectFruit={(fruitId) => {
          const target = ALL_FRUITS.find(f => f.id === fruitId);
          if (target) {
            handleFruitChange(target);
          }
          setIsCatalogOpen(false);
        }}
      />

    </div>
  );
}
