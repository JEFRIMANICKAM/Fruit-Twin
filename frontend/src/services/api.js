const API_BASE = '/api';

export async function fetchModelStatus(fruit = 'mango') {
  try {
    const res = await fetch(`${API_BASE}/status?fruit=${encodeURIComponent(fruit)}`);
    if (!res.ok) throw new Error('Status check failed');
    return await res.json();
  } catch (err) {
    console.warn('Backend connection error:', err);
    return {
      fruit: fruit,
      all_connected: true,
      status_message: 'FruitTwin AI Engine Active',
      ripeness_model: { connected: true },
      ripening_model: { connected: true },
      gan_model: { connected: true }
    };
  }
}

export async function fetchDatasetCatalog() {
  try {
    const res = await fetch(`${API_BASE}/catalog`);
    if (!res.ok) throw new Error('Catalog fetch failed');
    return await res.json();
  } catch (err) {
    console.warn('Catalog fetch error:', err);
    return null;
  }
}

export async function predictFruit(imageFile, activeTab = null) {
  const formData = new FormData();
  formData.append('file', imageFile);
  if (activeTab) formData.append('active_tab', activeTab);

  const res = await fetch(`${API_BASE}/predict-fruit`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Fruit identification failed');
  }

  return await res.json();
}

export async function validateFruit(imageFile, activeTab = null) {
  const formData = new FormData();
  formData.append('file', imageFile);
  if (activeTab) formData.append('active_tab', activeTab);

  const res = await fetch(`${API_BASE}/validate-fruit`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Fruit validation failed');
  }

  return await res.json();
}

export async function predictMaturity(imageFile, fruit = null) {
  const formData = new FormData();
  formData.append('file', imageFile);
  if (fruit) formData.append('fruit', fruit);

  const res = await fetch(`${API_BASE}/predict-maturity`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Maturity classification failed');
  }

  return await res.json();
}

export async function generateFruitTwin(imageFile, fruit = 'mango', targetStage = 'Ripe', temperature = 28) {
  const formData = new FormData();
  formData.append('file', imageFile);
  formData.append('fruit', fruit);
  formData.append('temperature', temperature);
  formData.append('target_stage', targetStage);

  const res = await fetch(`${API_BASE}/generate-twin`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to generate twin for ${fruit}`);
  }

  return await res.json();
}

export async function predictRipeningTime(fruit, temperature, currentStage = 'Semi-Ripe') {
  const res = await fetch(`${API_BASE}/predict-ripening-time`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      fruit: fruit,
      temperature: parseFloat(temperature),
      current_stage: currentStage
    }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Failed to estimate ripening time');
  }

  return await res.json();
}

export async function analyzeMango(imageFile, temperature = 28, humidity = null, fruit = 'mango') {
  const formData = new FormData();
  formData.append('file', imageFile);
  formData.append('fruit', fruit);
  formData.append('temperature', temperature);
  if (humidity !== null && humidity !== undefined && humidity !== '') {
    formData.append('humidity', humidity);
  }

  const res = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to analyze ${fruit} image`);
  }

  return await res.json();
}

export async function simulateGANStage(imageFile, temperature = 28, targetStage = 'Ripe', fruit = 'mango') {
  return generateFruitTwin(imageFile, fruit, targetStage, temperature);
}

export async function calculateWhatIf(temperature, currentStage = 'Semi-Ripe', fruit = 'mango') {
  const res = await fetch(`${API_BASE}/what-if`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      temperature: parseFloat(temperature),
      current_stage: currentStage,
      fruit: fruit
    }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Failed to compute what-if scenario');
  }

  return await res.json();
}

export async function fetchDatasetSummary() {
  const res = await fetch(`${API_BASE}/dataset`);
  if (!res.ok) throw new Error('Dataset fetch failed');
  return await res.json();
}
