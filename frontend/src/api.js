const API_BASE = "http://localhost:8000";

async function parseJSON(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data?.detail || data?.message || "Request failed";
    throw new Error(detail);
  }
  return data;
}

export async function health() {
  const response = await fetch(`${API_BASE}/health`);
  return parseJSON(response);
}

export async function loadModel(modelPath) {
  const response = await fetch(`${API_BASE}/load-model`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_path: modelPath }),
  });
  return parseJSON(response);
}

export async function predictFromBlob(blob) {
  const formData = new FormData();
  formData.append("file", blob, "frame.jpg");
  const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    body: formData,
  });
  return parseJSON(response);
}

export async function predictFromBase64(image) {
  const response = await fetch(`${API_BASE}/predict-base64`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image }),
  });
  return parseJSON(response);
}
