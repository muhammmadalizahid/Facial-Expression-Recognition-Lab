import React, { useEffect, useMemo, useState } from "react";
import { health, loadModel } from "./api";
import TestImageUpload from "./components/TestImageUpload";
import WebcamTester from "./components/WebcamTester";
import { CLASSES, EMOJI_MAP } from "./utils/emotionCalibration";

const DEFAULT_MODEL_PATH = "backend/models/fer_deploy_b3_160_stratified_cleanval_v3.pth";

function App() {
  const [bootStatus, setBootStatus] = useState("Starting model...");
  const [modelReady, setModelReady] = useState(false);
  const [error, setError] = useState("");

  const [livePrediction, setLivePrediction] = useState(null);
  const [imagePrediction, setImagePrediction] = useState(null);
  const [cameraStatus, setCameraStatus] = useState("Idle");

  useEffect(() => {
    let cancelled = false;

    async function bootstrapModel() {
      setError("");
      setBootStatus("Starting model...");

      try {
        const info = await health();
        if (cancelled) return;

        if (info.model_loaded) {
          setModelReady(true);
          setBootStatus("Model ready");
          return;
        }

        setBootStatus("Loading model...");
        await loadModel(DEFAULT_MODEL_PATH);

        if (cancelled) return;
        setModelReady(true);
        setBootStatus("Model ready");
      } catch (bootstrapError) {
        if (cancelled) return;
        setModelReady(false);
        setBootStatus("Model failed");
        setError(bootstrapError.message);
      }
    }

    void bootstrapModel();
    return () => {
      cancelled = true;
    };
  }, []);

  const legend = useMemo(
    () => CLASSES.map((name) => ({ name, emoji: EMOJI_MAP[name] || "" })),
    []
  );

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">FER Testing Dashboard</p>
          <h1>Facial Expression Recognition Lab</h1>
          <p className="subtitle">Test live camera and uploaded images with calibrated output.</p>
        </div>
        <span className={`pill ${modelReady ? "pill-ok" : "pill-wait"}`}>{bootStatus}</span>
      </header>

      {error ? <div className="error-banner">{error}</div> : null}

      <section className="hero-layout">
        <div className="camera-panel">
          <WebcamTester
            modelLoaded={modelReady}
            onPrediction={setLivePrediction}
            onStatusChange={setCameraStatus}
            onError={setError}
          />
        </div>

        <aside className="result-panel">
          <div className="result-card">
            <h2>Live Result</h2>
            {livePrediction ? (
              <>
                <div className="hero-emoji">{livePrediction.emoji}</div>
                <div className="hero-text">{livePrediction.topClass}</div>
              </>
            ) : (
              <div className="hero-placeholder">Awaiting live inference</div>
            )}
            <div className="confidence-row">
              <span>Confidence</span>
              <strong>{livePrediction?.confidence ? `${livePrediction.confidence}%` : "-"}</strong>
            </div>
            <div className="meta-row">
              <span>Inference Time</span>
              <strong>{livePrediction?.inferenceMs ? `${livePrediction.inferenceMs} ms` : "-"}</strong>
            </div>
            <div className="meta-row">
              <span>Camera Status</span>
              <strong>{cameraStatus}</strong>
            </div>
          </div>
        </aside>
      </section>

      <section className="dual-layout">
        <TestImageUpload
          modelLoaded={modelReady}
          onPrediction={setImagePrediction}
          onError={setError}
        />

        <aside className="image-side-panel">
          <div className="result-card secondary">
            <h2>Image Result</h2>
            {imagePrediction ? (
              <>
                <div className="hero-emoji small">{imagePrediction.emoji}</div>
                <div className="hero-text">{imagePrediction.topClass}</div>
              </>
            ) : (
              <div className="hero-placeholder">Awaiting image inference</div>
            )}
            <div className="confidence-row">
              <span>Confidence</span>
              <strong>{imagePrediction?.confidence ? `${imagePrediction.confidence}%` : "-"}</strong>
            </div>
            <div className="meta-row">
              <span>Inference Time</span>
              <strong>{imagePrediction?.inferenceMs ? `${imagePrediction.inferenceMs} ms` : "-"}</strong>
            </div>
          </div>

          <div className="legend-card">
            <h3>Emotion Reference</h3>
            <div className="legend-grid">
              {legend.map((item) => (
                <div key={item.name} className="legend-chip">
                  <span className="legend-emoji">{item.emoji}</span>
                  <span className="legend-label">{item.name}</span>
                </div>
              ))}
            </div>
          </div>
        </aside>
      </section>

      <section className="methodology-card">
        <h2>How This Project Works</h2>
        <div className="method-stack">
          <article className="method-item">
            <h4>1. Data and Labels</h4>
            <p>
              The model predicts seven expressions: angry, disgust, fear, happy, neutral, sad, and surprise.
              Training uses balanced data splits and tracked validation checks.
            </p>
          </article>
          <article className="method-item">
            <h4>2. Model Design</h4>
            <p>
              It uses EfficientNet-B3 as the main network with a custom final head for the seven classes.
              Input is RGB, resized to 160x160, and normalized with ImageNet mean and std values.
            </p>
          </article>
          <article className="method-item">
            <h4>3. Vision Pipeline</h4>
            <p>
              For webcam mode, a face is found first and cropped with padding before prediction.
              A latest-frame-first queue keeps the app responsive by skipping old frames.
            </p>
          </article>
          <article className="method-item">
            <h4>4. Practical AI Engineering</h4>
            <p>
              The system combines real-time inference, calibrated class tuning, and clean UI-based evaluation.
              In this project history, this setup is the best personal result achieved so far.
            </p>
          </article>
        </div>
      </section>
    </div>
  );
}

export default App;
