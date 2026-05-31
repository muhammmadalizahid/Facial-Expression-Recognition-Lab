import React, { useRef, useState } from "react";
import { FilesetResolver, FaceDetector } from "@mediapipe/tasks-vision";
import { predictFromBlob } from "../api";
import { applyCalibration, EMOJI_MAP, topClassFromMap } from "../utils/emotionCalibration";

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function normalizeBox(rawBox) {
  if (!rawBox) return null;
  const x = Number(rawBox.originX ?? rawBox.x);
  const y = Number(rawBox.originY ?? rawBox.y);
  const width = Number(rawBox.width);
  const height = Number(rawBox.height);
  if (![x, y, width, height].every(Number.isFinite)) return null;
  return { x, y, width, height };
}

async function blobFromCanvas(canvas) {
  return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
}

function TestImageUpload({ modelLoaded, onPrediction, onError }) {
  const [status, setStatus] = useState("Idle");
  const [previewUrl, setPreviewUrl] = useState("");
  const [allowFullFrame, setAllowFullFrame] = useState(false);
  const [isDragActive, setIsDragActive] = useState(false);
  const detectorRef = useRef(null);
  const inputRef = useRef(null);

  async function initDetector() {
    if (detectorRef.current) return detectorRef.current;

    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );

    detectorRef.current = await FaceDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
      },
      runningMode: "IMAGE",
      minDetectionConfidence: 0.5,
    });

    return detectorRef.current;
  }

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = src;
    });
  }

  async function buildPredictionBlob(img, detectionResult) {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    const detections = detectionResult?.detections || [];
    const box = detections
      .map((d) => d.boundingBox)
      .map(normalizeBox)
      .filter(Boolean)
      .sort((a, b) => b.width * b.height - a.width * a.height)[0];

    if (!box) {
      if (!allowFullFrame) return null;
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      return blobFromCanvas(canvas);
    }

    const padX = box.width * 0.2;
    const padY = box.height * 0.2;
    const sx = clamp(Math.floor(box.x - padX), 0, img.width - 1);
    const sy = clamp(Math.floor(box.y - padY), 0, img.height - 1);
    const sw = clamp(Math.floor(box.width + padX * 2), 1, img.width - sx);
    const sh = clamp(Math.floor(box.height + padY * 2), 1, img.height - sy);

    canvas.width = sw;
    canvas.height = sh;
    ctx.drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);
    return blobFromCanvas(canvas);
  }

  async function processFile(file) {
    if (!file) return;

    if (!modelLoaded) {
      onError("Model not loaded. Please wait for model startup.");
      return;
    }

    setStatus("Processing image...");
    onError("");

    const src = URL.createObjectURL(file);
    setPreviewUrl(src);

    try {
      const image = await loadImage(src);
      const detector = await initDetector();
      const detection = detector.detect(image);
      const blob = await buildPredictionBlob(image, detection);

      if (!blob) {
        throw new Error("No face found. Enable full image fallback.");
      }

      const result = await predictFromBlob(blob);
      const calibrated = applyCalibration(result.probabilities);
      const top = topClassFromMap(calibrated);

      onPrediction({
        topClass: top,
        confidence: Number(calibrated[top] ?? 0).toFixed(1),
        inferenceMs: result.inference_ms,
        emoji: EMOJI_MAP[top] || "",
      });

      setStatus("Prediction complete");
    } catch (err) {
      setStatus("Failed");
      onError(`Image prediction failed: ${err.message}`);
    }
  }

  async function handleUpload(event) {
    const file = event.target.files?.[0];
    await processFile(file);
  }

  function handleDragOver(event) {
    event.preventDefault();
    setIsDragActive(true);
  }

  function handleDragLeave(event) {
    event.preventDefault();
    setIsDragActive(false);
  }

  async function handleDrop(event) {
    event.preventDefault();
    setIsDragActive(false);
    const file = event.dataTransfer?.files?.[0];
    await processFile(file);
  }

  return (
    <section className="upload-card">
      <div className="panel-head">
        <h3>Image Inference</h3>
        <p>Upload one image and get a calibrated emotion result.</p>
      </div>
      <div className="upload-controls">
        <label className="upload-button" htmlFor="image-input">Choose Image</label>
        <input
          id="image-input"
          ref={inputRef}
          type="file"
          accept="image/*"
          onChange={handleUpload}
        />
        <label className="toggle-inline">
          <input
            type="checkbox"
            checked={allowFullFrame}
            onChange={(e) => setAllowFullFrame(e.target.checked)}
          />
          Use full image if no face
        </label>
      </div>
      <div
        className={`drop-zone ${isDragActive ? "active" : ""}`}
        onDragOver={handleDragOver}
        onDragEnter={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            inputRef.current?.click();
          }
        }}
      >
        <p>Drag and drop an image here</p>
        <span>or click to browse</span>
      </div>
      <p className="status-text">Status: {status}</p>

      <div className="preview-frame">
        {previewUrl ? (
          <img src={previewUrl} className="upload-preview" alt="Uploaded face" />
        ) : (
          <div className="preview-placeholder">Image preview will appear here</div>
        )}
      </div>
    </section>
  );
}

export default TestImageUpload;
