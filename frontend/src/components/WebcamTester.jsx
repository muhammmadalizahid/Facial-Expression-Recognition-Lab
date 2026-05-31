import React, { useEffect, useRef, useState } from "react";
import { FilesetResolver, FaceDetector } from "@mediapipe/tasks-vision";
import { predictFromBlob } from "../api";
import { applyCalibration, CLASSES, EMOJI_MAP, roundMap, topClassFromMap } from "../utils/emotionCalibration";

const TARGET_FPS = 13;
const INTERVAL_MS = Math.round(1000 / TARGET_FPS);
const EMA_ALPHA = 0.3;
const SWITCH_MARGIN = 3;
const SWITCH_CONFIRM_FRAMES = 2;

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

function WebcamTester({ modelLoaded, onPrediction, onStatusChange, onError }) {
  const videoRef = useRef(null);
  const overlayRef = useRef(null);

  const streamRef = useRef(null);
  const detectorRef = useRef(null);
  const timerRef = useRef(null);

  const requestInFlightRef = useRef(false);
  const latestTaskRef = useRef(null);

  const emaRef = useRef(null);
  const stableTopRef = useRef(null);
  const pendingSwitchRef = useRef({ candidate: null, count: 0 });

  const [cameraOn, setCameraOn] = useState(false);

  useEffect(() => {
    return () => stopCamera();
  }, []);

  function setStatus(nextStatus) {
    onStatusChange(nextStatus);
  }

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
      runningMode: "VIDEO",
      minDetectionConfidence: 0.5,
    });

    return detectorRef.current;
  }

  function clearOverlay() {
    const canvas = overlayRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  function stopCamera() {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    latestTaskRef.current = null;
    requestInFlightRef.current = false;
    clearOverlay();

    setCameraOn(false);
    setStatus("Stopped");
  }

  async function startCamera() {
    onError("");
    setStatus("Starting...");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 960, height: 540, facingMode: "user" },
        audio: false,
      });

      streamRef.current = stream;
      const video = videoRef.current;
      video.srcObject = stream;
      await video.play();
      await initDetector();

      setCameraOn(true);
      setStatus("Running");

      timerRef.current = setInterval(scheduleFrameTask, INTERVAL_MS);
    } catch (error) {
      setStatus("Camera failed");
      onError(`Camera error: ${error.message}`);
    }
  }

  function drawBox(box) {
    const video = videoRef.current;
    const canvas = overlayRef.current;
    if (!video || !canvas) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!box) return;

    ctx.lineWidth = 2;
    ctx.strokeStyle = "#89b4ff";
    ctx.strokeRect(box.x, box.y, box.width, box.height);
  }

  function getLargestFace(detections) {
    if (!detections?.length) return null;
    return detections
      .map((det) => normalizeBox(det.boundingBox))
      .filter(Boolean)
      .sort((a, b) => b.width * b.height - a.width * a.height)[0];
  }

  async function cropFaceToBlob(box) {
    const video = videoRef.current;
    if (!video || !box) return null;

    const temp = document.createElement("canvas");
    const ctx = temp.getContext("2d");

    const padX = box.width * 0.2;
    const padY = box.height * 0.2;

    const sx = clamp(Math.floor(box.x - padX), 0, video.videoWidth - 1);
    const sy = clamp(Math.floor(box.y - padY), 0, video.videoHeight - 1);
    const sw = clamp(Math.floor(box.width + padX * 2), 1, video.videoWidth - sx);
    const sh = clamp(Math.floor(box.height + padY * 2), 1, video.videoHeight - sy);

    temp.width = sw;
    temp.height = sh;
    ctx.drawImage(video, sx, sy, sw, sh, 0, 0, sw, sh);

    return new Promise((resolve) => temp.toBlob(resolve, "image/jpeg", 0.92));
  }

  function scheduleFrameTask() {
    const video = videoRef.current;
    const detector = detectorRef.current;

    if (!video || !detector || video.readyState < 2) return;

    if (!modelLoaded) {
      setStatus("Waiting for model");
      return;
    }

    const result = detector.detectForVideo(video, performance.now());
    const box = getLargestFace(result?.detections || []);
    drawBox(box);

    if (!box) {
      latestTaskRef.current = null;
      setStatus("No face detected");
      return;
    }

    setStatus("Face detected");

    // Latest-frame-wins queue: replace pending frame with newest frame.
    latestTaskRef.current = { box };

    if (!requestInFlightRef.current) {
      void processLatestTask();
    }
  }

  function updateEma(probabilities) {
    if (!emaRef.current) {
      emaRef.current = { ...probabilities };
      return emaRef.current;
    }

    const next = {};
    for (const cls of CLASSES) {
      const prev = Number(emaRef.current[cls] ?? 0);
      const curr = Number(probabilities[cls] ?? 0);
      next[cls] = prev * (1 - EMA_ALPHA) + curr * EMA_ALPHA;
    }

    emaRef.current = next;
    return next;
  }

  function chooseStableTop(emaMap) {
    const candidate = topClassFromMap(emaMap);

    if (!stableTopRef.current) {
      stableTopRef.current = candidate;
      pendingSwitchRef.current = { candidate: null, count: 0 };
      return candidate;
    }

    const stable = stableTopRef.current;
    if (candidate === stable) {
      pendingSwitchRef.current = { candidate: null, count: 0 };
      return stable;
    }

    const pending = pendingSwitchRef.current;
    if (pending.candidate === candidate) {
      pending.count += 1;
    } else {
      pendingSwitchRef.current = { candidate, count: 1 };
    }

    const candidateValue = Number(emaMap[candidate] ?? 0);
    const stableValue = Number(emaMap[stable] ?? 0);
    const confidentLead = candidateValue >= stableValue + SWITCH_MARGIN;
    const confirmed = pendingSwitchRef.current.count >= SWITCH_CONFIRM_FRAMES;

    if (confidentLead && confirmed) {
      stableTopRef.current = candidate;
      pendingSwitchRef.current = { candidate: null, count: 0 };
      return candidate;
    }

    return stable;
  }

  async function processLatestTask() {
    const task = latestTaskRef.current;
    if (!task) return;

    latestTaskRef.current = null;
    requestInFlightRef.current = true;

    try {
      const blob = await cropFaceToBlob(task.box);
      if (!blob) {
        throw new Error("Could not create face crop from camera frame.");
      }

      const result = await predictFromBlob(blob);
      const emaMap = updateEma(result.probabilities);
      const calibratedMap = applyCalibration(emaMap);
      const stableTop = chooseStableTop(calibratedMap);

      // Show confidence from the smoothed model output (not post-calibrated score).
      const stableConfidence = Number(emaMap[stableTop] ?? 0).toFixed(1);

      // UI updates are emitted only after receiving backend response.
      onPrediction({
        topClass: stableTop,
        confidence: stableConfidence,
        inferenceMs: result.inference_ms,
        emoji: EMOJI_MAP[stableTop] || "",
        allProbabilities: roundMap(emaMap),
        calibratedProbabilities: roundMap(calibratedMap),
      });

      onError("");
    } catch (error) {
      onError(`Prediction error: ${error.message}`);
    } finally {
      requestInFlightRef.current = false;
      if (latestTaskRef.current) {
        void processLatestTask();
      }
    }
  }

  return (
    <div className="camera-card">
      <div className="camera-head">
        <h2>Live Camera</h2>
        <div className="camera-actions">
          <button onClick={startCamera} disabled={cameraOn}>Start</button>
          <button className="ghost" onClick={stopCamera} disabled={!cameraOn}>Stop</button>
        </div>
      </div>

      <div className="video-wrap">
        <video ref={videoRef} autoPlay muted playsInline className="webcam-video" />
        <canvas ref={overlayRef} className="overlay-canvas" />
      </div>
    </div>
  );
}

export default WebcamTester;
