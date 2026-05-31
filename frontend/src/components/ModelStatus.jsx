import React from "react";

function ModelStatus({ modelInfo, serverInfo, error }) {
  const info = modelInfo || serverInfo;

  return (
    <div className="card">
      <h2>Model Status</h2>
      <p><strong>Server:</strong> {serverInfo ? "Online" : "Offline"}</p>
      <p><strong>Model Loaded:</strong> {info?.model_loaded ? "Yes" : "No"}</p>
      <p><strong>Device:</strong> {info?.device || "-"}</p>
      <p><strong>Model Path:</strong> {info?.model_path || "-"}</p>
      <p><strong>Classes:</strong> {info?.classes?.join(", ") || "-"}</p>
      <p><strong>Val Accuracy:</strong> {info?.metadata?.val_accuracy ?? "-"}</p>
      <p><strong>RAF Accuracy:</strong> {info?.metadata?.raf_accuracy ?? "-"}</p>
      <p><strong>Best Epoch:</strong> {info?.metadata?.best_epoch ?? "-"}</p>
      {error ? <p className="error-text">{error}</p> : null}
    </div>
  );
}

export default ModelStatus;
