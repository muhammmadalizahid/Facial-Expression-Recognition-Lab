import React from "react";

function ProbabilityBars({ title, classes, probabilities, topClass }) {
  return (
    <div className="card">
      <h2>{title}</h2>
      {!probabilities ? (
        <p className="hint">No predictions yet.</p>
      ) : (
        <div className="bars-wrap">
          {classes.map((name) => {
            const value = Number(probabilities[name] ?? 0);
            const active = topClass === name;
            return (
              <div key={name} className="bar-item">
                <div className="bar-label-row">
                  <span className={active ? "top-class" : ""}>{name}</span>
                  <span>{value.toFixed(1)}%</span>
                </div>
                <div className="bar-track">
                  <div
                    className={`bar-fill ${active ? "active" : ""}`}
                    style={{ width: `${Math.max(0, Math.min(100, value))}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default ProbabilityBars;
