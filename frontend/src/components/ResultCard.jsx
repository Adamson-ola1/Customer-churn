import React from "react";

const riskColor = { Low: "#2E7D32", Medium: "#F9A825", High: "#C62828" };

export default function ResultCard({ result }) {
  if (!result) return null;
  const { churn_prediction, churn_probability, risk_level } = result;

  return (
    <div className="result-card" style={{ borderColor: riskColor[risk_level] }}>
      <h3>Prediction Result</h3>
      <p className="result-headline">
        {churn_prediction ? "⚠️ Likely to Churn" : "✅ Likely to Stay"}
      </p>
      <div className="result-metrics">
        <div>
          <span className="label">Churn Probability</span>
          <span className="value">{(churn_probability * 100).toFixed(1)}%</span>
        </div>
        <div>
          <span className="label">Risk Level</span>
          <span className="value" style={{ color: riskColor[risk_level] }}>{risk_level}</span>
        </div>
      </div>
      <div className="progress-bar">
        <div
          className="progress-fill"
          style={{ width: `${churn_probability * 100}%`, background: riskColor[risk_level] }}
        />
      </div>
    </div>
  );
}
