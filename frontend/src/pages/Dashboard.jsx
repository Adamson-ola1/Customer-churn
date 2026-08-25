import React, { useState } from "react";
import PredictionForm from "../components/PredictionForm";
import ResultCard from "../components/ResultCard";
import { useAppContext } from "../context/AppContext";

export default function Dashboard() {
  const [result, setResult] = useState(null);
  const { addPrediction, history } = useAppContext();

  const handleResult = (res, input) => {
    setResult(res);
    addPrediction(res, input);
  };

  return (
    <div className="page">
      <h1>Customer Churn Risk Checker</h1>
      <p className="page-subtitle">
        Enter a customer's profile to estimate their probability of churning.
      </p>

      <div className="dashboard-grid">
        <PredictionForm onResult={handleResult} />
        <ResultCard result={result} />
      </div>

      {history.length > 0 && (
        <div className="history-table-wrap">
          <h2>Recent Predictions</h2>
          <table className="table">
            <thead>
              <tr>
                <th>Country</th><th>Age</th><th>Balance</th>
                <th>Probability</th><th>Risk</th>
              </tr>
            </thead>
            <tbody>
              {history.map((h, i) => (
                <tr key={i}>
                  <td>{h.country}</td>
                  <td>{h.age}</td>
                  <td>{h.balance?.toLocaleString()}</td>
                  <td>{(h.churn_probability * 100).toFixed(1)}%</td>
                  <td>{h.risk_level}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
