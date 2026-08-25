import React from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import useApi from "../hooks/useApi";
import { getModelInfo } from "../api/predictionApi";
import Loader from "../components/Loader";

export default function ModelInfo() {
  const { data, loading, error } = useApi(getModelInfo, []);

  if (loading) return <div className="page"><Loader label="Loading model info..." /></div>;
  if (error) return <div className="page"><p className="form-error">Could not load model info. Is the backend running?</p></div>;

  const { best_model, results, features } = data;

  return (
    <div className="page">
      <h1>Model Information</h1>
      <p className="page-subtitle">Best model selected by ROC AUC: <strong>{best_model}</strong></p>

      <div className="chart-card">
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={results}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="model" />
            <YAxis domain={[0, 1]} />
            <Tooltip />
            <Legend />
            <Bar dataKey="accuracy" fill="#4FA695" />
            <Bar dataKey="precision" fill="#3A6EA5" />
            <Bar dataKey="recall" fill="#F9A825" />
            <Bar dataKey="f1_score" fill="#8E44AD" />
            <Bar dataKey="roc_auc" fill="#C62828" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <h2>Metrics Table</h2>
      <table className="table">
        <thead>
          <tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC AUC</th></tr>
        </thead>
        <tbody>
          {results.map((r) => (
            <tr key={r.model} className={r.model === best_model ? "highlight-row" : ""}>
              <td>{r.model}</td>
              <td>{r.accuracy.toFixed(3)}</td>
              <td>{r.precision.toFixed(3)}</td>
              <td>{r.recall.toFixed(3)}</td>
              <td>{r.f1_score.toFixed(3)}</td>
              <td>{r.roc_auc.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h2>Model Features ({features.length})</h2>
      <div className="chip-list">
        {features.map((f) => <span className="chip" key={f}>{f}</span>)}
      </div>
    </div>
  );
}
