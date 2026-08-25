import React, { useState } from "react";
import { predictChurn } from "../api/predictionApi";
import Loader from "./Loader";

const initialForm = {
  credit_score: 650,
  country: "France",
  gender: "Female",
  age: 40,
  tenure: 5,
  balance: 50000,
  products_number: 1,
  credit_card: 1,
  active_member: 1,
  estimated_salary: 100000,
};

export default function PredictionForm({ onResult }) {
  const [form, setForm] = useState(initialForm);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value, type } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]: type === "number" ? Number(value) : value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const result = await predictChurn(form);
      onResult(result, form);
    } catch (err) {
      setError(err.response?.data?.detail || "Prediction failed. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <form className="prediction-form" onSubmit={handleSubmit}>
      <div className="form-grid">
        <label>
          Credit Score
          <input type="number" name="credit_score" min="300" max="900" value={form.credit_score} onChange={handleChange} />
        </label>
        <label>
          Country
          <select name="country" value={form.country} onChange={handleChange}>
            <option>France</option>
            <option>Spain</option>
            <option>Germany</option>
          </select>
        </label>
        <label>
          Gender
          <select name="gender" value={form.gender} onChange={handleChange}>
            <option>Female</option>
            <option>Male</option>
          </select>
        </label>
        <label>
          Age
          <input type="number" name="age" min="18" max="100" value={form.age} onChange={handleChange} />
        </label>
        <label>
          Tenure (years)
          <input type="number" name="tenure" min="0" max="15" value={form.tenure} onChange={handleChange} />
        </label>
        <label>
          Balance
          <input type="number" name="balance" min="0" value={form.balance} onChange={handleChange} />
        </label>
        <label>
          Number of Products
          <input type="number" name="products_number" min="1" max="4" value={form.products_number} onChange={handleChange} />
        </label>
        <label>
          Estimated Salary
          <input type="number" name="estimated_salary" min="0" value={form.estimated_salary} onChange={handleChange} />
        </label>
        <label className="checkbox-label">
          <input type="checkbox" name="credit_card" checked={!!form.credit_card}
                 onChange={(e) => setForm((p) => ({ ...p, credit_card: e.target.checked ? 1 : 0 }))} />
          Has Credit Card
        </label>
        <label className="checkbox-label">
          <input type="checkbox" name="active_member" checked={!!form.active_member}
                 onChange={(e) => setForm((p) => ({ ...p, active_member: e.target.checked ? 1 : 0 }))} />
          Active Member
        </label>
      </div>

      {error && <p className="form-error">{error}</p>}
      <button type="submit" className="btn-primary" disabled={loading}>
        {loading ? <Loader label="Predicting..." /> : "Predict Churn Risk"}
      </button>
    </form>
  );
}
