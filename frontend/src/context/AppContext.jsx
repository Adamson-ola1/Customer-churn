import React, { createContext, useContext, useState } from "react";

const AppContext = createContext(null);

export function AppProvider({ children }) {
  const [lastPrediction, setLastPrediction] = useState(null);
  const [history, setHistory] = useState([]);

  const addPrediction = (result, input) => {
    setLastPrediction(result);
    setHistory((prev) => [{ ...input, ...result, ts: Date.now() }, ...prev].slice(0, 20));
  };

  return (
    <AppContext.Provider value={{ lastPrediction, history, addPrediction }}>
      {children}
    </AppContext.Provider>
  );
}

export const useAppContext = () => {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useAppContext must be used within AppProvider");
  return ctx;
};
