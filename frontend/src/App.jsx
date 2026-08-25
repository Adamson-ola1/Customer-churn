import React from "react";
import { BrowserRouter } from "react-router-dom";
import { AppProvider } from "./context/AppContext";
import AppRoutes from "./routes";
import "./styles/dashboard.css";
import "./styles/forms.css";
import "./styles/table.css";

export default function App() {
  return (
    <BrowserRouter>
      <AppProvider>
        <AppRoutes />
      </AppProvider>
    </BrowserRouter>
  );
}
