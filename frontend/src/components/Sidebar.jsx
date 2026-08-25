import React from "react";
import { NavLink } from "react-router-dom";

const links = [
  { to: "/", label: "Dashboard", icon: "📊" },
  { to: "/model-info", label: "Model Info", icon: "🧠" },
];

export default function Sidebar() {
  return (
    <nav className="sidebar">
      {links.map((link) => (
        <NavLink
          key={link.to}
          to={link.to}
          end={link.to === "/"}
          className={({ isActive }) => `sidebar-link${isActive ? " active" : ""}`}
        >
          <span className="sidebar-icon">{link.icon}</span>
          {link.label}
        </NavLink>
      ))}
    </nav>
  );
}
