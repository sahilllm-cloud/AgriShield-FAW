import { NavLink } from "react-router-dom";

function Navbar() {
  return (
    <nav className="navbar">

      <NavLink to="/" className="navbar-brand">
        🌽 AgriShield
      </NavLink>

      <div className="navbar-links">

        <NavLink
          to="/"
          end
          className={({ isActive }) =>
            isActive ? "active" : ""
          }
        >
          Dashboard
        </NavLink>

        <NavLink
          to="/prediction"
          className={({ isActive }) =>
            isActive ? "active" : ""
          }
        >
          Prediction
        </NavLink>

        <NavLink
          to="/history"
          className={({ isActive }) =>
            isActive ? "active" : ""
          }
        >
          History
        </NavLink>

      </div>

    </nav>
  );
}

export default Navbar;