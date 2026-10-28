
import { useEffect, useRef, useState } from "react";
import { NavLink, useNavigate } from "react-router-dom";

function getRegisteredEmail() {
  try {
    const savedUser = localStorage.getItem("user");

    if (savedUser) {
      const user = JSON.parse(savedUser);
      const email = user?.email || user?.user?.email;

      if (typeof email === "string" && email.includes("@")) {
        return email;
      }
    }
  } catch {
    // Try reading the email from the token.
  }

  try {
    const token = localStorage.getItem("access_token");
    const part = token?.split(".")[1];

    if (part) {
      const base64 = part
        .replace(/-/g, "+")
        .replace(/_/g, "/");

      const padded = base64.padEnd(
        base64.length + ((4 - (base64.length % 4)) % 4),
        "="
      );

      const payload = JSON.parse(atob(padded));
      const email =
        payload.email ||
        (typeof payload.sub === "string" &&
        payload.sub.includes("@")
          ? payload.sub
          : "");

      if (typeof email === "string" && email.includes("@")) {
        return email;
      }
    }
  } catch {
    // Email may not be available in the token.
  }

  return "";
}

function Navbar() {
  const navigate = useNavigate();
  const profileRef = useRef(null);

  const [isLoggedIn, setIsLoggedIn] = useState(
    Boolean(localStorage.getItem("access_token"))
  );

  const [email, setEmail] = useState(getRegisteredEmail);
  const [profileOpen, setProfileOpen] = useState(false);

  useEffect(() => {
    const updateAuth = () => {
      const loggedIn = Boolean(
        localStorage.getItem("access_token")
      );

      setIsLoggedIn(loggedIn);
      setEmail(loggedIn ? getRegisteredEmail() : "");
      setProfileOpen(false);
    };

    window.addEventListener("auth-changed", updateAuth);
    window.addEventListener("storage", updateAuth);
    window.addEventListener("focus", updateAuth);

    return () => {
      window.removeEventListener("auth-changed", updateAuth);
      window.removeEventListener("storage", updateAuth);
      window.removeEventListener("focus", updateAuth);
    };
  }, []);

  useEffect(() => {
    if (!profileOpen) return;

    const handleOutsideClick = (event) => {
      if (
        profileRef.current &&
        !profileRef.current.contains(event.target)
      ) {
        setProfileOpen(false);
      }
    };

    const handleEscape = (event) => {
      if (event.key === "Escape") {
        setProfileOpen(false);
      }
    };

    document.addEventListener("mousedown", handleOutsideClick);
    document.addEventListener("keydown", handleEscape);

    return () => {
      document.removeEventListener(
        "mousedown",
        handleOutsideClick
      );
      document.removeEventListener("keydown", handleEscape);
    };
  }, [profileOpen]);

  function handleLogout() {
    localStorage.removeItem("access_token");
    localStorage.removeItem("user");

    setProfileOpen(false);
    setIsLoggedIn(false);
    setEmail("");

    window.dispatchEvent(new Event("auth-changed"));
    navigate("/login");
  }

  function openAccount() {
    setProfileOpen(false);
    navigate("/account");
  }

  const avatarLetter = email
    ? email.charAt(0).toUpperCase()
    : "U";

  // Use the email as the display name.
  const welcomeName = email
    ? email.split("@")[0]
    : "User";

  return (
    <nav className="navbar">
      <NavLink to="/" className="navbar-brand">
        🌽 AgriShield
      </NavLink>

      <div className="navbar-links">
        {!isLoggedIn ? (
          <>
            <NavLink
              to="/login"
              className={({ isActive }) =>
                isActive ? "active" : ""
              }
            >
              Login
            </NavLink>

            <NavLink
              to="/register"
              className={({ isActive }) =>
                isActive ? "active" : ""
              }
            >
              Register
            </NavLink>
          </>
        ) : (
          <>
            
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

            <button
              type="button"
              className="logout-btn"
              onClick={handleLogout}
              aria-label="Logout"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="17"
                height="17"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <path d="M10 17l5-5-5-5" />
                <path d="M15 12H3" />
                <path d="M12 3h6a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-6" />
              </svg>
              <span>Logout</span>
            </button>

            <div className="profile-menu" ref={profileRef}>
              <button
                type="button"
                className="profile-trigger"
                aria-label="Open profile menu"
                aria-expanded={profileOpen}
                onClick={() =>
                  setProfileOpen((previous) => !previous)
                }
              >
                <span className="profile-avatar">
                  {avatarLetter}
                </span>

                <span className="profile-trigger-details">
                  <span className="profile-trigger-email">
                    {email || "My Account"}
                  </span>
                  <span className="profile-trigger-label">
                    Registered User
                  </span>
                </span>

                <svg
                  className={`profile-chevron ${
                    profileOpen ? "is-open" : ""
                  }`}
                  xmlns="http://www.w3.org/2000/svg"
                  width="16"
                  height="16"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <path d="m6 9 6 6 6-6" />
                </svg>
              </button>

              {profileOpen && (
                <div className="profile-dropdown">
                  <div className="profile-dropdown-header">
                    <span className="profile-avatar profile-avatar-large">
                      {avatarLetter}
                    </span>

                    <div className="profile-account-details">
                      <strong>My Account</strong>
                      <span>
                        {email || "Email unavailable"}
                      </span>
                      <small>Registered Account</small>
                    </div>
                  </div>

                  <div className="profile-dropdown-divider" />

                  <button
                    type="button"
                    className="profile-info-item profile-info-button"
                    onClick={openAccount}
                  >
                    <span className="profile-menu-icon">
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="19"
                        height="19"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="1.8"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        aria-hidden="true"
                      >
                        <rect
                          x="3"
                          y="5"
                          width="18"
                          height="14"
                          rx="2"
                        />
                        <path d="m3 7 9 6 9-6" />
                      </svg>
                    </span>

                    <span className="profile-info-text">
                      <strong>Account Information</strong>
                      <span>
                        {email || "Email unavailable"}
                      </span>
                    </span>

                    <span className="profile-link-arrow">›</span>
                  </button>

                  <NavLink
                    to="/history"
                    className="profile-dropdown-link"
                    onClick={() => setProfileOpen(false)}
                  >
                    <span className="profile-menu-icon">
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="19"
                        height="19"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="1.8"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        aria-hidden="true"
                      >
                        <path d="M3 12a9 9 0 1 0 3-6.7L3 8" />
                        <path d="M3 3v5h5" />
                        <path d="M12 7v5l3 2" />
                      </svg>
                    </span>

                    <span className="profile-link-text">
                      <strong>My Prediction History</strong>
                      <small>View your saved scans</small>
                    </span>

                    <span className="profile-link-arrow">›</span>
                  </NavLink>

                  <div className="profile-dropdown-divider" />

                  <button
                    type="button"
                    className="profile-signout"
                    onClick={handleLogout}
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="18"
                      height="18"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      aria-hidden="true"
                    >
                      <path d="M10 17l5-5-5-5" />
                      <path d="M15 12H3" />
                      <path d="M12 3h6a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-6" />
                    </svg>
                    Sign out
                  </button>
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </nav>
  );
}

export default Navbar;
