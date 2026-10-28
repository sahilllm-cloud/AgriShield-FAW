
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import "./Login.css";

const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

function Login() {
  const navigate = useNavigate();

  const [form, setForm] = useState({
    email: "",
    password: "",
  });

  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  function handleChange(event) {
    const name = event.target.name;
    const value = event.target.value;

    setForm(function (previous) {
      return { ...previous, [name]: value };
    });

    setError("");
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setLoading(true);

    try {
      const response = await fetch(
        API_URL + "/api/auth/login",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(form),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Login failed.");
      }

      localStorage.setItem("access_token", data.access_token);
      localStorage.setItem("user", JSON.stringify(data.user));
      window.dispatchEvent(new Event("auth-changed"));

      navigate("/");
    } catch (err) {
      setError(
        err.message === "Failed to fetch"
          ? "Cannot connect to the backend. Check that it is running."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="agri-login">
      <section className="agri-login-visual">
        <div className="agri-login-glow agri-login-glow-one" />
        <div className="agri-login-glow agri-login-glow-two" />

        <Link to="/" className="agri-login-brand">
          <span className="agri-brand-symbol">
            <svg viewBox="0 0 48 48" fill="none">
              <path
                d="M24 40V18"
                stroke="currentColor"
                strokeWidth="3"
                strokeLinecap="round"
              />
              <path
                d="M24 27C10 27 7 18 8 9C19 9 28 14 24 27Z"
                fill="#B7F27B"
              />
              <path
                d="M24 34C37 34 42 25 40 16C29 16 21 22 24 34Z"
                fill="#78D79B"
              />
            </svg>
          </span>
          <span>
            <strong>AgriShield</strong>
            <small>INTELLIGENCE FOR AGRICULTURE</small>
          </span>
        </Link>

        <div className="agri-visual-content">
          <span className="agri-eyebrow">
            <span className="agri-eyebrow-dot" />
            SMART FARMING. SMARTER FUTURES.
          </span>

          <h1>
           Vision Meets
           <br />
            <span>the Field.</span>
          </h1>

          <p className="agri-visual-description">
            Turn maize leaf images and environmental signals into
            meaningful insights for Fall Armyworm risk assessment.
          </p>

          <div className="agri-feature-list">
            <div className="agri-feature">
              <span className="agri-feature-icon">◎</span>
              <span>
                <strong>AI-powered leaf analysis</strong>
                <small>Identify potential crop health issues</small>
              </span>
            </div>

            <div className="agri-feature">
              <span className="agri-feature-icon">↗</span>
              <span>
                <strong>Predictive field intelligence</strong>
                <small>Combine crop and environmental signals</small>
              </span>
            </div>

            <div className="agri-feature">
              <span className="agri-feature-icon">✓</span>
              <span>
                <strong>Your scans, in one place</strong>
                <small>Access your saved prediction history</small>
              </span>
            </div>
          </div>
        </div>

        <div className="agri-visual-footer">
          <span>BUILT FOR BETTER FARMING DECISIONS</span>
          <span>AGRISHIELD AI</span>
        </div>
      </section>

      <section className="agri-login-form-side">
        <div className="agri-mobile-brand">✳ AgriShield</div>

        <div className="agri-login-form-wrap">
          <div className="agri-login-topline">
            <span className="agri-login-status">
              <span />
              SECURE ACCESS
            </span>

            <Link to="/register" className="agri-top-register">
              New here? <strong>Create account</strong>
            </Link>
          </div>

          <div className="agri-login-heading">
            <span className="agri-form-eyebrow">
              YOUR FARM. YOUR INTELLIGENCE.
            </span>

            <h2>
              Welcome back<span>.</span>
            </h2>

            <p>
              Sign in to continue monitoring your field and
              reviewing your crop insights.
            </p>
          </div>

          <form className="agri-login-form" onSubmit={handleSubmit}>
            {error && (
              <div className="agri-login-error" role="alert">
                <span>!</span>
                <p>{error}</p>
              </div>
            )}

            <div className="agri-field">
              <label htmlFor="email">Email address</label>

              <div className="agri-input-wrap">
                <span aria-hidden="true">✉</span>
                <input
                  id="email"
                  name="email"
                  type="email"
                  value={form.email}
                  onChange={handleChange}
                  placeholder="you@example.com"
                  autoComplete="email"
                  required
                />
              </div>
            </div>

            <div className="agri-field">
              <div className="agri-label-row">
                <label htmlFor="password">Password</label>
                <span>Keep your account secure</span>
              </div>

              <div className="agri-input-wrap">
                <span aria-hidden="true">♙</span>
                <input
                  id="password"
                  name="password"
                  type={showPassword ? "text" : "password"}
                  value={form.password}
                  onChange={handleChange}
                  placeholder="Enter your password"
                  autoComplete="current-password"
                  required
                />

                <button
                  type="button"
                  className="agri-password-toggle"
                  onClick={function () {
                    setShowPassword(function (value) {
                      return !value;
                    });
                  }}
                >
                  {showPassword ? "HIDE" : "SHOW"}
                </button>
              </div>
            </div>

            <button
              className="agri-login-submit"
              type="submit"
              disabled={loading}
            >
              {loading ? "Signing you in..." : "Sign in to AgriShield"}
              {!loading && <span aria-hidden="true"> →</span>}
            </button>

            <div className="agri-login-divider">
              <span />
              <small>OR EXPLORE FIRST</small>
              <span />
            </div>

            <Link to="/prediction" className="agri-guest-button">
              Continue as guest <span>↗</span>
            </Link>
          </form>

          <div className="agri-login-bottom">
            <p>
              Don't have an account?{" "}
              <Link to="/register">Create one here</Link>
            </p>
            <span className="agri-login-privacy">
              ✓ Secure account access
            </span>
          </div>
        </div>

        <div className="agri-form-copyright">
          AgriShield AI · Smarter agriculture, together.
        </div>
      </section>
    </main>
  );
}

export default Login;
