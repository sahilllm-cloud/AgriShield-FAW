
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import "./Register.css";

const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

function Register() {
  const navigate = useNavigate();

  const [form, setForm] = useState({
    name: "",
    email: "",
    password: "",
  });

  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  function handleChange(event) {
    const { name, value } = event.target;

    setForm((previous) => ({
      ...previous,
      [name]: value,
    }));

    setError("");
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");

    if (form.password.length < 8) {
      setError("Your password must contain at least 8 characters.");
      return;
    }

    if (form.password !== confirmPassword) {
      setError("Your passwords do not match.");
      return;
    }

    setLoading(true);

    try {
      const response = await fetch(API_URL + "/api/auth/register", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(form),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Registration failed. Please try again."
        );
      }

      // If the backend returns a token, sign in immediately.
      if (data.access_token) {
        localStorage.setItem("access_token", data.access_token);

        if (data.user) {
          localStorage.setItem("user", JSON.stringify(data.user));
        }

        navigate("/");
      } else {
        // Otherwise, continue to login.
        navigate("/login", {
          state: { registered: true },
        });
      }
    } catch (err) {
      setError(
        err.message === "Failed to fetch"
          ? "Cannot connect to AgriShield. Please check that the backend is running."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="agri-register">
      {/* LEFT BRAND PANEL */}

      <section className="agri-register-visual">
        <div className="agri-register-glow" />

        <Link to="/" className="agri-register-brand">
          <span className="agri-register-logo" aria-hidden="true">
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

        <div className="agri-register-message">
          <span className="agri-register-eyebrow">
            YOUR JOURNEY STARTS HERE
          </span>

          <h1>
            Vision Meets
            <br />
            <span>the Field.</span>
          </h1>

          <p>
            Create your account and explore intelligent maize
            leaf analysis and Fall Armyworm risk assessment.
          </p>

          <div className="agri-register-points">
            <div>
              <span>01</span>
              <p>
                <strong>Analyse leaf images</strong>
                <small>Explore AI-powered crop health predictions.</small>
              </p>
            </div>

            <div>
              <span>02</span>
              <p>
                <strong>Understand field risk</strong>
                <small>Bring crop and environmental signals together.</small>
              </p>
            </div>

            <div>
              <span>03</span>
              <p>
                <strong>Keep your scans organised</strong>
                <small>Review saved results through your history.</small>
              </p>
            </div>
          </div>
        </div>

        <div className="agri-register-footer">
          AGRISHIELD-FAW <span>•</span> INTELLIGENT CROP PROTECTION
        </div>
      </section>

      {/* RIGHT REGISTRATION PANEL */}

      <section className="agri-register-form-side">
        <div className="agri-register-mobile-brand">
          <span>✳</span> AgriShield
        </div>

        <div className="agri-register-form-wrap">
          <div className="agri-register-top">
            <span>CREATE YOUR ACCOUNT</span>

            <p>
              Already registered?{" "}
              <Link to="/login">Sign in</Link>
            </p>
          </div>

          <div className="agri-register-heading">
            <span>WELCOME TO AGRISHIELD</span>
            <h2>
              Grow with <span>insight.</span>
            </h2>
            <p>
              Set up your account to access your crop intelligence workspace.
            </p>
          </div>

          <form
            className="agri-register-form"
            onSubmit={handleSubmit}
          >
            {error && (
              <div className="agri-register-error" role="alert">
                <span>!</span>
                <p>{error}</p>
              </div>
            )}

            <div className="agri-register-field">
              <label htmlFor="register-name">Full name</label>
              <input
                id="register-name"
                type="text"
                name="name"
                value={form.name}
                onChange={handleChange}
                placeholder="Enter your full name"
                autoComplete="name"
                maxLength={100}
                required
              />
            </div>

            <div className="agri-register-field">
              <label htmlFor="register-email">Email address</label>
              <input
                id="register-email"
                type="email"
                name="email"
                value={form.email}
                onChange={handleChange}
                placeholder="you@example.com"
                autoComplete="email"
                maxLength={255}
                required
              />
            </div>

            <div className="agri-register-field">
              <div className="agri-register-label-row">
                <label htmlFor="register-password">Password</label>
                <span>Minimum 8 characters</span>
              </div>

              <div className="agri-register-password">
                <input
                  id="register-password"
                  type={showPassword ? "text" : "password"}
                  name="password"
                  value={form.password}
                  onChange={handleChange}
                  placeholder="Create a strong password"
                  autoComplete="new-password"
                  minLength={8}
                  required
                />

                <button
                  type="button"
                  onClick={() => setShowPassword((value) => !value)}
                  aria-label={showPassword ? "Hide password" : "Show password"}
                >
                  {showPassword ? "HIDE" : "SHOW"}
                </button>
              </div>
            </div>

            <div className="agri-register-field">
              <label htmlFor="confirm-password">Confirm password</label>
              <input
                id="confirm-password"
                type={showPassword ? "text" : "password"}
                value={confirmPassword}
                onChange={(event) => {
                  setConfirmPassword(event.target.value);
                  setError("");
                }}
                placeholder="Enter your password again"
                autoComplete="new-password"
                required
              />
            </div>

            <button
              className="agri-register-submit"
              type="submit"
              disabled={loading}
            >
              {loading ? "Creating your account..." : "Create my account"}
              {!loading && <span aria-hidden="true">→</span>}
            </button>

            <p className="agri-register-terms">
              By creating an account, you agree to use AgriShield
              responsibly as a crop decision-support tool.
            </p>
          </form>

          <div className="agri-register-bottom">
            <Link to="/prediction">Continue as guest</Link>
            <span>Secure account access</span>
          </div>
        </div>

        <div className="agri-register-copyright">
          © {new Date().getFullYear()} AgriShield AI
        </div>
      </section>
    </main>
  );
}

export default Register;
