
import { useNavigate } from "react-router-dom";

function Account() {
  const navigate = useNavigate();

  let email = "";

  try {
    const savedUser = JSON.parse(
      localStorage.getItem("user") || "{}"
    );

    email = savedUser?.email || savedUser?.user?.email || "";
  } catch {
    email = "";
  }

  return (
    <main className="account-page">
      <section className="account-card">
        <div className="account-avatar">
          {email ? email.charAt(0).toUpperCase() : "U"}
        </div>

        <h1>Account Information</h1>
        <p className="account-subtitle">
          Your AgriShield account details
        </p>

        <div className="account-detail-row">
          <span>Email Address</span>
          <strong>{email || "Email unavailable"}</strong>
        </div>

        <div className="account-detail-row">
          <span>Account Status</span>
          <strong className="account-status">Registered</strong>
        </div>

        <button
          type="button"
          className="account-back-button"
          onClick={() => navigate(-1)}
        >
          Back
        </button>
      </section>
    </main>
  );
}

export default Account;
