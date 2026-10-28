
import { useCallback, useEffect, useState } from "react";

const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

function getGuestId() {
  let guestId = localStorage.getItem("agrishield_guest_id");

  if (!guestId) {
    guestId =
      typeof crypto !== "undefined" && crypto.randomUUID
        ? crypto.randomUUID()
        : `guest-${Date.now()}-${Math.random().toString(36).slice(2)}`;

    localStorage.setItem("agrishield_guest_id", guestId);
  }

  return guestId;
}

function formatDate(dateValue) {
  if (!dateValue) return "Date unavailable";

  const date = new Date(dateValue);

  if (Number.isNaN(date.getTime())) return dateValue;

  return date.toLocaleDateString("en-IN", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });
}

function formatProbability(value) {
  if (value === null || value === undefined || value === "") {
    return "-";
  }

  const number = Number(value);

  return Number.isFinite(number) ? `${number.toFixed(1)}%` : "-";
}

function normalizeRisk(value) {
  const risk = String(value || "Unknown").trim().toUpperCase();

  if (risk === "HIGH") return "High";
  if (risk === "MEDIUM") return "Medium";
  if (risk === "LOW") return "Low";

  return risk.charAt(0) + risk.slice(1).toLowerCase();
}

function History() {
  const [historyData, setHistoryData] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");
  const [expandedId, setExpandedId] = useState(null);

  const fetchHistory = useCallback(async () => {
    setIsLoading(true);
    setError("");

    try {
      const token = localStorage.getItem("access_token");
      const guestId = getGuestId();

      const headers = {
        "X-Guest-ID": guestId,
      };

      if (token) {
        headers.Authorization = `Bearer ${token}`;
      }

      const response = await fetch(`${API_URL}/api/history`, {
        method: "GET",
        headers,
      });

      const responseText = await response.text();

      let data;

      try {
        data = JSON.parse(responseText);
      } catch {
        throw new Error("The server returned an invalid response.");
      }

      if (!response.ok) {
        throw new Error(
          data.detail || `Unable to load history (${response.status}).`
        );
      }

      setHistoryData(
        Array.isArray(data.history) ? data.history : []
      );
    } catch (err) {
      console.error("History loading failed:", err);
      setError(
        err.message ||
          "Unable to load prediction history. Check that the backend is running."
      );
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  const detectionCount = historyData.filter((item) => {
    const result = String(item.result || "").toLowerCase();

    return (
      result === "infected" ||
      result.includes("detected") ||
      result.includes("disease")
    );
  }).length;

  const highRiskCount = historyData.filter(
    (item) => normalizeRisk(item.risk) === "High"
  ).length;

  const mediumRiskCount = historyData.filter(
    (item) => normalizeRisk(item.risk) === "Medium"
  ).length;

  return (
    <div className="history-page">
      {/* HEADER */}

      <section className="history-header">
        <span className="history-tag">
          AGRISHIELD ACTIVITY LOG
        </span>

        <h1>
          Prediction
          <br />
          History
        </h1>

        <p>
          Review previous Fall Armyworm detection and
          outbreak prediction results.
        </p>
      </section>

      {/* SUMMARY STRIP */}

      <section className="history-summary">
        <div>
          <span>TOTAL ANALYSES</span>
          <strong>
            {isLoading ? "…" : historyData.length}
          </strong>
        </div>

        <div>
          <span>INFECTED RESULTS</span>
          <strong>
            {isLoading ? "…" : detectionCount}
          </strong>
        </div>

        <div>
          <span>HIGH RISK EVENTS</span>
          <strong>
            {isLoading ? "…" : highRiskCount}
          </strong>
        </div>

        <div className="history-live">
          <span className="history-live-dot"></span>
          {isLoading ? "LOADING" : "LIVE RECORD"}
        </div>
      </section>

      {/* HISTORY PANEL */}

      <section className="history-panel">
        <div className="history-panel-header">
          <div>
            <span>FIELD INTELLIGENCE</span>
            <h2>Recent Analysis</h2>

            <p>
              {historyData.length} saved scans ·{" "}
              {detectionCount} infected ·{" "}
              {historyData.length - detectionCount} healthy or other
            </p>
          </div>

          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "12px",
              flexWrap: "wrap",
            }}
          >
            <div className="history-count">
              {historyData.length} RECORDS
            </div>

            <button
              type="button"
              className="predict-button"
              onClick={fetchHistory}
              disabled={isLoading}
            >
              {isLoading ? "Loading..." : "Refresh"}
            </button>
          </div>
        </div>

        {error && (
          <div
            role="alert"
            style={{
              margin: "16px",
              padding: "14px",
              borderRadius: "8px",
              background: "#fef2f2",
              color: "#b42318",
            }}
          >
            <strong>Could not load history</strong>
            <p>{error}</p>
            <p>
              Check that you are logged in to the correct account
              and that FastAPI is running.
            </p>
          </div>
        )}

        {isLoading && (
          <p style={{ padding: "24px", textAlign: "center" }}>
            Loading your saved prediction history...
          </p>
        )}

        {!isLoading && !error && historyData.length === 0 && (
          <div
            style={{
              padding: "36px 20px",
              textAlign: "center",
            }}
          >
            <h3>No scans found</h3>
            <p>
              Perform a prediction while logged in, then refresh
              this page to see the saved result.
            </p>
          </div>
        )}

        {/* TABLE */}

        {!isLoading && !error && historyData.length > 0 && (
          <div className="history-table">
            <table>
              <thead>
                <tr>
                  <th>DATE</th>
                  <th>ANALYSIS TYPE</th>
                  <th>RESULT</th>
                  <th>CONFIDENCE</th>
                  <th>FAW PROBABILITY</th>
                  <th>RISK</th>
                  <th>DETAILS</th>
                </tr>
              </thead>

              <tbody>
                {historyData.map((item, index) => {
                  const details = item.details || {};
                  const crop = details.crop || {};
                  const weather =
                    details.selected_date_weather ||
                    details.current_weather ||
                    details.weather ||
                    {};

                  const risk = normalizeRisk(item.risk);
                  const isExpanded = expandedId === item.id;

                  const result = String(
                    item.result || "Unknown"
                  );

                  const resultLabel =
                    result.toLowerCase() === "infected"
                      ? "FAW Detected"
                      : result.toLowerCase() === "healthy"
                        ? "Healthy"
                        : result;

                  const confidence = formatProbability(
                    item.confidence
                  );

                  return (
                    <>
                      <tr key={item.id ?? index}>
                        <td>
                          <div className="history-date">
                            <span className="history-date-icon">
                              ◉
                            </span>

                            <div>
                              <strong>
                                {formatDate(
                                  item.created_at || item.date
                                )}
                              </strong>

                              <small>
                                Analysis #
                                {String(index + 1).padStart(2, "0")}
                              </small>
                            </div>
                          </div>
                        </td>

                        <td>
                          <span className="history-type detection-type">
                            {item.type || "Hybrid FAW Prediction"}
                          </span>
                        </td>

                        <td>
                          <strong className="history-result">
                            {resultLabel}
                          </strong>

                          <small
                            style={{
                              display: "block",
                              marginTop: "5px",
                              overflowWrap: "anywhere",
                            }}
                          >
                            {item.filename || "Image unavailable"}
                          </small>
                        </td>

                        <td>
                          <span className="history-confidence">
                            {confidence}
                          </span>
                        </td>

                        <td>
                          <strong>
                            {formatProbability(
                              item.faw_attack_probability
                            )}
                          </strong>
                        </td>

                        <td>
                          <span className="history-risk">
                            <span></span>
                            {risk}
                          </span>
                        </td>

                        <td>
                          <button
                            type="button"
                            className="predict-button"
                            onClick={() =>
                              setExpandedId(
                                isExpanded ? null : item.id
                              )
                            }
                          >
                            {isExpanded ? "Hide" : "View"}
                          </button>
                        </td>
                      </tr>

                      {isExpanded && (
                        <tr key={`${item.id}-details`}>
                          <td colSpan={7}>
                            <div
                              style={{
                                padding: "16px",
                                borderRadius: "10px",
                                background: "rgba(233, 247, 237, 0.55)",
                                display: "grid",
                                gridTemplateColumns:
                                  "repeat(auto-fit, minmax(180px, 1fr))",
                                gap: "16px",
                              }}
                            >
                              <div>
                                <strong>Scan information</strong>
                                <p>
                                  Record ID: {item.id}
                                </p>
                                <p>
                                  Scan date:{" "}
                                  {formatDate(
                                    item.created_at || item.date
                                  )}
                                </p>
                                <p>
                                  Image: {item.filename || "-"}
                                </p>
                                <p>
                                  Result: {resultLabel}
                                </p>
                              </div>

                              <div>
                                <strong>Prediction details</strong>
                                <p>
                                  Healthy probability:{" "}
                                  {formatProbability(
                                    item.healthy_probability
                                  )}
                                </p>
                                <p>
                                  Infected probability:{" "}
                                  {formatProbability(
                                    item.infected_probability
                                  )}
                                </p>
                                <p>
                                  FAW attack probability:{" "}
                                  {formatProbability(
                                    item.faw_attack_probability
                                  )}
                                </p>
                                <p>Risk level: {risk}</p>
                              </div>

                              <div>
                                <strong>Crop information</strong>
                                <p>
                                  Stage: {crop.stage || "-"}
                                </p>
                                <p>
                                  Crop age:{" "}
                                  {crop.age_days != null
                                    ? `${crop.age_days} days`
                                    : "-"}
                                </p>
                                <p>
                                  Variety: {crop.maize_variety || "-"}
                                </p>
                                <p>
                                  Date assessed: {item.date || "-"}
                                </p>
                              </div>

                              <div>
                                <strong>Weather information</strong>
                                {Object.keys(weather).length > 0 ? (
                                  Object.entries(weather).map(
                                    ([key, value]) => (
                                      <p key={key}>
                                        {key.replaceAll("_", " ")}:{" "}
                                        {value ?? "-"}
                                      </p>
                                    )
                                  )
                                ) : (
                                  <p>No weather details available.</p>
                                )}
                              </div>

                              {details.location && (
                                <div>
                                  <strong>Field location</strong>
                                  <p>
                                    Latitude:{" "}
                                    {details.location.latitude ??
                                      details.latitude ??
                                      "-"}
                                  </p>
                                  <p>
                                    Longitude:{" "}
                                    {details.location.longitude ??
                                      details.longitude ??
                                      "-"}
                                  </p>
                                </div>
                              )}
                            </div>
                          </td>
                        </tr>
                      )}
                    </>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {!isLoading && !error && historyData.length > 0 && (
          <div
            style={{
              padding: "14px 18px",
              color: "var(--muted)",
              fontSize: "0.85rem",
            }}
          >
            Medium-risk scans: {mediumRiskCount}. Records are loaded
            from the authenticated History API.
          </div>
        )}
      </section>
    </div>
  );
}

export default History;
