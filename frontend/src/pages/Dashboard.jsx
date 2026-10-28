

import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";



import RiskChart from "../components/RiskChart";

import RiskBadge from "../components/RiskBadge";



const API_URL =

  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";



function getWelcomeName() {
  try {
    const savedUser = localStorage.getItem("user");
    if (savedUser) {
      const user = JSON.parse(savedUser);
      const email = user?.email || user?.user?.email || "";
      if (typeof email === "string" && email.includes("@")) {
        return email.split("@")[0];
      }
    }
  } catch {
    // Fall back to the token if the saved user value is not JSON.
  }

  try {
    const token = localStorage.getItem("access_token");
    const payloadPart = token?.split(".")[1];
    if (payloadPart) {
      const base64 = payloadPart.replace(/-/g, "+").replace(/_/g, "/");
      const padded = base64.padEnd(
        base64.length + ((4 - (base64.length % 4)) % 4),
        "="
      );
      const payload = JSON.parse(atob(padded));
      const email =
        payload.email ||
        (typeof payload.sub === "string" && payload.sub.includes("@")
          ? payload.sub
          : "");
      if (email) return email.split("@")[0];
    }
  } catch {
    // Use a neutral greeting if the email is unavailable.
  }

  return "Farmer";
}

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



function Dashboard() {

  const navigate = useNavigate();
  const welcomeName = getWelcomeName();
  const [risk, setRisk] = useState("Low");



  const [history, setHistory] = useState([]);

  const [historyLoading, setHistoryLoading] = useState(true);

  const [historyError, setHistoryError] = useState("");



  const [weather, setWeather] = useState(() => {

    const savedTemperature = localStorage.getItem("Temperature_C");

    const savedHumidity = localStorage.getItem("Humidity_%");

    const savedRainfall = localStorage.getItem("Rainfall_mm");

    const savedWindSpeed = localStorage.getItem("Wind_Speed_kmph");



    return {

      location: "Bengaluru",

      temperature: savedTemperature ? Number(savedTemperature) : 28,

      humidity: savedHumidity ? Number(savedHumidity) : 75,

      rainfall: savedRainfall ? Number(savedRainfall) : 10,

      windSpeed: savedWindSpeed ? Number(savedWindSpeed) : 8,

    };

  });



  // Fetch predictions belonging to the current user or guest.

  const fetchDashboardHistory = async () => {

    setHistoryLoading(true);

    setHistoryError("");



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

        headers,

      });



      const data = await response.json();



      if (!response.ok) {

        throw new Error(

          data.detail || "Unable to load prediction history."

        );

      }



      setHistory(Array.isArray(data.history) ? data.history : []);

    } catch (error) {

      console.error("Dashboard history request failed:", error);

      setHistoryError(

        error.message || "Unable to load prediction history."

      );

    } finally {

      setHistoryLoading(false);

    }

  };



  // Preserve the existing live-weather feature.

  const fetchLiveWeather = async () => {

    const fallbackWeather = {

      location: "Bengaluru",

      temperature: 28,

      humidity: 75,

      rainfall: 10,

      windSpeed: 8,

    };



    try {

      const response = await fetch(`${API_URL}/api/weather/live`);



      if (!response.ok) {

        throw new Error("Live weather request failed");

      }



      const data = await response.json();



      const savedTemperature = localStorage.getItem("Temperature_C");

      const savedHumidity = localStorage.getItem("Humidity_%");

      const savedRainfall = localStorage.getItem("Rainfall_mm");

      const savedWindSpeed = localStorage.getItem("Wind_Speed_kmph");



      setWeather({

        location: data.location || "Bengaluru",

        temperature:

          data.temperature ??

          (savedTemperature

            ? Number(savedTemperature)

            : fallbackWeather.temperature),

        humidity:

          data.humidity ??

          (savedHumidity

            ? Number(savedHumidity)

            : fallbackWeather.humidity),

        rainfall:

          data.rainfall ??

          (savedRainfall

            ? Number(savedRainfall)

            : fallbackWeather.rainfall),

        windSpeed:

          data.wind_speed ??

          (savedWindSpeed

            ? Number(savedWindSpeed)

            : fallbackWeather.windSpeed),

      });

    } catch (error) {

      console.error("Live weather request failed:", error);



      const savedTemperature = localStorage.getItem("Temperature_C");

      const savedHumidity = localStorage.getItem("Humidity_%");

      const savedRainfall = localStorage.getItem("Rainfall_mm");

      const savedWindSpeed = localStorage.getItem("Wind_Speed_kmph");



      setWeather({

        location: "Bengaluru",

        temperature: savedTemperature

          ? Number(savedTemperature)

          : fallbackWeather.temperature,

        humidity: savedHumidity

          ? Number(savedHumidity)

          : fallbackWeather.humidity,

        rainfall: savedRainfall

          ? Number(savedRainfall)

          : fallbackWeather.rainfall,

        windSpeed: savedWindSpeed

          ? Number(savedWindSpeed)

          : fallbackWeather.windSpeed,

      });

    }

  };



  // Preserve the existing weather-based FAW risk assessment.

  const fetchRisk = async () => {

    const payload = {

      Month: 6,

      Temperature_C: 28,

      "Humidity_%": 75,

      Rainfall_mm: 10,

      "Soil_Moisture_%": 55,

      Wind_Speed_kmph: 8,

      Crop_Stage: "Vegetative",

      Previous_Pest_Count: 2,

      Days_Since_Last_Attack: 7,

    };



    try {

      const response = await fetch(

        `${API_URL}/api/predict/weather`,

        {

          method: "POST",

          headers: {

            "Content-Type": "application/json",

          },

          body: JSON.stringify(payload),

        }

      );



      const responseBody = await response.text();



      if (!response.ok) {

        console.error(

          "Weather risk validation response:",

          responseBody

        );

        return;

      }



      const data = JSON.parse(responseBody);



      if (

        typeof data.risk_level === "string" &&

        data.risk_level.trim()

      ) {

        setRisk(data.risk_level);

      }

    } catch (error) {

      console.error("Weather risk request failed:", error);

    }

  };



  useEffect(() => {

    fetchDashboardHistory();

    fetchRisk();

    fetchLiveWeather();



    const savedTemperature = localStorage.getItem("Temperature_C");

    const savedHumidity = localStorage.getItem("Humidity_%");

    const savedRainfall = localStorage.getItem("Rainfall_mm");

    const savedWindSpeed = localStorage.getItem("Wind_Speed_kmph");



    setWeather((currentWeather) => ({

      ...currentWeather,

      location: "Bengaluru",

      temperature: savedTemperature

        ? Number(savedTemperature)

        : currentWeather.temperature,

      humidity: savedHumidity

        ? Number(savedHumidity)

        : currentWeather.humidity,

      rainfall: savedRainfall

        ? Number(savedRainfall)

        : currentWeather.rainfall,

      windSpeed: savedWindSpeed

        ? Number(savedWindSpeed)

        : currentWeather.windSpeed,

    }));

  }, []);



  // Calculate statistics from the records returned by the history API.

  const healthyCount = history.filter((item) => {

    const result = String(item.result || "").toLowerCase().trim();

    return result === "healthy";

  }).length;



  const infectedCount = history.filter((item) => {

    const result = String(item.result || "").toLowerCase().trim();



    return (

      result === "infected" ||

      result === "faw detected" ||

      result === "faw-infected" ||

      result === "faw infected" ||

      result === "fall armyworm detected"

    );

  }).length;



  return (

    <div className="dashboard">

      {/* PERSONALIZED WELCOME SECTION */}
      {Boolean(localStorage.getItem("access_token")) && (
        <section className="dashboard-welcome">
          <div className="dashboard-welcome-content">
            <span className="dashboard-welcome-eyebrow">
              AGRISHIELD · CROP INTELLIGENCE
            </span>
            <h2>Welcome back, {welcomeName}! <span aria-hidden="true">🌱</span></h2>
            <p>
              Your crop intelligence dashboard is ready. Monitor field
              conditions and assess Fall Armyworm risk with AgriShield.
            </p>
          </div>
          <button
            type="button"
            className="dashboard-welcome-button"
            onClick={() => navigate("/prediction")}
          >
            Start New Prediction <span aria-hidden="true">→</span>
          </button>
        </section>
      )}

      {/* HERO SECTION */}

      <section className="dashboard-hero">

        <div className="hero-content">

          <span className="hero-tag">

            AI-POWERED SMART AGRICULTURE

          </span>



          <h1>

            Protect Your Maize

            <br />

            From Fall Armyworm

          </h1>



          <p>

            AgriShield combines image-based detection and

            environmental prediction to provide early FAW

            risk assessment for maize crops.

          </p>



          <div className="hero-status">

            <span className="status-dot"></span>

            Monitoring System Active

          </div>

        </div>



        <div className="hero-visual">

          <div className="crop-icon">🌽</div>



          <div className="scan-ring ring-one"></div>

          <div className="scan-ring ring-two"></div>

          <div className="scan-ring ring-three"></div>

        </div>

      </section>



      {/* DASHBOARD STATISTICS */}

      <section className="dashboard-cards">

        <div className="card">

          <h3>Total Scans</h3>

          <h2>

            {historyLoading ? "..." : history.length}

          </h2>

          <p>Images analyzed</p>

        </div>



        <div className="card">

          <h3>Healthy</h3>

          <h2>

            {historyLoading ? "..." : healthyCount}

          </h2>

          <p>Healthy maize leaves</p>

        </div>



        <div className="card">

          <h3>FAW Detected</h3>

          <h2>

            {historyLoading ? "..." : infectedCount}

          </h2>

          <p>Infected maize leaves</p>

        </div>



        <div className="card">

          <h3>Current Risk</h3>

          <h2>

            <RiskBadge level={risk} />

          </h2>

          <p>Weather-based outbreak risk</p>

        </div>

      </section>



      {historyError && (

        <p role="alert" className="history-error">

          Unable to load your statistics: {historyError}

        </p>

      )}



      {/* FAW RISK TREND */}

      <RiskChart

        history={history}

        isLoading={historyLoading}

      />



      {/* WEATHER CONDITIONS */}

      <section className="weather-section">

        <h2>Current Weather Conditions</h2>

        <p>{weather.location}</p>



        <div className="weather-cards">

          <div className="weather-card">

            <h3>Temperature</h3>

            <h2>{weather.temperature}°C</h2>

          </div>



          <div className="weather-card">

            <h3>Humidity</h3>

            <h2>{weather.humidity}%</h2>

          </div>



          <div className="weather-card">

            <h3>Rainfall</h3>

            <h2>{weather.rainfall} mm</h2>

          </div>



          <div className="weather-card">

            <h3>Wind Speed</h3>

            <h2>{weather.windSpeed} km/h</h2>

          </div>

        </div>

      </section>



      {/* AI MODEL PIPELINE */}

      <section className="ai-pipeline">

        <div className="section-heading">

          <span>INTELLIGENCE ENGINE</span>



          <h2>How AgriShield Predicts FAW</h2>



          <p>

            Multiple AI models work together to detect

            infection and predict outbreak risk.

          </p>

        </div>



        {/* Main Pipeline */}

        <div className="pipeline">

          <div className="pipeline-card">

            <div className="pipeline-icon">📷</div>



            <span className="pipeline-number">01</span>



            <h3>Maize Leaf Image</h3>



            <p>

              Upload a maize leaf image for automated visual

              analysis.

            </p>

          </div>



          <div className="pipeline-arrow">→</div>



          <div className="pipeline-card highlight">

            <div className="pipeline-icon">🧠</div>



            <span className="pipeline-number">02</span>



            <h3>Swin Transformer</h3>



            <p>

              Analyzes visual features to detect Fall Armyworm

              infection.

            </p>

          </div>



          <div className="pipeline-arrow">→</div>



          <div className="pipeline-card">

            <div className="pipeline-icon">🌦️</div>



            <span className="pipeline-number">03</span>



            <h3>Environmental Data</h3>



            <p>

              Weather and historical conditions are analyzed

              for outbreak prediction.

            </p>

          </div>

        </div>



        {/* Prediction Models */}

        <div className="prediction-pipeline">

          <div className="model-plus">+</div>



          <div className="prediction-model">

            <span>LightGBM</span>

            <small>Risk Classification</small>

          </div>



          <div className="model-arrow">→</div>



          <div className="final-risk">

            <span>FINAL OUTPUT</span>

            <strong>FAW RISK</strong>

            <small>Low / Medium / High</small>

          </div>

        </div>

      </section>

    </div>

  );

}



export default Dashboard;
