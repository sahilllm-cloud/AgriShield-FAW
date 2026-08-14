import { useEffect, useState } from "react";
import RiskChart from "../components/RiskChart";
import RiskBadge from "../components/RiskBadge";

function Dashboard() {
  const [risk, setRisk] = useState("Low");
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

  const fetchLiveWeather = async () => {
    const fallbackWeather = {
      location: "Bengaluru",
      temperature: 28,
      humidity: 75,
      rainfall: 10,
      windSpeed: 8,
    };

    try {
      const response = await fetch("http://127.0.0.1:8000/api/weather/live");
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
        temperature: data.temperature ?? (savedTemperature ? Number(savedTemperature) : fallbackWeather.temperature),
        humidity: data.humidity ?? (savedHumidity ? Number(savedHumidity) : fallbackWeather.humidity),
        rainfall: data.rainfall ?? (savedRainfall ? Number(savedRainfall) : fallbackWeather.rainfall),
        windSpeed: data.wind_speed ?? (savedWindSpeed ? Number(savedWindSpeed) : fallbackWeather.windSpeed),
      });
    } catch (error) {
      const savedTemperature = localStorage.getItem("Temperature_C");
      const savedHumidity = localStorage.getItem("Humidity_%");
      const savedRainfall = localStorage.getItem("Rainfall_mm");
      const savedWindSpeed = localStorage.getItem("Wind_Speed_kmph");

      setWeather({
        location: "Bengaluru",
        temperature: savedTemperature ? Number(savedTemperature) : fallbackWeather.temperature,
        humidity: savedHumidity ? Number(savedHumidity) : fallbackWeather.humidity,
        rainfall: savedRainfall ? Number(savedRainfall) : fallbackWeather.rainfall,
        windSpeed: savedWindSpeed ? Number(savedWindSpeed) : fallbackWeather.windSpeed,
      });
    }
  };

  const fetchRisk = async () => {
    const response = await fetch("http://127.0.0.1:8000/api/predict/weather", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        Month: 6,
        Temperature_C: 28,
        "Humidity_%": 75,
        Rainfall_mm: 10,
        "Soil_Moisture_%": 55,
        Wind_Speed_kmph: 8,
        Crop_Stage: "vegetative",
        Previous_Pest_Count: 2,
        Days_Since_Last_Attack: 7,
      }),
    });
    const data = await response.json();
    setRisk(data.risk_level);
  };

  useEffect(() => {
    fetchRisk();
    fetchLiveWeather();

    const savedTemperature = localStorage.getItem("Temperature_C");
    const savedHumidity = localStorage.getItem("Humidity_%");
    const savedRainfall = localStorage.getItem("Rainfall_mm");
    const savedWindSpeed = localStorage.getItem("Wind_Speed_kmph");

    setWeather((currentWeather) => ({
      ...currentWeather,
      location: "Bengaluru",
      temperature: savedTemperature ? Number(savedTemperature) : currentWeather.temperature,
      humidity: savedHumidity ? Number(savedHumidity) : currentWeather.humidity,
      rainfall: savedRainfall ? Number(savedRainfall) : currentWeather.rainfall,
      windSpeed: savedWindSpeed ? Number(savedWindSpeed) : currentWeather.windSpeed,
    }));
  }, []);

  return (
    <div className="dashboard">

      {/* =====================================================
          HERO SECTION
          ===================================================== */}

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

          <div className="crop-icon">
            🌽
          </div>

          <div className="scan-ring ring-one"></div>
          <div className="scan-ring ring-two"></div>
          <div className="scan-ring ring-three"></div>

        </div>

      </section>


      {/* =====================================================
          DASHBOARD STATISTICS
          ===================================================== */}

      <section className="dashboard-cards">

        <div className="card">
          <h3>Total Scans</h3>
          <h2>0</h2>
          <p>Images analyzed</p>
        </div>

        <div className="card">
          <h3>Healthy</h3>
          <h2>0</h2>
          <p>Healthy maize leaves</p>
        </div>

        <div className="card">
          <h3>FAW Detected</h3>
          <h2>0</h2>
          <p>Infected maize leaves</p>
        </div>

        <div className="card">
          <h3>Current Risk</h3>

          <h2>
            <RiskBadge level={risk} />
          </h2>

          <p>Outbreak risk level</p>
        </div>

      </section>


      {/* =====================================================
          FAW RISK TREND
          ===================================================== */}

      <RiskChart />


      {/* =====================================================
          WEATHER CONDITIONS
          ===================================================== */}

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


      {/* =====================================================
          AI MODEL PIPELINE
          ===================================================== */}

      <section className="ai-pipeline">

        <div className="section-heading">

          <span>
            INTELLIGENCE ENGINE
          </span>

          <h2>
            How AgriShield Predicts FAW
          </h2>

          <p>
            Multiple AI models work together to detect
            infection and predict outbreak risk.
          </p>

        </div>


        {/* Main Pipeline */}

        <div className="pipeline">

          {/* Step 01 */}

          <div className="pipeline-card">

            <div className="pipeline-icon">
              📷
            </div>

            <span className="pipeline-number">
              01
            </span>

            <h3>
              Maize Leaf Image
            </h3>

            <p>
              Upload a maize leaf image for
              automated visual analysis.
            </p>

          </div>


          <div className="pipeline-arrow">
            →
          </div>


          {/* Step 02 */}

          <div className="pipeline-card highlight">

            <div className="pipeline-icon">
              🧠
            </div>

            <span className="pipeline-number">
              02
            </span>

            <h3>
              Swin Transformer
            </h3>

            <p>
              Analyzes visual features to detect
              Fall Armyworm infection.
            </p>

          </div>


          <div className="pipeline-arrow">
            →
          </div>


          {/* Step 03 */}

          <div className="pipeline-card">

            <div className="pipeline-icon">
              🌦️
            </div>

            <span className="pipeline-number">
              03
            </span>

            <h3>
              Environmental Data
            </h3>

            <p>
              Weather and historical conditions
              are analyzed for outbreak prediction.
            </p>

          </div>

        </div>


        {/* Prediction Models */}

        <div className="prediction-pipeline">

          <div className="model-plus">
            +
          </div>


          <div className="prediction-model">

            <span>
              LightGBM
            </span>

            <small>
              Risk Classification
            </small>

          </div>


          <div className="model-arrow">
            →
          </div>


          <div className="final-risk">

            <span>
              FINAL OUTPUT
            </span>

            <strong>
              FAW RISK
            </strong>

            <small>
              Low / Medium / High
            </small>

          </div>

        </div>

      </section>

    </div>
  );
}

export default Dashboard;