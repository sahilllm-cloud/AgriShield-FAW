import { useState } from "react";

function Prediction() {
  const [month, setMonth] = useState("6");
  const [temperature, setTemperature] = useState("");
  const [humidity, setHumidity] = useState("");
  const [rainfall, setRainfall] = useState("");
  const [soilMoisture, setSoilMoisture] = useState("");
  const [windSpeed, setWindSpeed] = useState("");
  const [cropStage, setCropStage] = useState("vegetative");
  const [previousPestCount, setPreviousPestCount] = useState("2");
  const [daysSinceLastAttack, setDaysSinceLastAttack] = useState("7");

  const [prediction, setPrediction] = useState(null);

  const handleUseLiveWeather = async () => {
    try {
      const response = await fetch("http://127.0.0.1:8000/api/weather/live");

      if (!response.ok) {
        throw new Error("Live weather request failed");
      }

      const data = await response.json();
      const liveSoilMoisture = Number(data.soil_moisture ?? data.soil_moisture_0_1cm ?? 0);
      const soilMoisturePercent = Number.isFinite(liveSoilMoisture)
        ? Number((liveSoilMoisture * 100).toFixed(1))
        : "";

      setMonth(String(new Date().getMonth() + 1));
      setTemperature(String(data.temperature ?? ""));
      setHumidity(String(data.humidity ?? ""));
      setRainfall(String(data.rainfall ?? ""));
      setSoilMoisture(String(soilMoisturePercent));
      setWindSpeed(String(data.wind_speed ?? ""));
    } catch (error) {
      console.error("Live weather error:", error);
    }
  };

  const handlePrediction = async () => {
    const payload = {
      Month: Number(month),
      Temperature_C: Number(temperature),
      "Humidity_%": Number(humidity),
      Rainfall_mm: Number(rainfall),
      "Soil_Moisture_%": Number(soilMoisture),
      Wind_Speed_kmph: Number(windSpeed),
      Crop_Stage: cropStage,
      Previous_Pest_Count: Number(previousPestCount),
      Days_Since_Last_Attack: Number(daysSinceLastAttack),
    };

    try {
      const response = await fetch("http://127.0.0.1:8000/api/predict/weather", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error("Risk prediction request failed");
      }

      const data = await response.json();

      localStorage.setItem("Month", String(payload.Month));
      localStorage.setItem("Temperature_C", String(payload.Temperature_C));
      localStorage.setItem("Humidity_%", String(payload["Humidity_%"]));
      localStorage.setItem("Rainfall_mm", String(payload.Rainfall_mm));
      localStorage.setItem("Soil_Moisture_%", String(payload["Soil_Moisture_%"]));
      localStorage.setItem("Wind_Speed_kmph", String(payload.Wind_Speed_kmph));
      localStorage.setItem("Crop_Stage", String(payload.Crop_Stage));
      localStorage.setItem("Previous_Pest_Count", String(payload.Previous_Pest_Count));
      localStorage.setItem("Days_Since_Last_Attack", String(payload.Days_Since_Last_Attack));
      localStorage.setItem("risk_level", String(data.risk_level));
      localStorage.setItem("confidence", String(data.confidence));

      setPrediction({
        lightgbm: data.risk_level,
        finalRisk: data.risk_level,
        confidence: `${(data.confidence * 100).toFixed(1)}%`,
        probabilities: data.probabilities || {},
      });
    } catch (error) {
      console.error("Prediction error:", error);
      setPrediction({
        lightgbm: "Error",
        finalRisk: "Error",
        confidence: "N/A",
        probabilities: {},
      });
    }
  };

  const probabilityRows = prediction
    ? [
        { label: "High", value: Number(prediction.probabilities?.High ?? 0) },
        { label: "Medium", value: Number(prediction.probabilities?.Medium ?? 0) },
        { label: "Low", value: Number(prediction.probabilities?.Low ?? 0) },
      ]
    : [];

  return (
    <div className="prediction-page">

      {/* =====================================================
          PAGE HEADER
          ===================================================== */}

      <section className="prediction-header">

        <span className="prediction-tag">
          AI RISK ENGINE
        </span>

        <h1>
          FAW Outbreak
          <br />
          Prediction
        </h1>

        <p>
          Analyze environmental conditions and estimate
          Fall Armyworm outbreak risk using time-series
          and machine-learning models.
        </p>

      </section>


      {/* =====================================================
          ENVIRONMENTAL INPUT PANEL
          ===================================================== */}

      <section className="prediction-console">

        <div className="prediction-console-header">

          <div>
            <span className="console-label">
              ENVIRONMENTAL SIGNALS
            </span>

            <h2>
              Current Field Conditions
            </h2>
          </div>

          <div className="prediction-status">
            <span></span>
            MODELS READY
          </div>

        </div>


        {/* =================================================
            INPUTS
            ================================================= */}

        <div className="environment-grid">

          <div className="environment-input">
            <div className="input-icon">📅</div>
            <label>
              Month
              <span>#</span>
            </label>
            <input
              type="number"
              min="1"
              max="12"
              value={month}
              onChange={(e) => setMonth(e.target.value)}
              placeholder="6"
            />
            <small>Month of the season</small>
          </div>

          <div className="environment-input">
            <div className="input-icon">🌡️</div>
            <label>
              Temperature
              <span>°C</span>
            </label>
            <input
              type="number"
              value={temperature}
              onChange={(e) => setTemperature(e.target.value)}
              placeholder="28"
            />
            <small>Atmospheric temperature</small>
          </div>

          <div className="environment-input">
            <div className="input-icon">💧</div>
            <label>
              Humidity
              <span>%</span>
            </label>
            <input
              type="number"
              value={humidity}
              onChange={(e) => setHumidity(e.target.value)}
              placeholder="75"
            />
            <small>Relative humidity</small>
          </div>

          <div className="environment-input">
            <div className="input-icon">🌧️</div>
            <label>
              Rainfall
              <span>mm</span>
            </label>
            <input
              type="number"
              value={rainfall}
              onChange={(e) => setRainfall(e.target.value)}
              placeholder="10"
            />
            <small>Recent precipitation</small>
          </div>

          <div className="environment-input">
            <div className="input-icon">🟫</div>
            <label>
              Soil Moisture
              <span>%</span>
            </label>
            <input
              type="number"
              value={soilMoisture}
              onChange={(e) => setSoilMoisture(e.target.value)}
              placeholder="55"
            />
            <small>Field moisture level</small>
          </div>

          <div className="environment-input">
            <div className="input-icon">💨</div>
            <label>
              Wind Speed
              <span>km/h</span>
            </label>
            <input
              type="number"
              value={windSpeed}
              onChange={(e) => setWindSpeed(e.target.value)}
              placeholder="8"
            />
            <small>Current wind conditions</small>
          </div>

          <div className="environment-input live-weather-wrapper" style={{ gridColumn: "1 / -1" }}>
            <button type="button" className="live-weather-button" onClick={handleUseLiveWeather}>
              <span className="live-weather-icon">☁️</span>
              Use Live Weather
            </button>
          </div>

          <div className="environment-input">
            <div className="input-icon">🌱</div>
            <label>
              Crop Stage
            </label>
            <select
              className="environment-select"
              value={cropStage}
              onChange={(e) => setCropStage(e.target.value)}
            >
              <option value="seedling">Seedling</option>
              <option value="vegetative">Vegetative</option>
              <option value="reproductive">Reproductive</option>
              <option value="maturity">Maturity</option>
            </select>
            <small>Crop growth stage</small>
          </div>

          <div className="environment-input">
            <div className="input-icon">🐛</div>
            <label>
              Previous Pest Count
            </label>
            <input
              type="number"
              value={previousPestCount}
              onChange={(e) => setPreviousPestCount(e.target.value)}
              placeholder="2"
            />
            <small>Recent pest observations</small>
          </div>

          <div className="environment-input">
            <div className="input-icon">⏱️</div>
            <label>
              Days Since Last Attack
            </label>
            <input
              type="number"
              value={daysSinceLastAttack}
              onChange={(e) => setDaysSinceLastAttack(e.target.value)}
              placeholder="7"
            />
            <small>Time since prior attack</small>
          </div>

        </div>


        {/* =================================================
            PREDICT BUTTON
            ================================================= */}

        <div className="prediction-action">

          <button
            className="predict-button"
            onClick={handlePrediction}
          >
            <span>✦</span>
            Analyze FAW Risk
          </button>

          <p>
            LightGBM environmental analysis
          </p>

        </div>


        {/* =================================================
            PREDICTION RESULT
            ================================================= */}

        {prediction && (

          <div className="prediction-result">

            <div className="prediction-result-header">

              <div>
                <span>
                  ANALYSIS COMPLETE
                </span>

                <h2>
                  FAW Risk Assessment
                </h2>
              </div>

              <div className="result-status">
                ●
              </div>

            </div>


            {/* Model outputs */}

            <div className="model-results">

              <div className="model-result-card">

                <div className="model-result-icon">
                  G
                </div>

                <div>
                  <span>
                    RISK CLASSIFIER
                  </span>

                  <h3>
                    LightGBM
                  </h3>
                </div>

                <strong>
                  {prediction.lightgbm}
                </strong>

              </div>

            </div>


            {/* Final result */}

            <div className="prediction-final">

              <span>
                FINAL AGRISHIELD ASSESSMENT
              </span>

              <h2>
                {prediction.finalRisk} Risk
              </h2>

              <p>
                Confidence: {prediction.confidence}
              </p>

            </div>

            <div className="prediction-probabilities">
              <div className="probability-header">
                <span>PROBABILITY BREAKDOWN</span>
              </div>

              {probabilityRows.map(({ label, value }) => (
                <div className="probability-row" key={label}>
                  <div className="probability-label-row">
                    <span>{label}</span>
                    <strong>{(value * 100).toFixed(1)}%</strong>
                  </div>

                  <div className="probability-bar">
                    <span style={{ width: `${Math.max(value * 100, 2)}%` }} />
                  </div>
                </div>
              ))}
            </div>

          </div>

        )}

      </section>

    </div>
  );
}

export default Prediction;