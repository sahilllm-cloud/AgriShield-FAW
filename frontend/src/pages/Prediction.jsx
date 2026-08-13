import { useState } from "react";

function Prediction() {
  const [temperature, setTemperature] = useState("");
  const [humidity, setHumidity] = useState("");
  const [rainfall, setRainfall] = useState("");
  const [windSpeed, setWindSpeed] = useState("");

  const [prediction, setPrediction] = useState(null);

  const handlePrediction = () => {
    // Temporary frontend prediction result.
    // Later this will be connected to the LSTM + LightGBM backend.

    setPrediction({
      lstm: "High",
      lightgbm: "High",
      finalRisk: "High",
    });
  };

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

          {/* Temperature */}

          <div className="environment-input">

            <div className="input-icon">
              🌡️
            </div>

            <label>
              Temperature
              <span>°C</span>
            </label>

            <input
              type="number"
              value={temperature}
              onChange={(e) =>
                setTemperature(e.target.value)
              }
              placeholder="28"
            />

            <small>
              Atmospheric temperature
            </small>

          </div>


          {/* Humidity */}

          <div className="environment-input">

            <div className="input-icon">
              💧
            </div>

            <label>
              Humidity
              <span>%</span>
            </label>

            <input
              type="number"
              value={humidity}
              onChange={(e) =>
                setHumidity(e.target.value)
              }
              placeholder="75"
            />

            <small>
              Relative humidity
            </small>

          </div>


          {/* Rainfall */}

          <div className="environment-input">

            <div className="input-icon">
              🌧️
            </div>

            <label>
              Rainfall
              <span>mm</span>
            </label>

            <input
              type="number"
              value={rainfall}
              onChange={(e) =>
                setRainfall(e.target.value)
              }
              placeholder="10"
            />

            <small>
              Recent precipitation
            </small>

          </div>


          {/* Wind */}

          <div className="environment-input">

            <div className="input-icon">
              💨
            </div>

            <label>
              Wind Speed
              <span>km/h</span>
            </label>

            <input
              type="number"
              value={windSpeed}
              onChange={(e) =>
                setWindSpeed(e.target.value)
              }
              placeholder="8"
            />

            <small>
              Current wind conditions
            </small>

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
            LSTM + LightGBM environmental analysis
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
                  L
                </div>

                <div>
                  <span>
                    TIME-SERIES MODEL
                  </span>

                  <h3>
                    LSTM
                  </h3>
                </div>

                <strong>
                  {prediction.lstm}
                </strong>

              </div>


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
                Combined model assessment
              </p>

            </div>

          </div>

        )}

      </section>

    </div>
  );
}

export default Prediction;