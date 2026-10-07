import { useEffect, useState } from "react";
import L from "leaflet";
import { MapContainer, Marker, TileLayer, useMap, useMapEvents } from "react-leaflet";
import markerIcon from "leaflet/dist/images/marker-icon.png";
import markerIconRetina from "leaflet/dist/images/marker-icon-2x.png";
import markerShadow from "leaflet/dist/images/marker-shadow.png";
import "leaflet/dist/leaflet.css";

L.Icon.Default.mergeOptions({
  iconRetinaUrl: markerIconRetina,
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
});

const locationPinIcon = L.icon({
  iconRetinaUrl: markerIconRetina,
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
});

const HYBRID_API_URL = "http://127.0.0.1:8000/api/predict/hybrid";
const DEFAULT_MAP_CENTER = [12.972, 77.594];

const WEATHER_LABELS = {
  Temperature_C: ["Temperature", "°C"],
  Humidity_pct: ["Humidity", "%"],
  Rainfall_mm: ["Rainfall", "mm"],
  Soil_Moisture_pct: ["Soil Moisture", "%"],
  Wind_Speed_kmph: ["Wind Speed", "km/h"],
  forecast_Temperature_C_mean_7d: ["Temperature", "°C"],
  forecast_Humidity_pct_mean_7d: ["Humidity", "%"],
  forecast_Rainfall_mm_mean_7d: ["Rainfall", "mm"],
  forecast_Soil_Moisture_pct_mean_7d: ["Soil Moisture", "%"],
  forecast_Wind_Speed_kmph_mean_7d: ["Wind Speed", "km/h"],
};

function WeatherPanel({ title, values }) {
  return (
    <div className="hybrid-detail-panel" style={{ padding: "24px" }}>
      <span className="console-label" style={{ display: "block", marginBottom: "16px" }}>{title}</span>
      <div className="hybrid-weather-grid" style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: "12px" }}>
        {Object.entries(values || {}).map(([key, value]) => (
          <div
            className="hybrid-weather-value"
            key={key}
            style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "16px", padding: "14px 16px", borderRadius: "10px", background: "rgba(233, 247, 237, 0.72)" }}
          >
            <span style={{ fontSize: "0.85rem", lineHeight: 1.35, color: "var(--muted)", textTransform: "none" }}>
              {WEATHER_LABELS[key]?.[0] || key.replaceAll("_", " ")}
            </span>
            <strong style={{ flexShrink: 0, fontSize: "1rem", lineHeight: 1.35, color: "var(--text)", whiteSpace: "nowrap" }}>
              {typeof value === "number" ? value.toFixed(1) : value}
              {WEATHER_LABELS[key]?.[1] && <small style={{ fontSize: "0.78rem", fontWeight: 600 }}> {WEATHER_LABELS[key][1]}</small>}
            </strong>
          </div>
        ))}
      </div>
    </div>
  );
}

function LocationMarker({ position, onSelect }) {
  useMapEvents({
    click(event) {
      onSelect(String(event.latlng.lat), String(event.latlng.lng));
    },
  });

  return position ? <Marker position={position} icon={locationPinIcon} /> : null;
}

function MapCenter({ latitude, longitude }) {
  const map = useMap();

  useEffect(() => {
    if (latitude && longitude) {
      map.setView([Number(latitude), Number(longitude)], map.getZoom(), { animate: true });
    }
  }, [latitude, longitude, map]);

  return null;
}

function Prediction() {
  const [image, setImage] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState("");
  const [sowingDate, setSowingDate] = useState("");
  const [latitude, setLatitude] = useState("");
  const [longitude, setLongitude] = useState("");
  const [locationSearch, setLocationSearch] = useState("");
  const [selectedLocationName, setSelectedLocationName] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [validationError, setValidationError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSearchingLocation, setIsSearchingLocation] = useState(false);

  useEffect(() => {
    if (!image) {
      setImagePreviewUrl("");
      return undefined;
    }

    const previewUrl = URL.createObjectURL(image);
    setImagePreviewUrl(previewUrl);

    return () => URL.revokeObjectURL(previewUrl);
  }, [image]);

  const handlePrediction = async (event) => {
    event.preventDefault();

    if (!image) {
      setValidationError("Upload a maize leaf image.");
      return;
    }
    if (!sowingDate) {
      setValidationError("Select a sowing date.");
      return;
    }
    if (!latitude || !longitude || !Number.isFinite(Number(latitude)) || !Number.isFinite(Number(longitude))) {
      setValidationError("Latitude and longitude must be valid numbers.");
      return;
    }

    setValidationError("");
    setPrediction(null);
    setIsSubmitting(true);

    const formData = new FormData();
    formData.append("image", image);
    formData.append("sowing_date", sowingDate);
    formData.append("latitude", latitude);
    formData.append("longitude", longitude);

    try {
      const response = await fetch(HYBRID_API_URL, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Hybrid prediction request failed.");
      }

      setPrediction(data);
    } catch (error) {
      console.error("Hybrid prediction error:", error);
      setValidationError(error.message || "Hybrid prediction failed.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const selectedPosition = latitude && longitude
    ? [Number(latitude), Number(longitude)]
    : null;

  const handleMapLocation = (selectedLatitude, selectedLongitude) => {
    setLatitude(selectedLatitude);
    setLongitude(selectedLongitude);
    setSelectedLocationName(`${selectedLatitude}, ${selectedLongitude}`);
    setValidationError("");
  };

  const handleLocationSearch = async (event) => {
    event.preventDefault();
    const query = locationSearch.trim();
    if (!query) {
      setValidationError("Enter a location to search.");
      return;
    }

    setValidationError("");
    setIsSearchingLocation(true);
    try {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?format=jsonv2&limit=1&q=${encodeURIComponent(query)}`,
      );
      if (!response.ok) {
        throw new Error("Location search failed.");
      }

      const results = await response.json();
      if (!results.length) {
        throw new Error("Location not found.");
      }

      const result = results[0];
      handleMapLocation(result.lat, result.lon);
      setSelectedLocationName(result.display_name || query);
    } catch (error) {
      setValidationError(error.message || "Location search failed.");
    } finally {
      setIsSearchingLocation(false);
    }
  };

  const handleUseLiveLocation = () => {
    if (!navigator.geolocation) {
      setValidationError("Live location is not supported by this browser.");
      return;
    }

    setValidationError("");
    navigator.geolocation.getCurrentPosition(
      ({ coords }) => handleMapLocation(coords.latitude.toFixed(6), coords.longitude.toFixed(6)),
      () => setValidationError("Could not access your live location."),
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 },
    );
  };

  return (
    <div className="prediction-page">
      <section className="prediction-header">
        <span className="prediction-tag">AI RISK ENGINE</span>
        <h1>
          FAW Outbreak
          <br />
          Prediction
        </h1>
        <p>
          Combine a maize leaf scan with location weather history to estimate
          Fall Armyworm attack probability.
        </p>
      </section>

      <section className="prediction-console">
        <div className="prediction-console-header">
          <div>
            <span className="console-label">HYBRID SIGNALS</span>
            <h2>Field Risk Assessment</h2>
          </div>
          <div className="prediction-status">
            <span />
            MODELS READY
          </div>
        </div>

        <form className="hybrid-form" onSubmit={handlePrediction}>
          <label className="hybrid-upload-field">
            <span className="input-icon">📷</span>
            <span className="hybrid-field-label">Maize leaf image</span>
            <input
              type="file"
              accept="image/*"
              onChange={(event) => setImage(event.target.files?.[0] || null)}
            />
            {image ? (
              <span style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                {imagePreviewUrl && (
                  <img
                    src={imagePreviewUrl}
                    alt="Selected maize leaf preview"
                    style={{ width: "48px", height: "48px", objectFit: "cover", borderRadius: "8px" }}
                  />
                )}
                <small>{image.name}</small>
              </span>
            ) : (
              <small>PNG, JPG, or WEBP</small>
            )}
          </label>

          <div className="hybrid-input-field" style={{ gridColumn: "span 2" }}>
            <span className="input-icon">📍</span>
            <span className="hybrid-field-label">Field location</span>
            <MapContainer
              center={DEFAULT_MAP_CENTER}
              zoom={5}
              scrollWheelZoom
              style={{ width: "100%", height: "240px", borderRadius: "10px", marginTop: "8px" }}
            >
              <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />
              <MapCenter latitude={latitude} longitude={longitude} />
              <LocationMarker position={selectedPosition} onSelect={handleMapLocation} />
            </MapContainer>
            <button
              className="predict-button"
              type="button"
              onClick={handleUseLiveLocation}
              style={{ marginTop: "10px", alignSelf: "flex-start" }}
            >
              Use My Live Location
            </button>
            <small>
              {selectedPosition
                ? `Selected: ${latitude}, ${longitude}`
                : "Click the map or use your live location"}
            </small>
            <div style={{ display: "flex", gap: "10px", alignItems: "center", marginTop: "12px" }}>
              <input
                type="text"
                value={locationSearch}
                onChange={(event) => setLocationSearch(event.target.value)}
                placeholder="Enter location (e.g. Bengaluru, Karnataka)"
                style={{ flex: 1, minWidth: 0 }}
              />
              <button
                className="predict-button"
                type="button"
                onClick={handleLocationSearch}
                disabled={isSearchingLocation}
              >
                {isSearchingLocation ? "Searching..." : "Search Location"}
              </button>
            </div>
            {selectedLocationName && (
              <small>Selected Location: {selectedLocationName}</small>
            )}
          </div>

          <label className="hybrid-input-field">
            <span className="input-icon">📅</span>
            <span className="hybrid-field-label">Sowing date</span>
            <input
              type="date"
              value={sowingDate}
              onChange={(event) => setSowingDate(event.target.value)}
              required
            />
            <small>DAS is calculated automatically</small>
          </label>

          <div className="prediction-action hybrid-submit-row">
            <button className="predict-button" type="submit" disabled={isSubmitting}>
              <span>✦</span>
              {isSubmitting ? "Analyzing..." : "Analyze FAW Risk"}
            </button>
            <p>Hybrid LSTM + LightGBM analysis</p>
            {validationError && <p role="alert">{validationError}</p>}
          </div>
        </form>

        {prediction && (
          <div className="prediction-result hybrid-result">
            <div className="prediction-result-header">
              <div>
                <span>ANALYSIS COMPLETE</span>
                <h2>FAW Risk Assessment</h2>
              </div>
              <div className="result-status">●</div>
            </div>

            <div className="hybrid-risk-summary" style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: "16px" }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "20px", padding: "22px 24px", borderRadius: "12px", background: "var(--green-soft)" }}>
                <span style={{ fontSize: "0.8rem", fontWeight: 700, letterSpacing: "0.08em", color: "var(--muted)" }}>FAW ATTACK PROBABILITY</span>
                <strong style={{ fontSize: "clamp(1.8rem, 4vw, 2.7rem)", lineHeight: 1, color: "var(--green-800)", whiteSpace: "nowrap" }}>{Number(prediction.faw_attack_probability).toFixed(1)}%</strong>
              </div>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "20px", padding: "22px 24px", borderRadius: "12px", background: "#fff7df" }}>
                <span style={{ fontSize: "0.8rem", fontWeight: 700, letterSpacing: "0.08em", color: "var(--muted)" }}>RISK LEVEL</span>
                <strong style={{ fontSize: "clamp(1.4rem, 3vw, 2rem)", lineHeight: 1, color: "var(--green-800)", whiteSpace: "nowrap" }}>{String(prediction.risk_level).toUpperCase()}</strong>
              </div>
            </div>

            <div className="hybrid-meta-grid" style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: "16px", marginTop: "16px" }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "16px", padding: "16px 20px", border: "1px solid var(--border)", borderRadius: "10px", background: "#fff" }}>
                <span style={{ fontSize: "0.82rem", fontWeight: 700, letterSpacing: "0.06em", color: "var(--muted)" }}>Days After Sowing (DAS)</span>
                <strong style={{ fontSize: "1.1rem", color: "var(--text)", whiteSpace: "nowrap" }}>{prediction.das} days</strong>
              </div>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "16px", padding: "16px 20px", border: "1px solid var(--border)", borderRadius: "10px", background: "#fff" }}>
                <span style={{ fontSize: "0.82rem", fontWeight: 700, letterSpacing: "0.06em", color: "var(--muted)" }}>CROP STAGE</span>
                <strong style={{ fontSize: "1.1rem", color: "var(--text)", whiteSpace: "nowrap" }}>{prediction.crop_stage}</strong>
              </div>
            </div>

                <div className="hybrid-detail-grid">
              <WeatherPanel title="CURRENT WEATHER" values={prediction.current_weather} />
              <WeatherPanel title="7-DAY FORECAST AVERAGE" values={prediction.forecast_7_day} />
            </div>
          </div>
        )}
      </section>
    </div>
  );
}

export default Prediction;