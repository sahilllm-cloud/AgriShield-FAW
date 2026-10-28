import { useEffect, useRef, useState } from "react";

import L from "leaflet";

import {

  MapContainer,

  Marker,

  TileLayer,

  useMap,

  useMapEvents,

} from "react-leaflet";

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

const HYBRID_API_URL =

  "http://localhost:8000/api/predict/hybrid";

const DEFAULT_MAP_CENTER = [12.972, 77.594];

// ============================================================

// TODAY'S DATE

// ============================================================

function getTodayDate() {

  const today = new Date();

  const year = today.getFullYear();

  const month = String(

    today.getMonth() + 1

  ).padStart(2, "0");

  const day = String(

    today.getDate()

  ).padStart(2, "0");

  return `${year}-${month}-${day}`;

}

// ============================================================

// CROP STAGE

// ============================================================

function getCropStage(das) {

  if (das <= 14) {

    return "Seedling";

  }

  if (das <= 30) {

    return "Vegetative";

  }

  if (das <= 45) {

    return "Whorl";

  }

  if (das <= 60) {

    return "Tasseling";

  }

  if (das <= 75) {

    return "Silking";

  }

  return "Maturity";

}

// ============================================================

// WEATHER PANEL

// ============================================================

function WeatherPanel({

  title,

  values,

}) {

  if (!values) {

    return null;

  }

  const labels = {

    Temperature_C: [

      "Temperature",

      "°C",

    ],

    "Humidity_%": [

      "Humidity",

      "%",

    ],

    Rainfall_mm: [

      "Rainfall",

      "mm",

    ],

    "Soil_Moisture_%": [

      "Soil Moisture",

      "%",

    ],

    Wind_Speed_kmph: [

      "Wind Speed",

      "km/h",

    ],

  };

  return (

    <div

      className="hybrid-detail-panel"

      style={{

        padding: "24px",

      }}

    >

      <span

        className="console-label"

        style={{

          display: "block",

          marginBottom: "16px",

        }}

      >

        {title}

      </span>

      <div

        style={{

          display: "grid",

          gridTemplateColumns:

            "repeat(2, minmax(0, 1fr))",

          gap: "12px",

        }}

      >

        {Object.entries(values).map(

          ([key, value]) => (

            <div

              key={key}

              style={{

                display: "flex",

                alignItems:

                  "center",

                justifyContent:

                  "space-between",

                gap: "16px",

                padding:

                  "14px 16px",

                borderRadius:

                  "10px",

                background:

                  "rgba(233, 247, 237, 0.72)",

              }}

            >

              <span

                style={{

                  fontSize:

                    "0.85rem",

                  color:

                    "var(--muted)",

                }}

              >

                {labels[key]?.[0] ||

                  key.replaceAll(

                    "_",

                    " "

                  )}

              </span>

              <strong

                style={{

                  fontSize:

                    "1rem",

                  color:

                    "var(--text)",

                  whiteSpace:

                    "nowrap",

                }}

              >

                {typeof value ===

                "number"

                  ? value.toFixed(1)

                  : value}

                {labels[key]?.[1] && (

                  <small>

                    {" "}

                    {

                      labels[key][1]

                    }

                  </small>

                )}

              </strong>

            </div>

          )

        )}

      </div>

    </div>

  );

}

// ============================================================

// MAP MARKER

// ============================================================

function LocationMarker({

  position,

  onSelect,

}) {

  useMapEvents({

    click(event) {

      onSelect(

        String(

          event.latlng.lat

        ),

        String(

          event.latlng.lng

        )

      );

    },

  });

  return position ? (

    <Marker

      position={position}

      icon={locationPinIcon}

    />

  ) : null;

}

// ============================================================

// MAP CENTER

// ============================================================

function MapCenter({

  latitude,

  longitude,

}) {

  const map = useMap();

  useEffect(() => {

    if (

      latitude &&

      longitude

    ) {

      map.setView(

        [

          Number(latitude),

          Number(longitude),

        ],

        map.getZoom(),

        {

          animate: true,

        }

      );

    }

  }, [

    latitude,

    longitude,

    map,

  ]);

  return null;

}

// ============================================================

// MAIN PAGE

// ============================================================

function Prediction() {

  const [image, setImage] =

    useState(null);

  const [

    imagePreviewUrl,

    setImagePreviewUrl,

  ] = useState("");

  const [

    sowingDate,

    setSowingDate,

  ] = useState("");

  const [

    latitude,

    setLatitude,

  ] = useState("");

  const [

    longitude,

    setLongitude,

  ] = useState("");

  const [

    locationSearch,

    setLocationSearch,

  ] = useState("");

  const [

    selectedLocationName,

    setSelectedLocationName,

  ] = useState("");

  const [

    prediction,

    setPrediction,

  ] = useState(null);

  const [

    validationError,

    setValidationError,

  ] = useState("");

  const [

    isSubmitting,

    setIsSubmitting,

  ] = useState(false);

  const [

    isSearchingLocation,

    setIsSearchingLocation,

  ] = useState(false);

  const predictionRequestId =

    useRef(0);

  // ==========================================================

  // IMAGE PREVIEW

  // ==========================================================

  useEffect(() => {

    if (!image) {

      setImagePreviewUrl("");

      return undefined;

    }

    const previewUrl =

      URL.createObjectURL(

        image

      );

    setImagePreviewUrl(

      previewUrl

    );

    return () =>

      URL.revokeObjectURL(

        previewUrl

      );

  }, [image]);

  // ==========================================================

  // IMAGE SELECTION

  // ==========================================================

  const handleImageChange =

    (event) => {

      const file =

        event.target.files?.[0];

      if (!file) {

        return;

      }

      setImage(file);

      setPrediction(null);

      setValidationError("");

      predictionRequestId.current += 1;

      console.info(

        "Hybrid image selected:",

        {

          name: file.name,

          size: file.size,

          lastModified: file.lastModified,

        }

      );

    };

  // ==========================================================

  // HYBRID PREDICTION

  // ==========================================================

  const handlePrediction =

    async (event) => {

      event.preventDefault();

      if (!image) {

        setValidationError(

          "Upload a maize leaf image."

        );

        return;

      }

      if (!sowingDate) {

        setValidationError(

          "Select the sowing date."

        );

        return;

      }

      if (

        !latitude ||

        !longitude ||

        !Number.isFinite(

          Number(latitude)

        ) ||

        !Number.isFinite(

          Number(longitude)

        )

      ) {

        setValidationError(

          "Select a valid field location."

        );

        return;

      }

      if (

        new Date(

          `${sowingDate}T00:00:00`

        ) >

        new Date(

          `${getTodayDate()}T00:00:00`

        )

      ) {

        setValidationError(

          "Sowing date cannot be after today."

        );

        return;

      }

      // --------------------------------------------------------

      // START

      // --------------------------------------------------------

      setValidationError("");

      setPrediction(null);

      setIsSubmitting(true);

      const requestId =

        predictionRequestId.current + 1;

      predictionRequestId.current =

        requestId;

      const submittedImage =

        image;

      const formData =

        new FormData();

      formData.append(

        "file",

        image

      );

      const url =

        new URL(

          HYBRID_API_URL

        );

      url.searchParams.set(

        "date",

        sowingDate

      );

      url.searchParams.set(

        "latitude",

        latitude

      );

      url.searchParams.set(

        "longitude",

        longitude

      );

      url.searchParams.set(

        "crop_stage",

        calculatedStage ||

          "Whorl"

      );

      url.searchParams.set(

        "crop_age_days",

        String(

          calculatedDas ??

            35

        )

      );

      url.searchParams.set(

        "maize_variety",

        "Local"

      );

      console.info(

        "Hybrid prediction request:",

        {

          requestId,

          name: submittedImage.name,

          size: submittedImage.size,

          date: sowingDate,

          latitude,

          longitude,

        }

      );

      try {

        const response =

          await fetch(

            url.toString(),

            {

              method:

                "POST",

              body:

                formData,

            }

          );

        const responseText =

          await response.text();

        let data;

        try {

          data =

            JSON.parse(

              responseText

            );

        } catch {

          throw new Error(

            "Backend returned an invalid response."

          );

        }

        if (!response.ok) {

          console.error(

            "Hybrid prediction error response:",

            data

          );

          const detail =

            Array.isArray(data.detail)

              ? data.detail

                  .map(

                    (validationError) =>

                      `${validationError.loc?.join(".") || "request"}: ${validationError.msg}`

                  )

                  .join("; ")

              : data.detail;

          throw new Error(

            detail ||

              "Hybrid prediction failed."

          );

        }

        console.info(

          "Hybrid prediction response:",

          {

            requestId,

            name: submittedImage.name,

            response: data,

            displayedProbability:

              data.prediction?.faw_attack_probability,

            displayedRisk:

              data.prediction?.risk_level,

          }

        );

        if (

          requestId !==

          predictionRequestId.current

        ) {

          console.info(

            "Ignoring stale hybrid response:",

            requestId

          );

          return;

        }

        setPrediction(

          data

        );

// Save successful predictions to History without interrupting the result.

        try {

          let guestId = localStorage.getItem("agrishield_guest_id");

          if (!guestId) {

            guestId =

              typeof crypto !== "undefined" && crypto.randomUUID

                ? crypto.randomUUID()

                : `guest-${Date.now()}-${Math.random().toString(36).slice(2)}`;

            localStorage.setItem("agrishield_guest_id", guestId);

          }

          const token = localStorage.getItem("access_token");

          const historyHeaders = {

            "Content-Type": "application/json",

            "X-Guest-ID": guestId,

          };

          if (token) {

            historyHeaders.Authorization = `Bearer ${token}`;

          }

          const historyResponse = await fetch(

            "http://127.0.0.1:8000/api/history",

            {

              method: "POST",

              headers: historyHeaders,

              body: JSON.stringify({ prediction: data }),

            }

          );

          if (!historyResponse.ok) {

            console.error(

              "Prediction succeeded, but History saving failed:",

              await historyResponse.text()

            );

          } else {

            console.info("Prediction saved to History successfully.");

          }

        } catch (historyError) {

          console.error("Could not save prediction to History:", historyError);

        }

      } catch (error) {

        console.error(

          "Hybrid prediction error:",

          error

        );

        setValidationError(

          error.message ||

            "Hybrid prediction failed."

        );

      } finally {

        setIsSubmitting(

          false

        );

      }

    };

  // ==========================================================

  // LOCATION

  // ==========================================================

  const selectedPosition =

    latitude &&

    longitude

      ? [

          Number(latitude),

          Number(longitude),

        ]

      : null;

  const handleMapLocation =

    (

      selectedLatitude,

      selectedLongitude

    ) => {

      setLatitude(

        selectedLatitude

      );

      setLongitude(

        selectedLongitude

      );

      setSelectedLocationName(

        `${selectedLatitude}, ${selectedLongitude}`

      );

      setValidationError("");

    };

  // ==========================================================

  // LOCATION SEARCH

  // ==========================================================

  const handleLocationSearch =

    async (event) => {

      event.preventDefault();

      const query =

        locationSearch.trim();

      if (!query) {

        setValidationError(

          "Enter a location to search."

        );

        return;

      }

      setValidationError("");

      setIsSearchingLocation(

        true

      );

      try {

        const response =

          await fetch(

            `https://nominatim.openstreetmap.org/search?format=jsonv2&limit=1&q=${encodeURIComponent(

              query

            )}`

          );

        if (!response.ok) {

          throw new Error(

            "Location search failed."

          );

        }

        const results =

          await response.json();

        if (!results.length) {

          throw new Error(

            "Location not found."

          );

        }

        const result =

          results[0];

        handleMapLocation(

          result.lat,

          result.lon

        );

        setSelectedLocationName(

          result.display_name ||

            query

        );

      } catch (error) {

        setValidationError(

          error.message ||

            "Location search failed."

        );

      } finally {

        setIsSearchingLocation(

          false

        );

      }

    };

  // ==========================================================

  // LIVE LOCATION

  // ==========================================================

  const handleUseLiveLocation =

    () => {

      if (

        !navigator.geolocation

      ) {

        setValidationError(

          "Live location is not supported by this browser."

        );

        return;

      }

      setValidationError("");

      navigator.geolocation.getCurrentPosition(

        ({ coords }) => {

          handleMapLocation(

            coords.latitude.toFixed(

              6

            ),

            coords.longitude.toFixed(

              6

            )

          );

        },

        () => {

          setValidationError(

            "Could not access your live location."

          );

        },

        {

          enableHighAccuracy:

            true,

          timeout:

            10000,

          maximumAge:

            0,

        }

      );

    };

  // ==========================================================

  // TODAY / DAS

  // ==========================================================

  const todayDate =

    getTodayDate();

  let calculatedDas =

    null;

  let calculatedStage =

    null;

  if (sowingDate) {

    const sowing =

      new Date(

        `${sowingDate}T00:00:00`

      );

    const today =

      new Date(

        `${todayDate}T00:00:00`

      );

    const difference =

      Math.floor(

        (

          today.getTime() -

          sowing.getTime()

        ) /

          (

            1000 *

            60 *

            60 *

            24

          )

      );

    if (difference >= 0) {

      calculatedDas =

        difference;

      calculatedStage =

        getCropStage(

          difference

        );

    }

  }

  // ==========================================================

  // UI

  // ==========================================================

  return (

    <div className="prediction-page">

      {/* ======================================================

          HEADER

          ====================================================== */}

      <section className="prediction-header">

        <span className="prediction-tag">

          AGRI SHIELD AI

        </span>

        <h1>

          FAW Outbreak

          <br />

          Prediction

        </h1>

        <p>

          Scan a maize leaf and combine

          image intelligence with crop

          stage and live environmental

          conditions to estimate FAW

          attack probability.

        </p>

      </section>

      {/* ======================================================

          MAIN HYBRID CONSOLE

          ====================================================== */}

      <section className="prediction-console">

        <div className="prediction-console-header">

          <div>

            <span className="console-label">

              HYBRID SIGNALS

            </span>

            <h2>

              Maize Field Assessment

            </h2>

          </div>

          <div className="prediction-status">

            <span />

            AI READY

          </div>

        </div>

        {/* ====================================================

            IMAGE SCANNER

            ==================================================== */}

        <div

          style={{

            marginBottom:

              "24px",

          }}

        >

          <div

            style={{

              display:

                "flex",

              alignItems:

                "center",

              justifyContent:

                "space-between",

              gap:

                "16px",

              marginBottom:

                "14px",

            }}

          >

            <div>

              <span className="console-label">

                IMAGE ANALYSIS

              </span>

              <h3

                style={{

                  margin:

                    "6px 0 0",

                  fontSize:

                    "1.35rem",

                }}

              >

                Maize Leaf Scanner

              </h3>

            </div>

            <div className="scanner-status">

              <span />

              SWIN READY

            </div>

          </div>

          <div

            className={`scanner-area ${

              imagePreviewUrl

                ? "has-image"

                : ""

            }`}

          >

            {!imagePreviewUrl ? (

              <label className="upload-zone">

                <input

                  type="file"

                  accept="image/*"

                  onChange={

                    handleImageChange

                  }

                />

                <div className="upload-icon">

                  📷

                </div>

                <h3>

                  Upload Maize Leaf

                </h3>

                <p>

                  Drop an image here

                  or click to browse

                </p>

                <span className="upload-format">

                  JPG • JPEG • PNG

                </span>

              </label>

            ) : (

              <div className="image-scanner">

                <img

                  src={

                    imagePreviewUrl

                  }

                  alt="Selected maize leaf"

                />

                <div className="scan-line" />

                <div className="corner top-left" />

                <div className="corner top-right" />

                <div className="corner bottom-left" />

                <div className="corner bottom-right" />

              </div>

            )}

          </div>

          <div

            className="scanner-actions"

            style={{

              marginTop:

                "14px",

            }}

          >

            <label className="change-image">

              {image

                ? "Change Image"

                : "Choose Image"}

              <input

                type="file"

                accept="image/*"

                onChange={

                  handleImageChange

                }

              />

            </label>

          </div>

</div>

        {/* ====================================================

            SOWING DATE

            ==================================================== */}

        <div

          className="hybrid-form"

        >

          <label className="hybrid-input-field">

            <span className="input-icon">

              🌱

            </span>

            <span className="hybrid-field-label">

              Sowing date

            </span>

            <input

              type="date"

              value={

                sowingDate

              }

              onChange={(

                event

              ) =>

                setSowingDate(

                  event.target.value

                )

              }

              required

            />

            <small>

              DAS and crop stage are

              calculated automatically.

            </small>

          </label>

          {/* TODAY */}

          <div

            className="hybrid-input-field"

          >

            <span className="input-icon">

              📅

            </span>

            <span className="hybrid-field-label">

              Today's date

            </span>

            <strong

              style={{

                display:

                  "block",

                marginTop:

                  "8px",

                fontSize:

                  "1rem",

              }}

            >

              {todayDate}

            </strong>

            <small>

              Live conditions are used

              for today's assessment.

            </small>

          </div>

          {/* ==================================================

              LOCATION

              ================================================== */}

          <div

            className="hybrid-input-field"

            style={{

              gridColumn:

                "span 2",

            }}

          >

            <span className="input-icon">

              📍

            </span>

            <span className="hybrid-field-label">

              Field location

            </span>

            <MapContainer

              center={

                DEFAULT_MAP_CENTER

              }

              zoom={5}

              scrollWheelZoom

              style={{

                width:

                  "100%",

                height:

                  "240px",

                borderRadius:

                  "10px",

                marginTop:

                  "8px",

              }}

            >

              <TileLayer

                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'

                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"

              />

              <MapCenter

                latitude={

                  latitude

                }

                longitude={

                  longitude

                }

              />

              <LocationMarker

                position={

                  selectedPosition

                }

                onSelect={

                  handleMapLocation

                }

              />

            </MapContainer>

            <button

              className="predict-button"

              type="button"

              onClick={

                handleUseLiveLocation

              }

              style={{

                marginTop:

                  "10px",

                alignSelf:

                  "flex-start",

              }}

            >

              Use My Live Location

            </button>

            <small>

              {selectedPosition

                ? `Selected: ${latitude}, ${longitude}`

                : "Click the map or use your live location"}

            </small>

            <div

              style={{

                display:

                  "flex",

                gap:

                  "10px",

                alignItems:

                  "center",

                marginTop:

                  "12px",

              }}

            >

              <input

                type="text"

                value={

                  locationSearch

                }

                onChange={(

                  event

                ) =>

                  setLocationSearch(

                    event.target.value

                  )

                }

                placeholder="Enter location"

                style={{

                  flex:

                    1,

                  minWidth:

                    0,

                }}

              />

              <button

                className="predict-button"

                type="button"

                onClick={

                  handleLocationSearch

                }

                disabled={

                  isSearchingLocation

                }

              >

                {isSearchingLocation

                  ? "Searching..."

                  : "Search Location"}

              </button>

            </div>

            {selectedLocationName && (

              <small>

                Selected Location:{" "}

                {

                  selectedLocationName

                }

              </small>

            )}

          </div>

          {/* ==================================================

              DAS + CROP STAGE

              ================================================== */}

          {calculatedDas !== null && (

            <div

              style={{

                gridColumn:

                  "span 2",

                display:

                  "grid",

                gridTemplateColumns:

                  "repeat(2, minmax(0, 1fr))",

                gap:

                  "12px",

              }}

            >

              <div

                style={{

                  padding:

                    "16px",

                  borderRadius:

                    "10px",

                  background:

                    "rgba(233, 247, 237, 0.72)",

                }}

              >

                <small>

                  DAYS AFTER SOWING

                </small>

                <strong

                  style={{

                    display:

                      "block",

                    fontSize:

                      "1.4rem",

                    marginTop:

                      "6px",

                  }}

                >

                  {calculatedDas}

                  {" "}

                  days

                </strong>

              </div>

              <div

                style={{

                  padding:

                    "16px",

                  borderRadius:

                    "10px",

                  background:

                    "rgba(233, 247, 237, 0.72)",

                }}

              >

                <small>

                  CROP STAGE

                </small>

                <strong

                  style={{

                    display:

                      "block",

                    fontSize:

                      "1.4rem",

                    marginTop:

                      "6px",

                  }}

                >

                  {calculatedStage}

                </strong>

              </div>

            </div>

          )}

          {/* ==================================================

              ANALYZE

              ================================================== */}

          <div

            className="prediction-action hybrid-submit-row"

          >

            <button

              className="predict-button"

              type="submit"

              onClick={

                handlePrediction

              }

              disabled={

                isSubmitting

              }

            >

              <span>

                ✦

              </span>

              {isSubmitting

                ? "Scanning & Analyzing..."

                : "Scan & Analyze FAW Risk"}

            </button>

            <p>

              Swin Transformer + LSTM

              + LightGBM hybrid analysis

            </p>

            {validationError && (

              <p

                role="alert"

                style={{

                  color:

                    "#b42318",

                }}

              >

                {validationError}

              </p>

            )}

          </div>

        </div>

        {/* ====================================================

            ONE COMBINED RESULT

            ==================================================== */}

        {prediction && (

          <div

            className="prediction-result hybrid-result"

            style={{

              marginTop:

                "28px",

            }}

          >

            <div className="prediction-result-header">

              <div>

                <span>

                  COMPLETE HYBRID ANALYSIS

                </span>

                <h2>

                  FAW Risk Assessment

                </h2>

              </div>

              <div className="result-status">

                ●

              </div>

            </div>

{/* ==================================================

                FINAL FAW RESULT

                ================================================== */}

            <div

              style={{

                display:

                  "grid",

                gridTemplateColumns:

                  "repeat(2, minmax(0, 1fr))",

                gap:

                  "16px",

                marginTop:

                  "16px",

              }}

            >

              <div

                style={{

                  display:

                    "flex",

                  alignItems:

                    "center",

                  justifyContent:

                    "space-between",

                  gap:

                    "20px",

                  padding:

                    "22px 24px",

                  borderRadius:

                    "12px",

                  background:

                    "var(--green-soft)",

                }}

              >

                <span

                  style={{

                    fontSize:

                      "0.8rem",

                    fontWeight:

                      700,

                    letterSpacing:

                      "0.08em",

                    color:

                      "var(--muted)",

                  }}

                >

                  FAW ATTACK PROBABILITY

                </span>

                <strong

                  style={{

                    fontSize:

                      "clamp(1.8rem, 4vw, 2.7rem)",

                    lineHeight:

                      1,

                    color:

                      "var(--green-800)",

                    whiteSpace:

                      "nowrap",

                  }}

                >

                  {

                    prediction

                      .prediction

                      ?.faw_attack_probability

                  }

                  %

                </strong>

              </div>

              <div

                style={{

                  display:

                    "flex",

                  alignItems:

                    "center",

                  justifyContent:

                    "space-between",

                  gap:

                    "20px",

                  padding:

                    "22px 24px",

                  borderRadius:

                    "12px",

                  background:

                    "#fff7df",

                }}

              >

                <span

                  style={{

                    fontSize:

                      "0.8rem",

                    fontWeight:

                      700,

                    letterSpacing:

                      "0.08em",

                    color:

                      "var(--muted)",

                  }}

                >

                  RISK LEVEL

                </span>

                <strong

                  style={{

                    fontSize:

                      "clamp(1.4rem, 3vw, 2rem)",

                    lineHeight:

                      1,

                    color:

                      "var(--green-800)",

                    whiteSpace:

                      "nowrap",

                  }}

                >

                  {

                    prediction

                      .prediction

                      ?.risk_level

                  }

                </strong>

              </div>

            </div>

            {/* ==================================================

                RISK-BASED FARMER RECOMMENDATIONS

                ================================================== */}

            {(() => {

              const riskLevel = String(

                prediction?.prediction?.risk_level ?? ""

              ).trim().toUpperCase();

              const recommendations = {

                HIGH: {

                  title: "Immediate Field Attention Recommended",

                  color: "#b91c1c",

                  background: "#fef2f2",

                  actions: [

                    "Inspect the field promptly, especially young leaves and plant whorls.",

                    "Look for FAW egg masses, larvae, and fresh feeding damage.",

                    "If infestation is confirmed, consult local agricultural guidance on suitable integrated pest management measures.",

                    "Reinspect affected areas and record changes in damage."

                  ]

                },

                MEDIUM: {

                  title: "Increase Field Monitoring",

                  color: "#92400e",

                  background: "#fffbeb",

                  actions: [

                    "Inspect maize plants regularly for new feeding damage.",

                    "Check young leaves and plant whorls for larvae.",

                    "Monitor crop and weather conditions for changes in risk.",

                    "Seek local agricultural advice if infestation is observed."

                  ]

                },

                LOW: {

                  title: "Continue Routine Monitoring",

                  color: "#166534",

                  background: "#f0fdf4",

                  actions: [

                    "Continue regular field inspections.",

                    "Check for new feeding damage and FAW egg masses.",

                    "Maintain good crop management and field hygiene.",

                    "Reassess risk if weather conditions or field observations change."

                  ]

                }

              };

              const advisory = recommendations[riskLevel];

              return (

                <section

                  aria-live="polite"

                  style={{

                    marginTop: "20px",

                    padding: "22px 24px",

                    borderRadius: "12px",

                    background: "#f4fbf5",

                    borderLeft: "5px solid #176b3a",

                    color: "#1f2937"

                  }}

                >

                  <h3

                    style={{

                      margin: "0 0 8px",

                      color: "#202820",

                      fontSize: "1.15rem"

                    }}

                  >

                    RECOMMENDED ACTIONS

                  </h3>

                  <p style={{ margin: "0 0 14px", fontWeight: 600 }}>

                    {advisory?.title || "Review the Risk Assessment"}

                  </p>

                  {advisory ? (

                    <div className="recommendation-action-grid">

                      {advisory.actions.map((action, index) => {

                
                        const actionTitles = [

                          "Inspect crops",

                          "Check for FAW",

                          "Maintain field hygiene",

                          "Reassess risk",

                        ];

                        return (

                          <article

                            className="recommendation-action-card"

                            key={`${riskLevel}-${index}`}

                          >

                            <span className="recommendation-action-icon" aria-hidden="true">

                              {["👁️", "🔎", "🌱", "↻"][index]}

                            </span>

                            <div className="recommendation-action-copy">

                              <h4>{actionTitles[index] || `Action ${index + 1}`}</h4>

                              <p>{action}</p>

                            </div>

                            <span className="recommendation-action-number">

                              {String(index + 1).padStart(2, "0")}

                            </span>

                          </article>

                        );

                      })}

                    </div>

                  ) : (

                    <p>

                      A recognized risk level is unavailable. Inspect field

                      conditions and review the prediction before deciding

                      on control measures.

                    </p>

                  )}

                  <p

                    style={{

                      margin: "14px 0 0",

                      fontSize: "0.85rem",

                      lineHeight: 1.5,

                      opacity: 0.8

                    }}

                  >

                    Advisory only: predicted risk does not confirm an

                    infestation. Verify field conditions and follow local

                    agricultural guidance.

                  </p>

                </section>

              );

            })()}

            {/* ==================================================

                FUTURE PEST PREVENTION

                ================================================== */}

            {(() => {

              const attackProbability = Number(

                prediction?.prediction?.faw_attack_probability ??

                  prediction?.faw_attack_probability

              );

              const prevention = Number.isFinite(attackProbability)

                ? attackProbability < 30

                  ? {

                      level: "LOW",

                      range: "Below 30%",

                      color: "#166534",

                      background: "#f0fdf4",

                      actions: [

                        "Keep a regular scouting schedule and check maize whorls and young leaves for early signs of FAW.",

                        "Remove and safely manage heavily damaged plant parts where appropriate; avoid spreading larvae between plants.",

                        "Maintain field hygiene and follow locally recommended crop and weed management practices.",

                        "Recheck the crop after weather changes or when fresh feeding damage appears."

                      ]

                    }

                  : attackProbability < 60

                    ? {

                        level: "MEDIUM",

                        range: "30%–59.9%",

                        color: "#92400e",

                        background: "#fffbeb",

                        actions: [

                          "Increase field scouting and inspect whorls, leaf undersides, egg masses, and fresh window-pane feeding damage.",

                          "Remove egg masses or small larvae when practical and safe, following local agricultural guidance.",

                          "Encourage natural enemies and avoid unnecessary broad-spectrum pesticide use.",

                          "Record affected areas and seek advice from a local agricultural officer if signs of infestation increase."

                        ]

                      }

                    : {

                        level: "HIGH",

                        range: "60%–100%",

                        color: "#b91c1c",

                        background: "#fef2f2",

                        actions: [

                          "Inspect the field as soon as possible, prioritizing young plants, whorls, and areas with fresh damage.",

                          "Confirm the presence of FAW eggs or larvae before taking control action; prediction percentage alone does not confirm infestation.",

                          "Consult local agricultural extension guidance for integrated pest management options suited to the crop stage and infestation level.",

                          "Continue frequent follow-up scouting and record whether damage is spreading after any intervention."

                        ]

                      }

                : null;

              return (

                <section

                  aria-live="polite"

                  style={{

                    marginTop: "20px",

                    padding: "22px 24px",

                    borderRadius: "12px",

                    background: "#f4fbf5",

                    borderLeft: "5px solid #176b3a",

                    color: "#1f2937"

                  }}

                >

                  <h3

                    style={{

                      margin: "0 0 8px",

                      color: "#202820",

                      fontSize: "1.15rem"

                    }}

                  >

                    FUTURE PEST PREVENTION

                  </h3>

                  <p style={{ margin: "0 0 14px", fontWeight: 600 }}>

                    {prevention

                      ? `${prevention.level} RISK PREVENTION (${prevention.range})`

                      : "Prevention guidance unavailable"}

                  </p>

                  {prevention ? (

                    <div className="recommendation-action-grid">

                      {prevention.actions.map((action, index) => {

                        const actionTitles = [

                          "Scout regularly",

                          "Manage plant damage",

                          "Protect field health",

                          "Review changing conditions",

                        ];

                        return (

                          <article

                            className="recommendation-action-card"

                            key={`prevention-${prevention.level}-${index}`}

                          >

                            <span className="recommendation-action-icon" aria-hidden="true">

                              {["👁️", "✂️", "🌿", "↻"][index]}

                            </span>

                            <div className="recommendation-action-copy">

                              <h4>{actionTitles[index] || `Prevention ${index + 1}`}</h4>

                              <p>{action}</p>

                            </div>

                            <span className="recommendation-action-number">

                              {String(index + 1).padStart(2, "0")}

                            </span>

                          </article>

                        );

                      })}

                    </div>

                  ) : (

                    <p>

                      A valid FAW attack probability was not returned. Inspect field conditions and follow local agricultural guidance.

                    </p>

                  )}

                  <p

                    style={{

                      margin: "14px 0 0",

                      fontSize: "0.85rem",

                      lineHeight: 1.5,

                      opacity: 0.8

                    }}

                  >

                    Percentage bands are guidance thresholds for this card, not independently validated disease thresholds. Confirm field conditions and follow local agricultural recommendations.

                  </p>

                </section>

              );

            })()}

{/* ==================================================

                WEATHER

                ================================================== */}

            <div

              className="hybrid-detail-grid"

              style={{

                marginTop:

                  "16px",

              }}

            >

              <WeatherPanel

                title="TODAY'S WEATHER"

                values={

                  prediction

                    .selected_date_weather

                }

              />

              <WeatherPanel

                title="LSTM WEATHER FORECAST"

                values={

                  prediction

                    .weather

                }

              />

            </div>

</div>

        )}

      </section>

    </div>

  );

}

export default Prediction;
