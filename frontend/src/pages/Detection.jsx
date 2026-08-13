import { useState } from "react";

function Detection() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);

  const handleImageChange = (event) => {
    const file = event.target.files[0];

    if (!file) return;

    setSelectedImage(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
  };

  const handleDetection = () => {
    if (!selectedImage) return;

    // Temporary frontend result.
    // Later this will call the Swin Transformer backend.
    setResult({
      label: "FAW Detected",
      confidence: "94.5%",
      model: "Swin Transformer",
    });
  };

  return (
    <div className="detection-page">

      {/* =====================================================
          PAGE HEADER
          ===================================================== */}

      <section className="detection-header">

        <span className="detection-tag">
          AI VISION ENGINE
        </span>

        <h1>
          Fall Armyworm
          <br />
          Detection
        </h1>

        <p>
          Upload a maize leaf image and let AgriShield
          analyze it using the Swin Transformer.
        </p>

      </section>


      {/* =====================================================
          SCANNER
          ===================================================== */}

      <section className="detection-scanner">

        <div className="scanner-top">

          <div>
            <span className="scanner-label">
              IMAGE ANALYSIS
            </span>

            <h2>
              Maize Leaf Scanner
            </h2>
          </div>

          <div className="scanner-status">
            <span></span>
            AI READY
          </div>

        </div>


        {/* Upload area */}

        <div
          className={`scanner-area ${
            preview ? "has-image" : ""
          }`}
        >

          {!preview ? (

            <label className="upload-zone">

              <input
                type="file"
                accept="image/*"
                onChange={handleImageChange}
              />

              <div className="upload-icon">
                📷
              </div>

              <h3>
                Upload Maize Leaf
              </h3>

              <p>
                Drop an image here or click to browse
              </p>

              <span className="upload-format">
                JPG • JPEG • PNG
              </span>

            </label>

          ) : (

            <div className="image-scanner">

              <img
                src={preview}
                alt="Selected maize leaf"
              />

              <div className="scan-line"></div>

              <div className="corner top-left"></div>
              <div className="corner top-right"></div>
              <div className="corner bottom-left"></div>
              <div className="corner bottom-right"></div>

            </div>

          )}

        </div>


        {/* =================================================
            ACTIONS
            ================================================= */}

        <div className="scanner-actions">

          <label className="change-image">

            Choose Image

            <input
              type="file"
              accept="image/*"
              onChange={handleImageChange}
            />

          </label>

          <button
            className="detect-button"
            onClick={handleDetection}
            disabled={!selectedImage}
          >
            <span>✦</span>
            Detect FAW
          </button>

        </div>


        {/* =================================================
            RESULT
            ================================================= */}

        {result && (

          <div className="detection-result">

            <div className="result-header">
              <span>
                ANALYSIS COMPLETE
              </span>

              <div className="result-dot"></div>
            </div>


            <div className="result-main">

              <div className="result-icon">
                ⚠️
              </div>

              <div>

                <small>
                  DETECTION RESULT
                </small>

                <h2>
                  {result.label}
                </h2>

              </div>

            </div>


            <div className="result-details">

              <div>
                <span>CONFIDENCE</span>
                <strong>
                  {result.confidence}
                </strong>
              </div>

              <div>
                <span>MODEL</span>
                <strong>
                  {result.model}
                </strong>
              </div>

            </div>

          </div>

        )}

      </section>

    </div>
  );
}

export default Detection;