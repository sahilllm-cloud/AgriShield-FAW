function History() {
  const historyData = [
    {
      date: "13 Aug 2026",
      type: "Image Detection",
      result: "FAW Detected",
      confidence: "94.5%",
      risk: "High",
      icon: "◉",
    },
    {
      date: "12 Aug 2026",
      type: "Outbreak Prediction",
      result: "High Risk",
      confidence: "-",
      risk: "High",
      icon: "◈",
    },
  ];

  return (
    <div className="history-page">

      {/* =====================================================
          HEADER
          ===================================================== */}

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


      {/* =====================================================
          SUMMARY STRIP
          ===================================================== */}

      <section className="history-summary">

        <div>
          <span>TOTAL ANALYSES</span>
          <strong>{historyData.length}</strong>
        </div>

        <div>
          <span>FAW DETECTIONS</span>
          <strong>
            {
              historyData.filter(
                (item) =>
                  item.type === "Image Detection"
              ).length
            }
          </strong>
        </div>

        <div>
          <span>HIGH RISK EVENTS</span>
          <strong>
            {
              historyData.filter(
                (item) =>
                  item.risk === "High"
              ).length
            }
          </strong>
        </div>

        <div className="history-live">
          <span className="history-live-dot"></span>
          LIVE RECORD
        </div>

      </section>


      {/* =====================================================
          HISTORY PANEL
          ===================================================== */}

      <section className="history-panel">

        <div className="history-panel-header">

          <div>
            <span>
              FIELD INTELLIGENCE
            </span>

            <h2>
              Recent Analysis
            </h2>
          </div>

          <div className="history-count">
            {historyData.length} RECORDS
          </div>

        </div>


        {/* =================================================
            TABLE
            ================================================= */}

        <div className="history-table">

          <table>

            <thead>
              <tr>

                <th>
                  DATE
                </th>

                <th>
                  ANALYSIS TYPE
                </th>

                <th>
                  RESULT
                </th>

                <th>
                  CONFIDENCE
                </th>

                <th>
                  RISK
                </th>

              </tr>
            </thead>


            <tbody>

              {historyData.map((item, index) => (

                <tr key={index}>

                  {/* DATE */}

                  <td>

                    <div className="history-date">

                      <span className="history-date-icon">
                        {item.icon}
                      </span>

                      <div>
                        <strong>
                          {item.date}
                        </strong>

                        <small>
                          Analysis #{String(index + 1).padStart(2, "0")}
                        </small>
                      </div>

                    </div>

                  </td>


                  {/* TYPE */}

                  <td>

                    <span
                      className={
                        item.type === "Image Detection"
                          ? "history-type detection-type"
                          : "history-type prediction-type"
                      }
                    >
                      {item.type}
                    </span>

                  </td>


                  {/* RESULT */}

                  <td>

                    <strong className="history-result">
                      {item.result}
                    </strong>

                  </td>


                  {/* CONFIDENCE */}

                  <td>

                    <span className="history-confidence">
                      {item.confidence}
                    </span>

                  </td>


                  {/* RISK */}

                  <td>

                    <span className="history-risk">
                      <span></span>
                      {item.risk}
                    </span>

                  </td>

                </tr>

              ))}

            </tbody>

          </table>

        </div>

      </section>

    </div>
  );
}

export default History;