
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

function RiskChart({ history = [], isLoading = false }) {
  const data = [...history]
    .filter((item) => {
      const value = Number(item.faw_attack_probability);

      return (
        item.faw_attack_probability !== null &&
        item.faw_attack_probability !== undefined &&
        item.faw_attack_probability !== "" &&
        Number.isFinite(value)
      );
    })
    .map((item, index) => {
      const dateValue = item.created_at || item.date || null;
      const date = dateValue ? new Date(dateValue) : null;
      const validDate =
        date && !Number.isNaN(date.getTime()) ? date : null;

      return {
        originalIndex: index,
        timestamp: validDate ? validDate.getTime() : null,
        fullDate: validDate
          ? validDate.toLocaleString("en-IN", {
              day: "2-digit",
              month: "short",
              year: "numeric",
              hour: "2-digit",
              minute: "2-digit",
            })
          : "Date unavailable",
        day: validDate
          ? validDate.toLocaleDateString("en-IN", {
              day: "2-digit",
              month: "short",
            })
          : "Date unavailable",
        risk: Number(item.faw_attack_probability),
        riskLevel: item.risk || item.risk_level || "Unknown",
        filename: item.filename || "Unknown image",
      };
    })
    // Sort dated predictions oldest to newest.
    .sort((a, b) => {
      if (a.timestamp === null && b.timestamp === null) {
        return a.originalIndex - b.originalIndex;
      }
      if (a.timestamp === null) return 1;
      if (b.timestamp === null) return -1;
      return a.timestamp - b.timestamp;
    })
    // Distinguish predictions made on the same day.
    .map((item, index, sortedData) => {
      const sameDayCount = sortedData.filter(
        (other) => other.day === item.day
      ).length;

      const occurrence = sortedData
        .slice(0, index + 1)
        .filter((other) => other.day === item.day).length;

      return {
        ...item,
        chartLabel:
          item.day === "Date unavailable"
            ? `Prediction ${index + 1}`
            : sameDayCount > 1
              ? `${item.day} (${occurrence})`
              : item.day,
      };
    });

  return (
    <div className="risk-chart">
      <h2>FAW Risk Trend</h2>

      {isLoading ? (
        <p>Loading your risk history...</p>
      ) : data.length === 0 ? (
        <p>
          No saved risk predictions yet. Run a prediction to see your trend.
        </p>
      ) : (
        <ResponsiveContainer width="100%" height={300}>
          <LineChart
            data={data}
            margin={{ top: 10, right: 18, left: 8, bottom: 8 }}
          >
            <CartesianGrid strokeDasharray="3 3" />

            <XAxis
              dataKey="chartLabel"
              tick={{ fontSize: 12 }}
              minTickGap={12}
            />

            <YAxis
              domain={[0, 100]}
              ticks={[0, 20, 40, 60, 80, 100]}
              tickFormatter={(value) => `${value}%`}
            />

            <Tooltip
              labelFormatter={(_, payload) =>
                payload?.[0]?.payload?.fullDate || "Date unavailable"
              }
              formatter={(value, name) => [
                `${Number(value).toFixed(2)}%`,
                name,
              ]}
              content={({ active, payload, label }) => {
                if (!active || !payload?.length) return null;

                const prediction = payload[0].payload;

                return (
                  <div
                    style={{
                      background: "#ffffff",
                      border: "1px solid #dcebe0",
                      borderRadius: "10px",
                      padding: "12px 14px",
                      boxShadow: "0 5px 18px rgba(13, 68, 38, 0.10)",
                    }}
                  >
                    <strong style={{ color: "#155c35" }}>
                      {prediction.fullDate}
                    </strong>
                    <p style={{ margin: "8px 0 4px" }}>
                      FAW Risk:{" "}
                      <strong>
                        {prediction.risk.toFixed(2)}%
                      </strong>
                    </p>
                    <p style={{ margin: "4px 0" }}>
                      Risk Level: {prediction.riskLevel}
                    </p>
                    <p
                      style={{
                        margin: "4px 0 0",
                        maxWidth: "220px",
                        overflowWrap: "anywhere",
                      }}
                    >
                      Image: {prediction.filename}
                    </p>
                  </div>
                );
              }}
            />

            <Line
              type="monotone"
              dataKey="risk"
              name="FAW Risk"
              stroke="#176b3a"
              strokeWidth={3}
              dot={{
                r: 4,
                fill: "#ffffff",
                stroke: "#176b3a",
                strokeWidth: 2,
              }}
              activeDot={{
                r: 6,
                fill: "#176b3a",
                stroke: "#ffffff",
                strokeWidth: 2,
              }}
            />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

export default RiskChart;
