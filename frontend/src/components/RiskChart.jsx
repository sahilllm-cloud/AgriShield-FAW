import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const data = [
  { day: "Mon", risk: 20 },
  { day: "Tue", risk: 35 },
  { day: "Wed", risk: 45 },
  { day: "Thu", risk: 40 },
  { day: "Fri", risk: 60 },
  { day: "Sat", risk: 70 },
  { day: "Sun", risk: 65 },
];

function RiskChart() {
  return (
    <div className="risk-chart">
      <h2>FAW Risk Trend</h2>

      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="day" />
          <YAxis />
          <Tooltip />
          <Line
            type="monotone"
            dataKey="risk"
            stroke="#1b5e20"
            strokeWidth={3}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default RiskChart;