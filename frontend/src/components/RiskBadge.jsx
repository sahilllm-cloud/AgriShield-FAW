function RiskBadge({ level }) {
  const safeLevel = typeof level === "string" && level.trim() ? level : "Unknown";

  return (
    <span className={`risk-badge ${safeLevel.toLowerCase()}`}>
      {safeLevel}
    </span>
  );
}

export default RiskBadge;