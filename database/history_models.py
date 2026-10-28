
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, JSON, String, Float, func
from sqlalchemy.orm import Mapped, mapped_column

from database.auth_models import AuthBase


class ScanHistory(AuthBase):
    __tablename__ = "scan_history"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True,
    )

    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    guest_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )

    analysis_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="Hybrid FAW Prediction",
    )

    image_filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    predicted_class: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    healthy_probability: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    infected_probability: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    faw_attack_probability: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    risk_level: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    selected_date: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
    )

    details: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
