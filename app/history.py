
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.encoders import jsonable_encoder
from jose import JWTError, jwt
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth import SECRET_KEY, ALGORITHM, get_db
from database.auth_models import User
from database.history_models import ScanHistory

router = APIRouter(prefix="/api/history", tags=["History"])


class HistoryRequest(BaseModel):
    prediction: dict[str, Any]


def get_optional_user(
    authorization: str | None,
    db: Session,
):
    if not authorization:
        return None

    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    token = authorization.split(" ", 1)[1]

    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
        )
        user_id = int(payload["sub"])
    except (JWTError, ValueError, TypeError, KeyError):
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")

    return user


@router.post("")
def save_history(
    request: HistoryRequest,
    db: Session = Depends(get_db),
    authorization: str | None = Header(default=None),
    guest_id: str | None = Header(default=None, alias="X-Guest-ID"),
):
    user = get_optional_user(authorization, db)

    if user is None and not guest_id:
        raise HTTPException(
            status_code=400,
            detail="Guest ID is required for guest history",
        )

    result = request.prediction
    image = result.get("image") or {}
    prediction = result.get("prediction") or {}

    try:
        record = ScanHistory(
            user_id=user.id if user else None,
            guest_id=guest_id if user is None else None,
            analysis_type="Hybrid FAW Prediction",
            image_filename=str(image.get("filename") or "unknown"),
            predicted_class=str(image.get("predicted_class") or "unknown"),
            healthy_probability=float(image.get("healthy_probability", 0)),
            infected_probability=float(image.get("infected_probability", 0)),
            faw_attack_probability=float(
                prediction.get(
                    "faw_attack_probability",
                    result.get("faw_attack_probability", 0),
                )
            ),
            risk_level=str(
                prediction.get("risk_level", result.get("risk_level", "Unknown"))
            ),
            selected_date=str(result.get("date") or ""),
            details=jsonable_encoder(result),
        )

        db.add(record)
        db.commit()
        db.refresh(record)

        return {
            "message": "Prediction saved to history",
            "id": record.id,
        }

    except (TypeError, ValueError) as exc:
        db.rollback()
        raise HTTPException(
            status_code=422,
            detail=f"Invalid prediction data: {exc}",
        )
    except Exception:
        db.rollback()
        raise


@router.get("")
def get_history(
    db: Session = Depends(get_db),
    authorization: str | None = Header(default=None),
    guest_id: str | None = Header(default=None, alias="X-Guest-ID"),
):
    user = get_optional_user(authorization, db)

    query = db.query(ScanHistory)

    if user:
        query = query.filter(ScanHistory.user_id == user.id)
    else:
        if not guest_id:
            raise HTTPException(
                status_code=400,
                detail="Guest ID is required to retrieve guest history",
            )
        query = query.filter(
            ScanHistory.user_id.is_(None),
            ScanHistory.guest_id == guest_id,
        )

    records = query.order_by(ScanHistory.created_at.desc()).all()

    return {
        "total_scans": len(records),
        "history": [
            {
                "id": record.id,
                "date": record.selected_date,
                "created_at": record.created_at,
                "type": record.analysis_type,
                "filename": record.image_filename,
                "result": record.predicted_class,
                "confidence": record.infected_probability,
                "healthy_probability": record.healthy_probability,
                "infected_probability": record.infected_probability,
                "faw_attack_probability": record.faw_attack_probability,
                "risk": record.risk_level,
                "details": record.details,
            }
            for record in records
        ],
    }
