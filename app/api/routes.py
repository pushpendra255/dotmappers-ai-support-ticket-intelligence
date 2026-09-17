from fastapi import APIRouter, HTTPException, Query
from app.models.schemas import QueryRequest, QueryResponse, AnomalyResponse
from app.services.query_service import answer_question
from app.services.anomaly_service import detect_anomalies
from app.db.database import database_is_healthy

router = APIRouter(prefix="/api")

@router.post("/query", response_model=QueryResponse)
def query_tickets(request: QueryRequest):
    try:
        return answer_question(request.question)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}")

@router.get("/anomalies", response_model=AnomalyResponse)
def anomalies(start_date: str | None = Query(default=None), end_date: str | None = Query(default=None)):
    try:
        return detect_anomalies(start_date=start_date, end_date=end_date)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {exc}")

@router.get("/health")
def health():
    return {
        "status": "healthy" if database_is_healthy() else "degraded",
        "database": "connected" if database_is_healthy() else "unavailable"
    }
