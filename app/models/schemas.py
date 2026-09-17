from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)


class QueryResponse(BaseModel):
    question: str
    answer: str
    plan: Dict[str, Any]
    data: Any


class AnomalyItem(BaseModel):
    ticket_id: str
    anomaly_type: str
    severity: str
    reason: str
    details: Dict[str, Any] = {}


class AnomalyResponse(BaseModel):
    reference_time: str
    long_resolution_threshold_hours: float
    total_anomalies: int
    anomalies: List[AnomalyItem]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
