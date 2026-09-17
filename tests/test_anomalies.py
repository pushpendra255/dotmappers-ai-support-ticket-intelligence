from app.db.database import initialize_database
from app.services.anomaly_service import detect_anomalies

def test_anomaly_detection():
    initialize_database()
    result = detect_anomalies()
    assert result["total_anomalies"] >= 0
    assert result["long_resolution_threshold_hours"] > 0
    assert isinstance(result["anomalies"], list)
