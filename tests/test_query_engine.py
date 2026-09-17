from app.db.database import initialize_database
from app.services.query_service import execute_plan

def setup_module():
    initialize_database()

def test_count_open():
    result = execute_plan({
        "operation": "count",
        "filters": {"status": "Open"}
    })
    assert result["count"] == 111

def test_technical_average():
    result = execute_plan({
        "operation": "average",
        "field": "customer_rating",
        "filters": {"category": "Technical"}
    })
    assert result["value"] is not None
