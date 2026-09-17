import sqlite3
from pathlib import Path
import pandas as pd
from app.config import DATA_PATH, DB_PATH

TABLE_NAME = "support_tickets"

EXPECTED_COLUMNS = [
    "ticket_id", "created_at", "category", "priority", "status",
    "response_time_hrs", "resolution_time_hrs", "agent_id",
    "customer_rating", "issue_summary"
]

def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_database():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if df["ticket_id"].duplicated().any():
        raise ValueError("ticket_id must be unique.")

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    if df["created_at"].isna().any():
        raise ValueError("Invalid created_at values found.")

    for col in ["response_time_hrs", "resolution_time_hrs", "customer_rating"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if not df["customer_rating"].dropna().between(1, 5).all():
        raise ValueError("customer_rating must be between 1 and 5.")

    # Store datetimes in ISO text for SQLite.
    df["created_at"] = df["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S")

    with get_connection() as conn:
        df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON support_tickets(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_priority ON support_tickets(priority)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON support_tickets(category)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_agent ON support_tickets(agent_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON support_tickets(created_at)")
        conn.commit()

    return len(df)

def fetch_all():
    with get_connection() as conn:
        return pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)

def database_is_healthy():
    try:
        with get_connection() as conn:
            conn.execute("SELECT 1 FROM support_tickets LIMIT 1").fetchone()
        return True
    except Exception:
        return False
