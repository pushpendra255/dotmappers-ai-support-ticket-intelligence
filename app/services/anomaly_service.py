from typing import Any, Dict, Optional

import pandas as pd

from app.db.database import fetch_all


def _filter_date_range(df: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    if start_date:
        df = df[df["created_at"].dt.date >= pd.Timestamp(start_date).date()]
    if end_date:
        df = df[df["created_at"].dt.date <= pd.Timestamp(end_date).date()]
    return df


def detect_anomalies(start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
    """Run deterministic anomaly rules, optionally scoped by ticket creation date."""
    df = fetch_all()
    df["created_at"] = pd.to_datetime(df["created_at"])

    dataset_reference_time = df["created_at"].max()
    scoped_df = _filter_date_range(df, start_date, end_date)

    # Keep the statistical threshold deterministic and based on the complete dataset.
    # This makes a weekly query comparable with the full-system anomaly rule.
    resolved_times = df["resolution_time_hrs"].dropna()
    if resolved_times.empty:
        threshold = 0.0
    else:
        q1 = resolved_times.quantile(0.25)
        q3 = resolved_times.quantile(0.75)
        iqr = q3 - q1
        threshold = float(q3 + 1.5 * iqr)

    anomalies = []

    # Rule 1: statistically long resolution time.
    long_df = scoped_df[
        scoped_df["resolution_time_hrs"].notna()
        & (scoped_df["resolution_time_hrs"] > threshold)
    ]
    for _, row in long_df.iterrows():
        resolution = float(row["resolution_time_hrs"])
        severity = "high" if resolution > threshold * 1.5 else "medium"
        anomalies.append(
            {
                "ticket_id": row["ticket_id"],
                "anomaly_type": "long_resolution_time",
                "severity": severity,
                "reason": (
                    f"Resolution time {resolution:.1f}h exceeds the IQR threshold "
                    f"of {threshold:.1f}h."
                ),
                "details": {"resolution_time_hrs": resolution},
            }
        )

    # Rule 2: unresolved High/Critical tickets older than 24h relative to the
    # historical dataset reference time.
    age_hours = (dataset_reference_time - scoped_df["created_at"]).dt.total_seconds() / 3600
    mask = (
        scoped_df["priority"].isin(["High", "Critical"])
        & scoped_df["status"].isin(["Open", "Escalated"])
        & (age_hours > 24)
    )
    old_df = scoped_df[mask]
    for _, row in old_df.iterrows():
        age = float(age_hours.loc[row.name])
        severity = "critical" if row["priority"] == "Critical" else "high"
        anomalies.append(
            {
                "ticket_id": row["ticket_id"],
                "anomaly_type": "old_unresolved_high_priority",
                "severity": severity,
                "reason": (
                    f"{row['priority']} ticket remains {row['status'].lower()} and is "
                    f"{age:.1f}h old relative to the dataset reference time."
                ),
                "details": {
                    "priority": row["priority"],
                    "status": row["status"],
                    "age_hours": round(age, 1),
                },
            }
        )

    return {
        "reference_time": dataset_reference_time.strftime("%Y-%m-%d %H:%M:%S"),
        "long_resolution_threshold_hours": round(threshold, 2),
        "total_anomalies": len(anomalies),
        "anomalies": anomalies,
        "start_date": start_date,
        "end_date": end_date,
    }
