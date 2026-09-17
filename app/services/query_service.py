import re
from typing import Any, Dict, List

import pandas as pd

from app.db.database import TABLE_NAME, fetch_all, get_connection
from app.services.anomaly_service import detect_anomalies
from app.services.llm_service import build_date_context, create_query_plan


ALLOWED_FILTERS = {
    "category",
    "priority",
    "status",
    "agent_id",
}

ALLOWED_FIELDS = {
    "response_time_hrs",
    "resolution_time_hrs",
    "customer_rating",
}

ALLOWED_OPERATIONS = {
    "count",
    "average",
    "min",
    "max",
    "sum",
    "list",
    "group_by",
}

ALLOWED_GROUPS = {
    "agent_id",
    "category",
    "priority",
    "status",
}


# -------------------------------------------------------------------
# PLAN NORMALIZATION
# -------------------------------------------------------------------

def _normalize_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize the structured query plan before execution.
    """

    if not isinstance(plan, dict):
        raise ValueError("Query plan must be an object.")

    plan = dict(plan)

    operation = str(plan.get("operation") or "").lower().strip()

    if operation not in ALLOWED_OPERATIONS:
        raise ValueError(f"Unsupported operation: {operation}")

    # ---------------------------------------------------------------
    # FIX: COUNT + GROUP BY
    # ---------------------------------------------------------------
    # Some natural-language questions like:
    # "How many tickets are there in each category?"
    # may return:
    # operation = count
    # group_by = category
    #
    # Treat this as a grouped count query.
    if operation == "count" and plan.get("group_by"):
        operation = "group_by"
        plan["operation"] = "group_by"
        plan["metric"] = "count"

    # -----------------------------
    # Filters
    # -----------------------------

    filters = plan.get("filters") or {}

    if not isinstance(filters, dict):
        raise ValueError("filters must be an object.")

    # Remove null / empty filters.
    filters = {
        key: value
        for key, value in filters.items()
        if value is not None and value != "null" and value != ""
    }

    unknown_filters = set(filters) - ALLOWED_FILTERS

    if unknown_filters:
        raise ValueError(
            f"Unsupported filter fields: {sorted(unknown_filters)}"
        )

    plan["filters"] = filters

    # -----------------------------
    # Numeric field
    # -----------------------------

    field = plan.get("field")

    if operation in {"average", "min", "max", "sum"}:
        if field not in ALLOWED_FIELDS:
            raise ValueError(
                f"{operation} requires a supported numeric field."
            )
    else:
        plan["field"] = None

    # -----------------------------
    # Grouping
    # -----------------------------

    if operation == "group_by":

        group_by = plan.get("group_by")

        if group_by not in ALLOWED_GROUPS:
            raise ValueError(
                f"Unsupported group_by field: {group_by}"
            )

        metric = plan.get("metric") or "count"

        if metric != "count" and metric not in ALLOWED_FIELDS:
            raise ValueError(
                f"Unsupported group metric: {metric}"
            )

        plan["metric"] = metric

    else:
        plan["group_by"] = None
        plan["metric"] = None

    # -----------------------------
    # Threshold
    # -----------------------------

    if plan.get("threshold_hours") is not None:
        try:
            threshold = float(plan["threshold_hours"])

            if threshold < 0:
                raise ValueError

            plan["threshold_hours"] = threshold

        except (TypeError, ValueError) as exc:
            raise ValueError(
                "threshold_hours must be a non-negative number."
            ) from exc

    else:
        plan["threshold_hours"] = None

    # -----------------------------
    # Dates
    # -----------------------------

    plan["start_date"] = plan.get("start_date") or None
    plan["end_date"] = plan.get("end_date") or None

    # -----------------------------
    # Limit
    # -----------------------------

    try:
        limit = int(plan.get("limit") or 50)
    except (TypeError, ValueError):
        limit = 50

    plan["limit"] = min(max(limit, 1), 100)

    # -----------------------------
    # Order
    # -----------------------------

    order = str(plan.get("order") or "asc").lower().strip()

    if order not in {"asc", "desc"}:
        order = "asc"

    plan["order"] = order

    plan["operation"] = operation

    return plan


# -------------------------------------------------------------------
# SQL EXECUTION
# -------------------------------------------------------------------

def execute_plan(plan: Dict[str, Any]) -> Any:
    """
    Execute a validated query plan against SQLite.
    """

    plan = _normalize_plan(plan)

    operation = plan["operation"]
    filters = plan["filters"]

    where: List[str] = []
    params: List[Any] = []

    # ---------------------------------------------------------------
    # Filters
    # ---------------------------------------------------------------

    for key, value in filters.items():

        if isinstance(value, list):

            if not value:
                continue

            placeholders = ",".join(["?"] * len(value))

            where.append(
                f"{key} IN ({placeholders})"
            )

            params.extend(value)

        else:

            where.append(
                f"{key} = ?"
            )

            params.append(value)

    # ---------------------------------------------------------------
    # Date filters
    # ---------------------------------------------------------------

    start_date = plan.get("start_date")
    end_date = plan.get("end_date")

    if start_date:
        where.append(
            "date(created_at) >= date(?)"
        )
        params.append(start_date)

    if end_date:
        where.append(
            "date(created_at) <= date(?)"
        )
        params.append(end_date)

    # ---------------------------------------------------------------
    # Resolution threshold
    # ---------------------------------------------------------------

    threshold = plan.get("threshold_hours")

    if threshold is not None:

        # Matches:
        # 1. unresolved tickets -> resolution_time_hrs IS NULL
        # 2. resolved late -> resolution_time_hrs > threshold
        where.append(
            "(resolution_time_hrs IS NULL OR resolution_time_hrs > ?)"
        )

        params.append(float(threshold))

    # ---------------------------------------------------------------
    # WHERE SQL
    # ---------------------------------------------------------------

    where_sql = ""

    if where:
        where_sql = " WHERE " + " AND ".join(where)

    # ---------------------------------------------------------------
    # Database
    # ---------------------------------------------------------------

    with get_connection() as conn:

        # ===========================================================
        # COUNT
        # ===========================================================

        if operation == "count":

            row = conn.execute(
                f"""
                SELECT COUNT(*) AS value
                FROM {TABLE_NAME}
                {where_sql}
                """,
                params,
            ).fetchone()

            return {
                "count": int(row["value"])
            }

        # ===========================================================
        # AGGREGATIONS
        # ===========================================================

        if operation in {
            "average",
            "min",
            "max",
            "sum",
        }:

            field = plan["field"]

            functions = {
                "average": "AVG",
                "min": "MIN",
                "max": "MAX",
                "sum": "SUM",
            }

            sql_function = functions[operation]

            row = conn.execute(
                f"""
                SELECT {sql_function}({field}) AS value
                FROM {TABLE_NAME}
                {where_sql}
                """,
                params,
            ).fetchone()

            value = row["value"]

            return {
                "operation": operation,
                "field": field,
                "value": (
                    None
                    if value is None
                    else round(float(value), 2)
                ),
            }

        # ===========================================================
        # LIST
        # ===========================================================

        if operation == "list":

            rows = conn.execute(
                f"""
                SELECT
                    ticket_id,
                    created_at,
                    category,
                    priority,
                    status,
                    response_time_hrs,
                    resolution_time_hrs,
                    agent_id,
                    customer_rating,
                    issue_summary
                FROM {TABLE_NAME}
                {where_sql}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                params + [plan["limit"]],
            ).fetchall()

            return [
                dict(row)
                for row in rows
            ]

        # ===========================================================
        # GROUP BY / RANKING
        # ===========================================================

        if operation == "group_by":

            group_by = plan["group_by"]
            metric = plan["metric"]

            if metric == "count":

                metric_sql = "COUNT(*)"

            else:

                metric_sql = f"AVG({metric})"

            order_sql = (
                "DESC"
                if plan["order"] == "desc"
                else "ASC"
            )

            rows = conn.execute(
                f"""
                SELECT
                    {group_by} AS group_value,
                    {metric_sql} AS value
                FROM {TABLE_NAME}
                {where_sql}
                GROUP BY {group_by}
                HAVING value IS NOT NULL
                ORDER BY value {order_sql}, group_value ASC
                LIMIT ?
                """,
                params + [plan["limit"]],
            ).fetchall()

            result = []

            for row in rows:

                value = row["value"]

                if value is None:
                    continue

                if metric == "count":

                    numeric_value = int(value)

                else:

                    numeric_value = round(
                        float(value),
                        2,
                    )

                result.append(
                    {
                        "group": row["group_value"],
                        "value": numeric_value,
                    }
                )

            return result

    raise ValueError(
        "Could not execute query plan."
    )


# -------------------------------------------------------------------
# ANOMALY DETECTION
# -------------------------------------------------------------------

def _is_anomaly_question(question: str) -> bool:
    """
    Detect anomaly-related questions and route them directly
    to the deterministic anomaly detector.
    """

    q = (question or "").lower()

    has_anomaly_word = bool(
        re.search(
            r"\b(anomal(?:y|ies)|outlier|unusually|abnormally|abnormal)\b",
            q,
        )
    )

    has_resolution_context = bool(
        re.search(
            r"\b(resolution|resolved|resolution\s+time|ticket|tickets)\b",
            q,
        )
    )

    return (
        has_anomaly_word
        and has_resolution_context
    )


def _anomaly_date_range(
    question: str,
    reference_date: str,
):
    """
    Resolve anomaly date ranges deterministically.
    """

    q = (question or "").lower()

    ctx = build_date_context(reference_date)

    if re.search(r"\bthis\s+week\b", q):

        return (
            ctx["this_week_start"],
            ctx["this_week_end"],
        )

    if re.search(r"\blast\s+week\b", q):

        return (
            ctx["last_week_start"],
            ctx["last_week_end"],
        )

    if re.search(r"\bthis\s+month\b", q):

        return (
            ctx["this_month_start"],
            ctx["this_month_end"],
        )

    if re.search(r"\blast\s+month\b", q):

        return (
            ctx["last_month_start"],
            ctx["last_month_end"],
        )

    if re.search(r"\b(today|today's)\b", q):

        return (
            ctx["today"],
            ctx["today"],
        )

    if re.search(r"\byesterday\b", q):

        return (
            ctx["yesterday"],
            ctx["yesterday"],
        )

    return None, None


# -------------------------------------------------------------------
# ANSWER GENERATION
# -------------------------------------------------------------------

def answer_question(question: str) -> Dict[str, Any]:
    """
    Main entry point for natural-language ticket analytics.
    """

    question = (question or "").strip()

    if len(question) < 3:
        raise ValueError(
            "Question must be at least 3 characters long."
        )

    # ---------------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------------

    df = fetch_all()

    if df.empty:
        raise ValueError(
            "No ticket data is available."
        )

    reference_date = (
        pd.to_datetime(df["created_at"])
        .max()
        .strftime("%Y-%m-%d")
    )

    # ---------------------------------------------------------------
    # ANOMALY QUERY
    # ---------------------------------------------------------------

    if _is_anomaly_question(question):

        start_date, end_date = _anomaly_date_range(
            question,
            reference_date,
        )

        result = detect_anomalies(
            start_date=start_date,
            end_date=end_date,
        )

        # Resolution-time-specific questions should return
        # only long-resolution anomalies.
        if re.search(
            r"\bresolution[-\s]+time(?:s)?\b",
            question.lower(),
        ):
            resolution_anomalies = [
                anomaly
                for anomaly in result.get("anomalies", [])
                if anomaly.get("anomaly_type") == "long_resolution_time"
            ]

            result = dict(result)
            result["anomalies"] = resolution_anomalies
            result["total_anomalies"] = len(resolution_anomalies)

        scope = ""

        if start_date and end_date:

            scope = (
                f" from {start_date} to {end_date}"
            )

        if result["total_anomalies"] == 0:

            answer = (
                f"No anomalies were detected{scope}."
            )

        else:

            answer = (
                f"Found "
                f"{result['total_anomalies']} "
                f"anomalies{scope}."
            )

        plan = {
            "operation": "anomaly_detection",
            "field": (
                "resolution_time_hrs"
                if "resolution" in question.lower()
                else None
            ),
            "group_by": None,
            "metric": None,
            "filters": {},
            "start_date": start_date,
            "end_date": end_date,
            "threshold_hours": result.get(
                "long_resolution_threshold_hours"
            ),
            "order": None,
            "limit": None,
        }

        return {
            "question": question,
            "answer": answer,
            "plan": plan,
            "data": result,
        }

    # ---------------------------------------------------------------
    # NORMAL NL QUERY
    # ---------------------------------------------------------------

    plan = create_query_plan(
        question,
        reference_date,
    )

    # Normalize the plan before execution so count + group_by
    # is consistently treated as a grouped count query.
    plan = _normalize_plan(plan)

    result = execute_plan(plan)

    operation = plan["operation"]

    # ===============================================================
    # COUNT
    # ===============================================================

    if operation == "count":

        answer = (
            f"There are "
            f"{result['count']} "
            f"matching tickets."
        )

    # ===============================================================
    # AVERAGE
    # ===============================================================

    elif operation == "average":

        value = result["value"]

        if value is None:

            answer = (
                "No matching numeric data was found."
            )

        else:

            answer = (
                f"The average "
                f"{plan['field']} "
                f"is {value}."
            )

    # ===============================================================
    # MIN
    # ===============================================================

    elif operation == "min":

        value = result["value"]

        if value is None:

            answer = (
                "No matching numeric data was found."
            )

        else:

            answer = (
                f"The minimum "
                f"{plan['field']} "
                f"is {value}."
            )

    # ===============================================================
    # MAX
    # ===============================================================

    elif operation == "max":

        value = result["value"]

        if value is None:

            answer = (
                "No matching numeric data was found."
            )

        else:

            answer = (
                f"The maximum "
                f"{plan['field']} "
                f"is {value}."
            )

    # ===============================================================
    # SUM
    # ===============================================================

    elif operation == "sum":

        value = result["value"]

        if value is None:

            answer = (
                "No matching numeric data was found."
            )

        else:

            answer = (
                f"The total "
                f"{plan['field']} "
                f"is {value}."
            )

    # ===============================================================
    # GROUP BY / RANKING
    # ===============================================================

    elif operation == "group_by":

        if not result:

            answer = (
                "No matching grouped data was found."
            )

        else:

            top = result[0]

            group_name = str(
                top["group"]
            )

            value = top["value"]

            metric = (
                plan.get("metric")
                or "count"
            )

            direction = (
                "highest"
                if plan["order"] == "desc"
                else "lowest"
            )

            # With limit=1, the result is intentionally
            # exactly one ranked group.
            answer = (
                f"{group_name} has the "
                f"{direction} {metric} "
                f"value at {value}."
            )

    # ===============================================================
    # LIST
    # ===============================================================

    elif operation == "list":

        answer = (
            f"Found "
            f"{len(result)} "
            f"matching tickets."
        )

    else:

        answer = (
            f"The {operation} "
            f"value is "
            f"{result.get('value')}."
        )

    return {
        "question": question,
        "answer": answer,
        "plan": plan,
        "data": result,
    }