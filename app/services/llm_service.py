import calendar
import json
import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import requests

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT


SYSTEM_PROMPT = """
You are the natural-language understanding layer for a customer support ticket analytics system.

Your ONLY job is to convert the user's question into a JSON query plan.
Do not answer the question.
Do not invent data.
Do not write SQL.

Dataset fields:
ticket_id (string)
created_at (datetime)
category: General | Billing | Technical
priority: Low | Medium | High | Critical
status: Open | Resolved | Escalated
response_time_hrs (number)
resolution_time_hrs (number; null for unresolved)
agent_id (string)
customer_rating (number 1-5; null for unresolved)
issue_summary (string)

Allowed operations:
count, average, min, max, sum, list, group_by

Allowed filter fields:
category, priority, status, agent_id

Allowed aggregate fields:
response_time_hrs, resolution_time_hrs, customer_rating

Allowed group_by:
agent_id, category, priority, status

CRITICAL RULES:

1. created_at is NEVER a filter field.
   Dates MUST use start_date and end_date.

2. Only add category, priority, status, or agent_id when explicitly mentioned
   by the user.

3. If the user does not mention a date/time period:
   start_date and end_date MUST be null.

4. "currently open" means:
   filters.status = "Open"

5. "unresolved" means:
   filters.status = ["Open", "Escalated"]

6. "show me all", "list", "display", "give me all", or similar wording
   means operation = "list".

7. "how many", "number of", or "count" means:
   operation = "count".

8. "which agent resolved the most tickets" means:
   operation = "group_by"
   group_by = "agent_id"
   metric = "count"
   filters.status = "Resolved"
   order = "desc"
   limit = 1

9. "highest" or "most" means order = "desc".
   "lowest" or "least" means order = "asc".

10. For "highest/lowest average X by Y":
    group_by = Y
    metric = X

11. "not resolved within N hours" should use:
    threshold_hours = N
    status = ["Open", "Escalated"]
    when the question is asking about currently unresolved tickets.

12. Do NOT use an operation named "filter".
    If the user asks to show/list/filter tickets, use operation = "list".

13. For a plain count:
    operation = "count"
    field = null

14. GROUP-BY COUNT:
    If the user asks "how many tickets are there in each category",
    "how many tickets in each category", or equivalent wording:
    operation = "group_by"
    group_by = "category"
    metric = "count"

    The same rule applies when the user asks for ticket counts
    for each agent, priority, or status:
    group_by = the explicitly mentioned field
    metric = "count"

    "each X" + "how many tickets" means group_by X with metric count.

15. RESOLUTION-TIME ANOMALY:
    If the user specifically asks about anomalies in:
    - resolution time
    - resolution times
    - long resolution time
    - unusually long resolution times

    then the anomaly query must target ONLY resolution-time anomalies.

    Do NOT include old_unresolved_high_priority anomalies
    for a resolution-time-specific question.

Return ONLY valid JSON:

{
  "operation": "...",
  "field": null,
  "group_by": null,
  "metric": null,
  "filters": {},
  "start_date": null,
  "end_date": null,
  "threshold_hours": null,
  "order": null,
  "limit": null
}
"""


ALLOWED_FILTERS = {
    "category",
    "priority",
    "status",
    "agent_id",
}

ALLOWED_CATEGORIES = {
    "general": "General",
    "billing": "Billing",
    "technical": "Technical",
}

ALLOWED_PRIORITIES = {
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "critical": "Critical",
}

ALLOWED_STATUSES = {
    "open": "Open",
    "resolved": "Resolved",
    "escalated": "Escalated",
}

ALLOWED_FIELDS = {
    "response_time_hrs",
    "resolution_time_hrs",
    "customer_rating",
}

ALLOWED_GROUPS = {
    "agent_id",
    "category",
    "priority",
    "status",
}


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract one JSON object from an Ollama response."""

    text = (text or "").strip()

    if not text:
        raise ValueError("LLM returned an empty response.")

    # Direct JSON
    try:
        parsed = json.loads(text)

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        pass

    # Remove markdown fences
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.I,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    ).strip()

    # Balanced JSON extraction
    start = text.find("{")

    if start == -1:
        raise ValueError("LLM did not return a JSON object.")

    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):

        ch = text[i]

        if in_string:

            if escaped:
                escaped = False

            elif ch == "\\":
                escaped = True

            elif ch == '"':
                in_string = False

            continue

        if ch == '"':
            in_string = True

        elif ch == "{":
            depth += 1

        elif ch == "}":
            depth -= 1

            if depth == 0:
                return json.loads(
                    text[start : i + 1]
                )

    raise ValueError("LLM returned incomplete JSON.")


def _parse_reference_date(reference_date: str) -> date:
    try:
        return datetime.strptime(
            reference_date,
            "%Y-%m-%d",
        ).date()

    except (TypeError, ValueError) as exc:
        raise ValueError(
            "reference_date must be in YYYY-MM-DD format."
        ) from exc


def build_date_context(reference_date: str) -> Dict[str, str]:
    """Build deterministic relative-date ranges."""

    ref = _parse_reference_date(reference_date)

    # Week
    this_week_start = (
        ref - timedelta(days=ref.weekday())
    )

    this_week_end = (
        this_week_start + timedelta(days=6)
    )

    last_week_start = (
        this_week_start - timedelta(days=7)
    )

    last_week_end = (
        this_week_start - timedelta(days=1)
    )

    # Month
    this_month_start = ref.replace(day=1)

    this_month_end = ref.replace(
        day=calendar.monthrange(
            ref.year,
            ref.month,
        )[1]
    )

    previous_month_last_day = (
        this_month_start - timedelta(days=1)
    )

    last_month_start = (
        previous_month_last_day.replace(day=1)
    )

    last_month_end = previous_month_last_day

    return {
        "today": ref.isoformat(),
        "yesterday": (
            ref - timedelta(days=1)
        ).isoformat(),

        "this_week_start": this_week_start.isoformat(),
        "this_week_end": this_week_end.isoformat(),

        "last_week_start": last_week_start.isoformat(),
        "last_week_end": last_week_end.isoformat(),

        "this_month_start": this_month_start.isoformat(),
        "this_month_end": this_month_end.isoformat(),

        "last_month_start": last_month_start.isoformat(),
        "last_month_end": last_month_end.isoformat(),
    }


def _date_range_from_question(
    question: str,
    reference_date: str,
) -> Tuple[Optional[str], Optional[str]]:

    q = question.lower().strip()

    ctx = build_date_context(reference_date)

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

    # ISO date
    match = re.search(
        r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b",
        q,
    )

    if match:

        try:

            d = date(
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
            ).isoformat()

            return d, d

        except ValueError:
            pass

    return None, None


def _question_explicit_filters(
    question: str,
) -> Dict[str, Any]:

    q = question.lower()

    filters: Dict[str, Any] = {}

    # Unresolved
    if re.search(
        r"\b(unresolved|not\s+resolved)\b",
        q,
    ):

        filters["status"] = [
            "Open",
            "Escalated",
        ]

    else:

        mentioned_statuses = []

        for word, label in ALLOWED_STATUSES.items():

            if re.search(
                rf"\b{re.escape(word)}\b",
                q,
            ):

                mentioned_statuses.append(label)

        if mentioned_statuses:

            if len(mentioned_statuses) == 1:
                filters["status"] = (
                    mentioned_statuses[0]
                )

            else:
                filters["status"] = (
                    mentioned_statuses
                )

    # Category
    for word, label in ALLOWED_CATEGORIES.items():

        if re.search(
            rf"\b{re.escape(word)}\b",
            q,
        ):

            filters["category"] = label
            break

    # Priority
    for word, label in ALLOWED_PRIORITIES.items():

        if re.search(
            rf"\b{re.escape(word)}\b",
            q,
        ):

            filters["priority"] = label
            break

    # Agent
    agent = re.search(
        r"\bAGT[-_]?\d{1,3}\b",
        question,
        flags=re.I,
    )

    if agent:

        raw = (
            agent.group(0)
            .upper()
            .replace("_", "-")
        )

        if not raw.startswith("AGT-"):

            raw = (
                "AGT-"
                + raw.split("-")[-1]
            )

        filters["agent_id"] = raw

    return filters


def _infer_operation_from_question(
    question: str,
) -> Optional[str]:

    q = question.lower().strip()

    # Show/list/display tickets
    if re.search(
        r"\b(show|list|display|give|fetch|get)\b.*\b(ticket|tickets)\b",
        q,
    ):

        return "list"

    # Explicit "all tickets"
    if re.search(
        r"\ball\b.*\btickets?\b",
        q,
    ):

        return "list"

    # Count
    if re.search(
        r"\b(how\s+many|number\s+of|count)\b",
        q,
    ):

        return "count"

    # Average
    if re.search(
        r"\baverage\b|\bavg\b",
        q,
    ):

        return "average"

    # Minimum
    if re.search(
        r"\bminimum\b|\bmin\b|\blowest\b|\bsmallest\b",
        q,
    ):

        if re.search(
            r"\b(customer\s+rating|response\s+time|resolution\s+time)\b",
            q,
        ):
            return "min"

    # Maximum
    if re.search(
        r"\bmaximum\b|\bmax\b|\bhighest\b|\blargest\b",
        q,
    ):

        if re.search(
            r"\b(customer\s+rating|response\s+time|resolution\s+time)\b",
            q,
        ):
            return "max"

    # Sum / total
    if re.search(
        r"\b(total|sum)\b",
        q,
    ):

        return "sum"

    return None


def _sanitize_plan(
    question: str,
    raw_plan: Dict[str, Any],
    reference_date: str,
) -> Dict[str, Any]:

    plan = dict(raw_plan or {})

    q = question.lower().strip()

    # ---------------------------------------------------------
    # OPERATION
    # ---------------------------------------------------------

    operation = str(
        plan.get("operation") or ""
    ).lower().strip()

    allowed_operations = {
        "count",
        "average",
        "min",
        "max",
        "sum",
        "list",
        "group_by",
    }

    # LLM sometimes returns "filter".
    # Convert it deterministically instead of failing.
    if operation == "filter":

        inferred = _infer_operation_from_question(
            question
        )

        operation = inferred or "list"

    # If LLM returns invalid/empty operation,
    # infer from the question.
    if operation not in allowed_operations:

        inferred = _infer_operation_from_question(
            question
        )

        if inferred:

            operation = inferred

        else:

            raise ValueError(
                f"Unsupported operation from LLM: {operation}"
            )

    plan["operation"] = operation

    # ---------------------------------------------------------
    # FILTERS
    # ---------------------------------------------------------

    explicit_filters = (
        _question_explicit_filters(question)
    )

    # Ignore any hallucinated LLM filters.
    # Only filters explicitly present in user question survive.
    filters: Dict[str, Any] = {}

    for key in ALLOWED_FILTERS:

        if key in explicit_filters:

            filters[key] = explicit_filters[key]

    plan["filters"] = filters

    # ---------------------------------------------------------
    # DATES
    # ---------------------------------------------------------

    start_date, end_date = (
        _date_range_from_question(
            question,
            reference_date,
        )
    )

    plan["start_date"] = start_date
    plan["end_date"] = end_date

    # ---------------------------------------------------------
    # FIELD
    # ---------------------------------------------------------

    field = plan.get("field")

    if field not in ALLOWED_FIELDS:

        field = None

    plan["field"] = field

    # ---------------------------------------------------------
    # GROUP BY
    # ---------------------------------------------------------

    group_by = plan.get("group_by")

    if group_by not in ALLOWED_GROUPS:

        group_by = None

    plan["group_by"] = group_by

    # ---------------------------------------------------------
    # METRIC
    # ---------------------------------------------------------

    metric = plan.get("metric")

    if (
        metric != "count"
        and metric not in ALLOWED_FIELDS
    ):

        metric = None

    plan["metric"] = metric
    
    # ---------------------------------------------------------
    # GROUP-BY COUNT: "HOW MANY TICKETS IN EACH X"
    # ---------------------------------------------------------

    if (
        re.search(
            r"\bhow\s+many\s+tickets?\b",
            q,
        )
        and re.search(
            r"\beach\s+(category|priority|status|agent)\b",
            q,
        )
    ):

        dimension_match = re.search(
            r"\beach\s+(category|priority|status|agent)\b",
            q,
        )

        dimension = (
            dimension_match.group(1)
            if dimension_match
            else None
        )

        group_map = {
            "agent": "agent_id",
            "category": "category",
            "priority": "priority",
            "status": "status",
        }

        if dimension:

            plan["operation"] = "group_by"
            plan["group_by"] = group_map[dimension]
            plan["metric"] = "count"
            plan["field"] = None
            plan["order"] = "asc"
            plan["limit"] = 100

    # ---------------------------------------------------------
    # RANKING: AGENT RESOLVED MOST
    # ---------------------------------------------------------

    if (
        re.search(r"\bwhich\s+agent\b", q)
        and re.search(
            r"\b(resolved|resolve|closed)\b",
            q,
        )
        and re.search(
            r"\b(most|highest|maximum|max)\b",
            q,
        )
    ):

        plan["operation"] = "group_by"
        plan["group_by"] = "agent_id"
        plan["metric"] = "count"
        plan["field"] = None
        plan["filters"]["status"] = "Resolved"
        plan["order"] = "desc"
        plan["limit"] = 1

    # ---------------------------------------------------------
    # LOWEST CUSTOMER RATING BY AGENT
    # ---------------------------------------------------------

    elif (
        re.search(r"\bwhich\s+agent\b", q)
        and re.search(
            r"\b(lowest|least|minimum|min)\b",
            q,
        )
        and re.search(
            r"\b(customer\s+rating|rating)\b",
            q,
        )
    ):

        plan["operation"] = "group_by"
        plan["group_by"] = "agent_id"
        plan["metric"] = "customer_rating"
        plan["field"] = None
        plan["order"] = "asc"
        plan["limit"] = 1

    # ---------------------------------------------------------
    # GENERAL GROUP RANKING
    # ---------------------------------------------------------

    elif (
        re.search(
            r"\bwhich\s+(category|priority|status|agent)\b",
            q,
        )
        and re.search(
            r"\b(highest|most|maximum|max|lowest|least|minimum|min)\b",
            q,
        )
    ):

        dimension_match = re.search(
            r"\b(category|priority|status|agent)\b",
            q,
        )

        dimension = (
            dimension_match.group(1)
            if dimension_match
            else None
        )

        group_map = {
            "agent": "agent_id",
            "category": "category",
            "priority": "priority",
            "status": "status",
        }

        if dimension:

            plan["operation"] = "group_by"
            plan["group_by"] = (
                group_map[dimension]
            )
            plan["field"] = None

            if not plan.get("metric"):

                plan["metric"] = "count"

            if re.search(
                r"\b(highest|most|maximum|max)\b",
                q,
            ):

                plan["order"] = "desc"

            else:

                plan["order"] = "asc"

            plan["limit"] = 1

    # ---------------------------------------------------------
    # "NOT RESOLVED WITHIN N HOURS"
    # ---------------------------------------------------------

    threshold_match = re.search(
        r"\bnot\s+resolved\s+within\s+(\d+(?:\.\d+)?)\s*hours?\b",
        q,
    )

    if threshold_match:

        plan["threshold_hours"] = float(
            threshold_match.group(1)
        )

        # For this assessment wording, keep unresolved
        # tickets as Open + Escalated.
        plan["filters"]["status"] = [
            "Open",
            "Escalated",
        ]

        # It is a ticket listing request when user says
        # "show me all".
        if re.search(
            r"\b(show|list|display|all)\b",
            q,
        ):

            plan["operation"] = "list"

    else:

        threshold = plan.get(
            "threshold_hours"
        )

        if threshold is not None:

            try:

                plan["threshold_hours"] = float(
                    threshold
                )

            except (
                TypeError,
                ValueError,
            ):

                plan["threshold_hours"] = None

    # ---------------------------------------------------------
    # METRIC WORDING
    # ---------------------------------------------------------

    if (
        "average customer rating" in q
        or "avg customer rating" in q
    ):

        if plan["operation"] == "group_by":

            plan["metric"] = (
                "customer_rating"
            )

        elif plan["operation"] in {
            "average",
            "min",
            "max",
            "sum",
        }:

            plan["field"] = (
                "customer_rating"
            )

    elif (
        "average response time" in q
        or "avg response time" in q
    ):

        if plan["operation"] == "group_by":

            plan["metric"] = (
                "response_time_hrs"
            )

        elif plan["operation"] in {
            "average",
            "min",
            "max",
            "sum",
        }:

            plan["field"] = (
                "response_time_hrs"
            )

    elif (
        "average resolution time" in q
        or "avg resolution time" in q
    ):

        if plan["operation"] == "group_by":

            plan["metric"] = (
                "resolution_time_hrs"
            )

        elif plan["operation"] in {
            "average",
            "min",
            "max",
            "sum",
        }:

            plan["field"] = (
                "resolution_time_hrs"
            )

    # ---------------------------------------------------------
    # DEFAULTS
    # ---------------------------------------------------------

    if plan["operation"] != "group_by":

        plan["order"] = None
        plan["limit"] = 50

    else:

        plan["order"] = (
            "desc"
            if str(
                plan.get("order") or "desc"
            ).lower()
            == "desc"
            else "asc"
        )

        try:

            plan["limit"] = min(
                max(
                    int(
                        plan.get("limit")
                        or 1
                    ),
                    1,
                ),
                100,
            )

        except (
            TypeError,
            ValueError,
        ):

            plan["limit"] = 1

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    if (
        plan["operation"]
        in {
            "average",
            "min",
            "max",
            "sum",
        }
        and not plan["field"]
    ):

        raise ValueError(
            "The query requires a numeric field, "
            "but the LLM did not provide one."
        )

    if (
        plan["operation"] == "group_by"
        and not plan["group_by"]
    ):

        raise ValueError(
            "The grouped query requires a group_by field."
        )

    if (
        plan["operation"] == "group_by"
        and not plan["metric"]
    ):

        plan["metric"] = "count"

    return plan


def create_query_plan(
    question: str,
    reference_date: str,
) -> Dict[str, Any]:

    question = (
        question or ""
    ).strip()

    if len(question) < 3:

        raise ValueError(
            "Question must be at least 3 characters long."
        )

    _parse_reference_date(
        reference_date
    )

    url = (
        f"{OLLAMA_BASE_URL.rstrip('/')}"
        "/api/chat"
    )

    user_prompt = (
        f"Dataset reference date: {reference_date}\n"
        "Important: follow the JSON schema exactly. "
        "Never use created_at as a filter. "
        "Never invent category, priority, status, or agent filters. "
        "Never use operation='filter'; use operation='list' for "
        "ticket filtering/listing requests.\n"
        f"User question: {question}"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
        },
    }

    try:

        response = requests.post(
            url,
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )

        response.raise_for_status()

        response_json = response.json()

        content = response_json[
            "message"
        ][
            "content"
        ]

        raw_plan = _extract_json(
            content
        )

        return _sanitize_plan(
            question,
            raw_plan,
            reference_date,
        )

    except requests.RequestException as exc:

        raise RuntimeError(
            f"LLM is unavailable. "
            f"Start Ollama and pull '{OLLAMA_MODEL}'. "
            f"Details: {exc}"
        ) from exc

    except (
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:

        raise RuntimeError(
            f"Invalid structured response from LLM: {exc}"
        ) from exc