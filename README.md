````markdown
# DOTMappers AI Engineer Assessment — AI Support Ticket Intelligence

An AI-powered customer support ticket analytics system built for the DOTMappers AI Engineer Assessment.

The system allows users to ask natural-language questions about support tickets, converts those questions into validated query plans using a local LLM, executes the queries against the ticket dataset, and performs deterministic anomaly detection.

---

## Requirements Covered

- CSV ingestion and queryable support-ticket data
- Natural-language querying using a local LLM
- Deterministic anomaly detection
- REST API
- Minimal browser UI
- Health-check endpoint
- No paid API or external AI service required
- Safe and validated query execution
- Automated tests

---

## Architecture

```text
                         Browser UI
                             |
                             v
                          FastAPI
                             |
                 +-----------+-----------+
                 |                       |
                 v                       v
        Natural Language           Anomaly Engine
           Query Flow              (Pandas/Stats)
                 |
                 v
          Ollama Local LLM
                 |
                 v
          JSON Query Plan
                 |
                 v
     Deterministic Validation
                 |
                 v
      Query Plan Normalization
                 |
                 v
             SQLite
                 ^
                 |
      support_tickets.csv
````

### Architecture Approach

The LLM is used for natural-language understanding and query-plan generation.

The LLM does not directly generate or execute SQL.

The application validates and normalizes the generated query plan before executing the requested operation against the SQLite database.

This keeps analytical results grounded in the supplied ticket data and prevents arbitrary LLM-generated SQL from being executed.

Anomaly detection is implemented separately using deterministic statistical rules so that anomaly results are reproducible and explainable.

---

## Dataset

The supplied `support_tickets.csv` is included in the `data/` directory.

### Dataset Information

* Rows: 500
* Columns: 10
* Date range: 2024-01-01 to 2024-03-30
* Categories: Billing, General, Technical
* Priorities: Low, Medium, High, Critical
* Statuses: Open, Resolved, Escalated

### Dataset Fields

| Field                 | Description                      |
| --------------------- | -------------------------------- |
| `ticket_id`           | Unique support ticket identifier |
| `created_at`          | Ticket creation timestamp        |
| `category`            | Ticket category                  |
| `priority`            | Ticket priority                  |
| `status`              | Current ticket status            |
| `response_time_hrs`   | Response time in hours           |
| `resolution_time_hrs` | Resolution time in hours         |
| `agent_id`            | Support agent identifier         |
| `customer_rating`     | Customer rating                  |
| `issue_summary`       | Short description of the issue   |

`resolution_time_hrs` and `customer_rating` can be null for unresolved tickets.

---

## Technology Stack

* Python
* FastAPI
* SQLite
* Pandas
* Pydantic
* Ollama
* Llama 3.2 3B
* HTML/CSS/JavaScript
* Pytest

---

## Model

The default local LLM is:

```text
llama3.2:3b
```

The model runs locally through Ollama.

No paid LLM API key is required.

The Ollama model can be changed using the `OLLAMA_MODEL` configuration.

---

# Setup

## 1. Clone / Download the Project

Place the project on your local machine and open the project directory in a terminal.

---

## 2. Create a Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Windows CMD

```cmd
python -m venv .venv
.venv\Scripts\activate
```

---

## 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Install Ollama

Install Ollama on the local machine.

Then download the default model:

```bash
ollama pull llama3.2:3b
```

Make sure Ollama is running locally at:

```text
http://localhost:11434
```

If another Ollama model is used, configure the `OLLAMA_MODEL` setting accordingly.

---

## 5. Start the Application

Run:

```bash
python run.py
```

The application will start on:

```text
http://localhost:8000
```

Open the browser UI:

```text
http://localhost:8000
```

FastAPI Swagger documentation:

```text
http://localhost:8000/docs
```

---

# REST API

## 1. Health Check

### Endpoint

```http
GET /api/health
```

### Example Response

```json
{
  "status": "healthy",
  "database": "connected"
}
```

---

## 2. Natural-Language Query

### Endpoint

```http
POST /api/query
```

### Request

```json
{
  "question": "How many tickets are currently open?"
}
```

### Example

The endpoint processes the natural-language question, generates a validated query plan, executes it against the ticket data, and returns the answer together with structured query information.

---

## 3. Anomaly Detection

### Endpoint

```http
GET /api/anomalies
```

This endpoint executes the deterministic anomaly-detection engine against the support-ticket dataset.

---

# Natural-Language Querying

The system supports natural-language questions instead of requiring users to write SQL.

Examples include:

```text
How many tickets are currently open?

Which agent resolved the most tickets?

What is the average customer rating for Technical category tickets?

Show me all Critical tickets not resolved within 12 hours.

Are there any anomalies in resolution times?
```

The system is not hardcoded to only these sample questions.

The LLM converts the natural-language question into a structured JSON query plan, which is then validated and normalized by the application before execution.

---

# Example Queries and Results

## 1. How many tickets are currently open?

For the supplied dataset:

```text
111
```

---

## 2. Which agent resolved the most tickets?

The query is interpreted as:

* Status = Resolved
* Group by agent
* Count resolved tickets
* Sort in descending order
* Return the highest result

For the supplied dataset, the highest count is tied between:

```text
AGT-09
AGT-12
```

with:

```text
37 resolved tickets each
```

---

## 3. What is the average customer rating for Technical category tickets?

For the supplied dataset:

```text
3.74
```

---

## 4. Show me all Critical tickets not resolved within 12 hours.

The query engine treats a ticket as matching when:

* Priority is Critical
* The ticket is unresolved, OR
* Its resolution time is greater than 12 hours

For the supplied dataset:

```text
31 matching tickets
```

---

## 5. Are there any anomalies in resolution times?

Resolution-time anomalies use the following statistical rule:

```text
Q3 + 1.5 × IQR
```

For the supplied dataset:

```text
Q3: 22.95 hours
IQR: 16.80 hours
Threshold: 48.15 hours
```

Tickets with resolution times above the threshold are classified as:

```text
long_resolution_time
```

The general anomaly detector also checks unresolved High/Critical tickets that are older than 24 hours relative to the latest dataset timestamp.

For questions specifically asking about resolution-time anomalies, the system returns only resolution-time anomalies and does not mix them with the old unresolved High/Critical rule.

---

# Query Plan Validation and Normalization

The LLM proposes the initial query structure, but the application performs deterministic validation and normalization before execution.

This helps prevent common LLM interpretation errors such as:

* Treating `created_at` as a direct filter instead of using `start_date` and `end_date`
* Inventing filters that were not mentioned by the user
* Adding an unintended date range
* Misinterpreting `highest`, `lowest`, `most`, and `least`
* Incorrectly handling grouped count queries
* Incorrectly handling "not resolved within N hours"
* Allowing unsupported fields or operations

The application only allows predefined fields, filters, operations, and grouping dimensions.

Relative date expressions are resolved against the latest timestamp available in the supplied dataset so that results remain deterministic for the assessment dataset.

---

# LLM Safety

The LLM does not directly execute SQL.

The query flow is:

```text
User Question
     |
     v
Ollama
     |
     v
JSON Query Plan
     |
     v
Application Validation
     |
     v
Deterministic Normalization
     |
     v
Allowed Operations / Fields
     |
     v
SQLite Query Execution
     |
     v
Exact Dataset Result
```

The system does not allow arbitrary SQL generated by the LLM.

Only supported operations, fields, filters, and grouping dimensions are accepted.

---

# Supported Query Operations

The query engine supports controlled analytical operations including:

* Count
* Average
* Minimum
* Maximum
* Sum
* List
* Group-by

Supported grouping dimensions include:

* Agent
* Category
* Priority
* Status

Supported analytical fields include:

* `response_time_hrs`
* `resolution_time_hrs`
* `customer_rating`

---

# Filtering

The system supports controlled filtering based on user questions.

Supported filters include:

* Category
* Priority
* Status
* Agent

Examples:

```text
Show all Critical tickets.

How many Technical tickets are open?

How many tickets were resolved?

Show Billing tickets handled by AGT-05.
```

---

# Date Handling

The system supports relative date expressions such as:

```text
today
yesterday
this week
last week
this month
last month
```

Relative dates are calculated against the latest date available in the supplied historical dataset.

This makes date-based results deterministic for the provided assessment dataset.

---

# Null Handling

The system explicitly handles nullable ticket fields.

* `resolution_time_hrs` can be null for unresolved tickets.
* `customer_rating` can be null for unresolved tickets.
* Aggregate calculations ignore null values.
* "Not resolved within N hours" explicitly includes unresolved tickets.

---

# Anomaly Detection

Anomaly detection is deterministic and does not rely on the LLM to generate anomaly results.

## Long Resolution Time

The system calculates the resolution-time anomaly threshold using:

```text
Q3 + 1.5 × IQR
```

Tickets above this threshold are classified as:

```text
long_resolution_time
```

For the supplied dataset:

```text
Q3: 22.95 hours
IQR: 16.80 hours
Threshold: 48.15 hours
```

## Old Unresolved High-Priority Tickets

The general anomaly detector also flags unresolved High/Critical tickets that are older than 24 hours relative to the latest dataset timestamp.

This rule is kept separate from resolution-time anomalies.

Therefore:

```text
Resolution-time anomaly question
        |
        v
Only long-resolution anomalies
```

while:

```text
General anomaly detection
        |
        +--> Long resolution anomalies
        |
        +--> Old unresolved High/Critical tickets
```

---

# Minimal Browser UI

The project includes a minimal browser-based UI.

The UI provides access to:

* Natural-language ticket queries
* Query results
* Structured query plans
* Anomaly detection
* Health status

Start the application with:

```bash
python run.py
```

Then open:

```text
http://localhost:8000
```

---

# Testing

Run the complete test suite with:

```bash
pytest -q
```

The tests cover core functionality including:

* Open-ticket counting
* Category/customer-rating aggregation
* Anomaly detection
* Health endpoint
* UI endpoint

---

# Project Structure

```text
.
├── app/
│   ├── api/
│   │   └── routes.py
│   │
│   ├── db/
│   │   └── database.py
│   │
│   ├── models/
│   │   └── schemas.py
│   │
│   └── services/
│       ├── anomaly_service.py
│       ├── llm_service.py
│       └── query_service.py
│
├── data/
│   └── support_tickets.csv
│
├── tests/
│
├── requirements.txt
├── run.py
└── README.md
```

---

# Requirements

Python dependencies are listed in:

```text
requirements.txt
```

The project uses a local Ollama model, so no paid AI API dependency is required.

---

# Known Limitations

1. The default local LLM requires Ollama and a locally downloaded model.

2. Relative date expressions are resolved against the dataset reference date.

3. The supported query language intentionally focuses on safe analytics operations rather than arbitrary SQL.

4. The anomaly rules are designed for the assessment dataset and would require domain-specific calibration for a production environment.

5. The browser UI is intentionally minimal.

6. Natural-language anomaly questions are routed to the deterministic anomaly engine; the LLM does not generate anomaly results.

7. The application is designed for the supplied assessment dataset and would require additional scalability and operational controls for very large production datasets.

---

# Future Improvements

Possible production enhancements include:

* Richer date and time parsing
* Authentication and authorization
* Production-grade persistent database
* Query caching
* Observability and distributed tracing
* Configurable anomaly policies
* LLM prompt and version tracking
* Model evaluation and benchmarking
* Docker-based deployment
* Automated Ollama/model setup
* Improved UI and data visualization

---

# Quick Start

For a quick setup:

```bash
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install -r requirements.txt
```

Install the local model:

```bash
ollama pull llama3.2:3b
```

Start the application:

```bash
python run.py
```

Open:

```text
http://localhost:8000
```

API documentation:

```text
http://localhost:8000/docs
```

Run tests:

```bash
pytest -q
```

---

# Conclusion

This project implements an end-to-end AI-powered support-ticket intelligence system using a local LLM for natural-language understanding and deterministic application logic for data querying and anomaly detection.

The architecture separates LLM interpretation from data execution, providing controlled, reproducible, and explainable analytical results.

```
```
