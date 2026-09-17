from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.api.routes import router
from app.db.database import initialize_database


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_database()
    yield


app = FastAPI(
    title="DOTMappers AI Support Ticket Intelligence",
    version="1.0.0",
    description="Natural-language analytics and anomaly detection for support tickets.",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/", response_class=HTMLResponse)
def home():
    html_path = Path(__file__).resolve().parent / "templates" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))
