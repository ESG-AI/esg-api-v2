import os

from dotenv import load_dotenv

load_dotenv()  # must run before config imports so env vars are available

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from db.session import init_db

from routes.pdf import router as pdf_router
from routes.evaluate import router as evaluate_router
from routes.documents import router as documents_router
from routes.utils import router as utils_router
from routes.auth import router as auth_router
from routes.admin import router as admin_router

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ESG Scoring API",
    description="Backend API for ESG document analysis and scoring.",
    version="2.0.0",
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------
# Read comma-separated origins from env, e.g.:
#   ALLOWED_ORIGINS=https://app.yourdomain.com,https://www.yourdomain.com
# Falls back to ["*"] when unset (local development only).
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS: list[str] = (
    [o.strip() for o in _raw_origins.split(",") if o.strip()]
    if _raw_origins
    else ["*"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_db_client():
    init_db()

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(pdf_router)
app.include_router(evaluate_router)
app.include_router(documents_router)
app.include_router(utils_router)

# ---------------------------------------------------------------------------
# Local dev entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)