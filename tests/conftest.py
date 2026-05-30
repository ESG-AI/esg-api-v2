import os
import sys

# Ensure root directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Force database URL to in-memory SQLite before db.session is imported
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["NEON_DATABASE_URL"] = "sqlite:///:memory:"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-unit-testing-only-12345"

# Register SQLite compilation overrides for Postgres-specific types (JSONB)
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.dialects.postgresql import JSONB
@compiles(JSONB, "sqlite")
def compile_jsonb_sqlite(type_, compiler, **kw):
    return "JSON"

# Import db.session and patch it BEFORE importing main, models, or repositories
import db.session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

# Apply the patch immediately
db.session.engine = test_engine
db.session.SessionLocal = TestingSessionLocal

# Now import Base and app safely
import pytest
from fastapi.testclient import TestClient
from db.models import Base
from main import app

@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    # Create all tables for the session
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)

@pytest.fixture(autouse=True)
def clean_db():
    # Ensure test isolation by clearing database tables before each test runs
    Base.metadata.drop_all(bind=test_engine)
    Base.metadata.create_all(bind=test_engine)
    yield

@pytest.fixture
def client():
    # Return TestClient for testing the FastAPI application endpoints
    with TestClient(app) as c:
        yield c
