from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch
from backend.main import app
from backend.database import Base, get_db
import pytest
import os

TEST_DATABASE_URL = "sqlite:///./test_users.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_database():
    if os.path.exists("./test_users.db"):
        os.remove("./test_users.db")
    Base.metadata.create_all(bind=engine)
    yield
    if os.path.exists("./test_users.db"):
        os.remove("./test_users.db")

def test_signup():
    response = client.post(
        "/signup",
        json={"username": "testuser", "email": "test@example.com", "password": "password123"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["username"] == "testuser"

def test_login():
    response = client.post("/token", data={"username": "testuser", "password": "password123"})
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_login_invalid():
    response = client.post("/token", data={"username": "testuser", "password": "wrongpassword"})
    assert response.status_code == 401

def test_get_current_user():
    token = client.post("/token", data={"username": "testuser", "password": "password123"}).json()["access_token"]
    response = client.get("/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["username"] == "testuser"

def test_chat_history_auth():
    token = client.post("/token", data={"username": "testuser", "password": "password123"}).json()["access_token"]

    with patch("backend.main.run_full_research") as mock_research:
        mock_research.return_value = {
            "report": "Solar power is a renewable energy source.",
            "suggestions": ["What is wind energy?"]
        }
        chat_res = client.post(
            "/research",
            json={"query": "Solar Power"},
            headers={"Authorization": f"Bearer {token}"}
        )
    assert chat_res.status_code == 200

    hist_res = client.get("/history", headers={"Authorization": f"Bearer {token}"})
    assert hist_res.status_code == 200
    history = hist_res.json()
    assert len(history) > 0
    assert history[0]["query"] == "Solar Power"
