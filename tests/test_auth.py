from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.main import app
from backend.database import Base, get_db
import pytest
import os

# Create a test database
TEST_DATABASE_URL = "sqlite:///./test_users.db"

engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Override get_db to use test database
def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

# This fixture will run once for the module
@pytest.fixture(scope="module", autouse=True)
def setup_database():
    # Ensure clean state
    if os.path.exists("./test_users.db"):
        os.remove("./test_users.db")
    
    # Initialize DB (creates tables)
    Base.metadata.create_all(bind=engine)
    
    yield
    
    # Cleanup
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
    response = client.post(
        "/token",
        data={"username": "testuser", "password": "password123"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data

def test_login_invalid():
    response = client.post(
        "/token",
        data={"username": "testuser", "password": "wrongpassword"}
    )
    assert response.status_code == 401

def test_get_current_user():
    # Login first
    login_res = client.post(
        "/token",
        data={"username": "testuser", "password": "password123"}
    )
    token = login_res.json()["access_token"]
    
    response = client.get(
        "/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["username"] == "testuser"

def test_chat_history_auth():
    # Login first
    login_res = client.post(
        "/token",
        data={"username": "testuser", "password": "password123"}
    )
    token = login_res.json()["access_token"]
    
    # Send a chat
    chat_res = client.post(
        "/research",
        json={"query": "Solar Power"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert chat_res.status_code == 200
    
    # Get history
    hist_res = client.get(
        "/history",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert hist_res.status_code == 200
    history = hist_res.json()
    assert len(history) > 0
    assert history[0]["query"] == "Solar Power"
