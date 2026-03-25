from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from slugify import slugify
import os
from dotenv import load_dotenv

from .models import ResearchRequest, ResearchResponse
from .research_chain import run_full_research
from .database import get_db, KnowledgeBase, ChatHistory, User, init_db
from .auth import get_password_hash, verify_password, create_access_token, decode_access_token

load_dotenv()
init_db()

app = FastAPI(title="Energy Intelligence Engine", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class UserCreate(BaseModel):
    username: str
    email: str
    password: str


class UserResponse(BaseModel):
    username: str
    email: str


class Token(BaseModel):
    access_token: str
    token_type: str
    username: str


def normalize_topic(topic: str) -> str:
    return slugify(topic.strip())


def fetch_from_cache(topic: str, db: Session):
    slug = normalize_topic(topic)
    return db.query(KnowledgeBase).filter(KnowledgeBase.slug == slug).first()


def store_in_cache(topic: str, report: str, db: Session):
    slug = normalize_topic(topic)
    if not db.query(KnowledgeBase).filter(KnowledgeBase.slug == slug).first():
        db.add(KnowledgeBase(query=topic, slug=slug, content=report))
        db.commit()


def save_to_history(topic: str, report: str, db: Session):
    db.add(ChatHistory(query=topic, response=report))
    db.commit()


def archive_report(topic: str, report: str) -> Optional[str]:
    try:
        directory = os.path.join(os.path.dirname(__file__), "knowledge_base")
        os.makedirs(directory, exist_ok=True)
        filename = f"{normalize_topic(topic)}.txt"
        full_path = os.path.join(directory, filename)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(f"Topic: {topic}\nGenerated At: {datetime.utcnow()}\n\n{report}")
        return full_path
    except Exception:
        return None


async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(User).filter(User.username == payload.get("sub")).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


@app.post("/signup", response_model=Token)
async def signup(user: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(
        (User.username == user.username) | (User.email == user.email)
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")
    db_user = User(
        username=user.username,
        email=user.email,
        hashed_password=get_password_hash(user.password)
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    token = create_access_token(payload={"sub": db_user.username})
    return Token(access_token=token, token_type="bearer", username=db_user.username)


@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(payload={"sub": user.username})
    return Token(access_token=token, token_type="bearer", username=user.username)


@app.get("/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return UserResponse(username=current_user.username, email=current_user.email)


@app.post("/research", response_model=ResearchResponse)
async def research_controller(
    payload: ResearchRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        cached = fetch_from_cache(payload.query, db)
        if cached:
            save_to_history(payload.query, cached.content, db)
            return ResearchResponse(query=payload.query, result=cached.content, file_path="database-cache", suggestions=[])

        output = run_full_research(payload.query, payload.thread_id)
        report = output.get("report")
        suggestions = output.get("suggestions", [])

        store_in_cache(payload.query, report, db)
        save_to_history(payload.query, report, db)
        file_path = archive_report(payload.query, report)

        return ResearchResponse(query=payload.query, result=report, file_path=file_path, suggestions=suggestions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Research failed: {str(e)}")


@app.get("/history", response_model=List[ResearchResponse])
async def recent_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    logs = db.query(ChatHistory).order_by(ChatHistory.created_at.desc()).limit(5).all()
    return [ResearchResponse(query=e.query, result=e.response, file_path=None, suggestions=[]) for e in logs]
