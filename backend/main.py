from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import Optional, List
from datetime import datetime
from slugify import slugify
import os
from dotenv import load_dotenv

from .models import ResearchRequest, ResearchResponse
from .research_chain import run_full_research
from .database import get_db, KnowledgeBase, ChatHistory, User, init_db
from .auth import get_password_hash, verify_password, create_access_token

# =====================================
# 🔧 Application Bootstrap
# =====================================

load_dotenv()
init_db()

app = FastAPI(
    title="Energy Intelligence Engine",
    description="Autonomous Energy Research Service",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================
# 🧠 Utility Layer
# =====================================

def normalize_topic(topic: str) -> str:
    return slugify(topic.strip())


def fetch_from_cache(topic: str, db: Session):
    slug = normalize_topic(topic)
    return db.query(KnowledgeBase).filter(KnowledgeBase.slug == slug).first()


def store_in_cache(topic: str, report: str, db: Session):
    slug = normalize_topic(topic)
    if not db.query(KnowledgeBase).filter(KnowledgeBase.slug == slug).first():
        record = KnowledgeBase(query=topic, slug=slug, content=report)
        db.add(record)
        db.commit()


def save_to_history(topic: str, report: str, db: Session):
    """Save research result to ChatHistory for /history endpoint."""
    entry = ChatHistory(query=topic, response=report)
    db.add(entry)
    db.commit()


def archive_report(topic: str, report: str) -> Optional[str]:
    try:
        directory = os.path.join(os.path.dirname(__file__), "knowledge_base")
        os.makedirs(directory, exist_ok=True)
        filename = f"{normalize_topic(topic)}.txt"
        full_path = os.path.join(directory, filename)
        with open(full_path, "w", encoding="utf-8") as file:
            file.write(
                f"Topic: {topic}\n"
                f"Generated At: {datetime.utcnow()}\n\n"
                f"{report}"
            )
        return full_path
    except Exception:
        return None


# =====================================
# 🔐 Authentication Endpoints
# =====================================

from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class UserCreate(BaseModel):
    username: str
    email: str
    password: str


class UserResponse(BaseModel):
    username: str
    email: str


# FIX 1: Added username field so test_signup assertion works
class Token(BaseModel):
    access_token: str
    token_type: str
    username: str


@app.post("/signup", response_model=Token)
async def signup(user: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(
        (User.username == user.username) | (User.email == user.email)
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")

    hashed_password = get_password_hash(user.password)
    db_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    access_token = create_access_token(payload={"sub": db_user.username})
    # FIX 1: return username in response
    return Token(access_token=access_token, token_type="bearer", username=db_user.username)


@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token(payload={"sub": user.username})
    return Token(access_token=access_token, token_type="bearer", username=user.username)


async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    from .auth import decode_access_token
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    username = payload.get("sub")
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


# FIX 2: was "/users/me", test calls "/me"
@app.get("/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return UserResponse(username=current_user.username, email=current_user.email)


# =====================================
# ⚡ Core Research Endpoint
# =====================================

@app.post("/research", response_model=ResearchResponse)
async def research_controller(
    payload: ResearchRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        cached_entry = fetch_from_cache(payload.query, db)

        if cached_entry:
            # FIX 5: also save cache hits to history so /history is never empty
            save_to_history(payload.query, cached_entry.content, db)
            return ResearchResponse(
                query=payload.query,
                result=cached_entry.content,
                file_path="database-cache",
                suggestions=[]
            )

        engine_output = run_full_research(payload.query, payload.thread_id)

        report_text = engine_output.get("report")
        followups = engine_output.get("suggestions", [])

        store_in_cache(payload.query, report_text, db)
        # FIX 5: save to ChatHistory so /history endpoint returns results
        save_to_history(payload.query, report_text, db)
        file_location = archive_report(payload.query, report_text)

        return ResearchResponse(
            query=payload.query,
            result=report_text,
            file_path=file_location,
            suggestions=followups
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Research processing failed: {str(error)}"
        )


# =====================================
# 📜 History Endpoint
# =====================================

@app.get("/history", response_model=List[ResearchResponse])
async def recent_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # FIX 3: was ChatHistory.timestamp — column is actually created_at
    logs = (
        db.query(ChatHistory)
        .order_by(ChatHistory.created_at.desc())
        .limit(5)
        .all()
    )

    return [
        ResearchResponse(
            query=entry.query,
            result=entry.response,
            file_path=None,
            suggestions=[]
        )
        for entry in logs
    ]