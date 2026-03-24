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
from .database import get_db, KnowledgeBase, ChatHistory, init_db

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
        record = KnowledgeBase(
            query=topic,
            slug=slug,
            content=report
        )
        db.add(record)
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
# ⚡ Core Research Endpoint
# =====================================

@app.post("/research", response_model=ResearchResponse)
async def research_controller(
    payload: ResearchRequest,
    db: Session = Depends(get_db)
):
    try:
        # 1️⃣ Check cache
        cached_entry = fetch_from_cache(payload.query, db)

        if cached_entry:
            return ResearchResponse(
                query=payload.query,
                result=cached_entry.content,
                file_path="database-cache",
                suggestions=[]
            )

        # 2️⃣ Run AI research pipeline
        engine_output = run_full_research(
            payload.query,
            payload.thread_id
        )

        report_text = engine_output.get("report")
        followups = engine_output.get("suggestions", [])

        # 3️⃣ Store results
        store_in_cache(payload.query, report_text, db)
        file_location = archive_report(payload.query, report_text)

        # 4️⃣ Return structured response
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
# 📜 Optional History Endpoint
# =====================================

@app.get("/history", response_model=List[ResearchResponse])
async def recent_history(db: Session = Depends(get_db)):
    logs = (
        db.query(ChatHistory)
        .order_by(ChatHistory.timestamp.desc())
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


# =====================================
# 🚀 Local Dev Entry
# =====================================

