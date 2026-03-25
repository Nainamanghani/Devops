from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
import os

# ===============================
# 🗄 Database Configuration
# ===============================

def build_database_url() -> str:
    return os.getenv("DATABASE_URL", "sqlite:///./energy_data.db")


DATABASE_URI = build_database_url()

engine = create_engine(
    DATABASE_URI,
    connect_args={"check_same_thread": False} if DATABASE_URI.startswith("sqlite") else {}
)

SessionFactory = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base = declarative_base()


# ===============================
# ⏱ Timestamp Mixin
# ===============================

class TimestampMixin:
    created_at = Column(DateTime, default=datetime.utcnow)


# ===============================
# 📚 Knowledge Base Entity
# ===============================

class KnowledgeBase(Base, TimestampMixin):
    __tablename__ = "knowledge_base"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(String(255), nullable=False)
    slug = Column(String(255), unique=True, index=True)
    content = Column(Text, nullable=False)


# ===============================
# 💬 Chat History Entity
# ===============================

class ChatHistory(Base, TimestampMixin):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    thread_id = Column(String(255), index=True)


# ===============================
# � User Entity
# ===============================

class User(Base, TimestampMixin):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)


# ===============================
# �🔧 DB Utilities
# ===============================

def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionFactory()
    try:
        yield db
    finally:
        db.close()