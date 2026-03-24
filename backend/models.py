from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List


# ==============================
# 📥 Incoming Research Payload
# ==============================

class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Energy research topic")
    thread_id: Optional[str] = Field(
        default=None,
        description="Optional session identifier for conversational memory"
    )


# ==============================
# 📤 Research API Response
# ==============================

class ResearchResponse(BaseModel):
    query: str
    result: str
    file_path: Optional[str] = None
    suggestions: List[str] = Field(default_factory=list)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Future of solar energy in India",
                "result": "Solar capacity is expected to grow significantly...",
                "file_path": "knowledge_base/solar-energy.txt",
                "suggestions": [
                    "How does policy impact solar investments?",
                    "What are storage challenges?",
                    "How does India compare globally?"
                ]
            }
        }
    )