from pydantic import BaseModel, Field


class RAGResponse(BaseModel):
    answer: str = Field(
        description="Answer to the user's question."
    )

    confidence: float = Field(
        description="Confidence score between 0 and 1."
    )

    sources_used: list[str] = Field(
        description="List of sources used to answer."
    )

    follow_up: str = Field(
        description="Suggested follow-up question."
    )