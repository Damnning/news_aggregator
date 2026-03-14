from typing import Optional

from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    content: str = Field(description="Content to classify")
    filter_prompt: str = Field(description="Prompt to filter content")
    temperature: Optional[float] = Field(0, ge=0.0, le=2.0, description="Llm generation temperature")
    # NOTE: если буду использовать не openrouter api возможно надо будет делать нормализацию параметра, т.к. не у всех может быть от 0 до 2


class ClassifyResponse(BaseModel):
    is_relevant: bool = Field(description="Is content relevant to filter")
