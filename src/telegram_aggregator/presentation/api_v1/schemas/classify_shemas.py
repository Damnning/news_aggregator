from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    content: str = Field(description="Content to classify")
    filter_prompt: str = Field(description="Prompt to filter content")


class ClassifyResponse(BaseModel):
    is_relevant: bool = Field(description="Is content relevant to filter")
