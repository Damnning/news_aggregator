from pydantic import BaseModel, Field

class FilterContentModel(BaseModel):
    is_relevant : bool = Field(description="Is content relevant to filter")