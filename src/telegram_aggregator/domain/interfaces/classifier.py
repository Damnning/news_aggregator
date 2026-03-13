from src.telegram_aggregator.domain.schemas.filter_schemas import FilterContentModel
from abc import ABC, abstractmethod


class Classifier(ABC):
    @abstractmethod
    def filter_content(self, *, filter_prompt: str, content: str) -> FilterContentModel:
        pass
