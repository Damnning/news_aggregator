from src.telegram_aggregator.domain.interfaces.classifier import Classifier


class ClassifyContentUseCase:
    def __init__(self, *, classifier: Classifier):
        self.classifier = classifier

    def invoke(self, *, filter_prompt: str, content: str) -> bool:
        response = self.classifier.filter_content(filter_prompt=filter_prompt, content=content)
        return response.is_relevant
