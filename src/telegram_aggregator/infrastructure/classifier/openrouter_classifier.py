from src.telegram_aggregator.infrastructure.classifier.openai_classifier import OpenaiClassifier


class OpenrouterClassifier(OpenaiClassifier):
    def __init__(self, *, model: str, api_key: str, temperature: float) -> None:
        super().__init__(model=model, api_key=api_key, temperature=temperature, base_url="https://openrouter.ai/api/v1")
