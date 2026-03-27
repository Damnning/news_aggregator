from src.telegram_aggregator.domain.interfaces.classifier import Classifier
from src.telegram_aggregator.domain.prompts.prompts import CLASSIFY_TEMPLATE
from src.telegram_aggregator.domain.schemas.filter_schemas import FilterContentModel
from groq import Groq


class GroqClassifier(Classifier):
    def __init__(self, *, model: str, api_key: str, temperature: float) -> None:
        self.model = model
        self.client = Groq(
            api_key=api_key,
            base_url="https://groq.com"
        )
        self.temperature = temperature

    def filter_content(self, *, filter_prompt: str, content: str,
                       temperature: float | None = None) -> FilterContentModel | None:
        try:
            current_temp = temperature if temperature is not None else self.temperature
            prompt = CLASSIFY_TEMPLATE.format(text=content, filter=filter_prompt)
            messages = [{"role": "user", "content": prompt}]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=current_temp
            )

            content_str = response.choices[0].message.content
            return FilterContentModel.model_validate_json(content_str)

        except Exception as e:
            print(f"An error occurred: {e}")
            return None
