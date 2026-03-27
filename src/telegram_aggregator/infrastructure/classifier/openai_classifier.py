from openai.types.chat import ChatCompletionUserMessageParam

from src.telegram_aggregator.domain.interfaces.classifier import Classifier
from src.telegram_aggregator.domain.prompts.prompts import CLASSIFY_TEMPLATE
from src.telegram_aggregator.domain.schemas.filter_schemas import FilterContentModel
from openai import OpenAI

from abc import ABC


class OpenaiClassifier(Classifier, ABC):
    def __init__(self, *, model: str, api_key: str, temperature: float, base_url: str) -> None:
        self.model = model
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.temperature = temperature

    def filter_content(self, *, filter_prompt: str, content: str, temperature: float | None = None) -> FilterContentModel | None:
        try:
            current_temp = temperature if temperature is not None else self.temperature
            prompt = CLASSIFY_TEMPLATE.format(text=content, filter=filter_prompt)
            messages: list[ChatCompletionUserMessageParam] = [{"role": "user", "content": prompt}]

            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=FilterContentModel,
                temperature=current_temp
            )
            parsed: FilterContentModel = response.choices[0].message.parsed
            return parsed
        except Exception as e:
            print(f"An error occurred: {e}") # TODO: посмотреть как делать логгеры и сделать его тут
