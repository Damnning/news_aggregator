from openai.types.chat import ChatCompletionUserMessageParam

from src.telegram_aggregator.domain.interfaces.classifier import Classifier
from src.telegram_aggregator.domain.prompts.prompts import CLASSIFY_TEMPLATE
from src.telegram_aggregator.domain.schemas.filter_schemas import FilterContentModel
from openai import OpenAI


class OpenrouterClassifier(Classifier):
    def __init__(self, *, model: str, api_key: str) -> None:
        self.model = model
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

    def filter_content(self, *, filter_prompt: str, content: str) -> FilterContentModel | None:
        try:
            prompt = CLASSIFY_TEMPLATE.format(text=content, filter=filter_prompt)
            messages: list[ChatCompletionUserMessageParam] = [{"role": "user", "content": prompt}]

            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=FilterContentModel
            )
            parsed: FilterContentModel = response.choices[0].message.parsed
            return parsed
        except Exception as e:
            print(f"An error occurred: {e}") # TODO: посмотреть как делать логгеры и сделать его тут
