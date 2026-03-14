from fastapi import APIRouter

from src.telegram_aggregator.infrastructure.openrouter_classifier import OpenrouterClassifier
from src.telegram_aggregator.presentation.api_v1.schemas.classify_shemas import ClassifyRequest, ClassifyResponse
from src.telegram_aggregator.use_cases.classify_content import ClassifyContentUseCase
from src.telegram_aggregator.infrastructure.config import settings


llm_router = APIRouter(prefix="/llm", tags=["llm"])

@llm_router.post("/classify", response_model=ClassifyResponse)
def classify_content(data: ClassifyRequest):
    classifier = OpenrouterClassifier(model=settings.openrouter_model, api_key=settings.openrouter_key, temperature=data.temperature)
    use_case = ClassifyContentUseCase(classifier=classifier) # TODO: вынести создание юз кейса в зависимости
    response = use_case.invoke(filter_prompt=data.filter_prompt, content=data.content) # TODO: сделать конвертеры для схем
    return ClassifyResponse(is_relevant=response)
