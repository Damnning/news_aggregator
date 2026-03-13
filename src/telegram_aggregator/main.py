from src.telegram_aggregator.presentation.api_v1.routes.llm_routes import llm_router

from fastapi import FastAPI

app = FastAPI()

app.include_router(llm_router)