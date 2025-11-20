"""API package for AI Text Evaluator."""

from fastapi import APIRouter

# Create the main API router
api_router = APIRouter()

# Import and include all endpoint routers here
from .endpoints import evaluate  # noqa: E402

# Include the endpoint routers
api_router.include_router(evaluate.router, prefix="/evaluations", tags=["evaluations"])
