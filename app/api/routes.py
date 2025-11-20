"""API routes for the application."""
from fastapi import APIRouter

# Import all endpoint routers
from app.api.endpoints import evaluate

# Create the main API router
api_router = APIRouter()

# Include the endpoint routers
api_router.include_router(evaluate.router, prefix="/evaluations", tags=["evaluations"])
