from fastapi import APIRouter, HTTPException, Depends, status
from typing import List, Optional
from loguru import logger

from app.schemas.evaluation import (
    EvaluationRequest,
    EvaluationResponse,
    HealthCheck
)
from app.services.evaluation_service import evaluation_service
from app.db.session import get_db

router = APIRouter()

@router.post(
    "/evaluate",
    response_model=EvaluationResponse,
    summary="Evaluate text for safety and quality",
    description="""
    Evaluates the given text across multiple dimensions including:
    - Toxicity: Detects harmful, offensive, or inappropriate content
    - PII: Identifies personally identifiable information
    - Bias: Detects potential biases in the text
    - Hallucination: Identifies potential factual inaccuracies
    
    Returns an overall safety score (0-100) and detailed evaluation results.
    """
)
async def evaluate_text(
    request: EvaluationRequest,
    db=Depends(get_db)
) -> EvaluationResponse:
    """
    Evaluate text for safety and quality across multiple dimensions.
    """
    try:
        logger.info(f"Received evaluation request for text: {request.text[:100]}...")
        
        # Call the evaluation service
        result = await evaluation_service.evaluate_text(
            text=request.text,
            context=request.context,
            evaluations=request.evaluations
        )
        
        # Log the evaluation result (in a real app, you might want to store this in a database)
        logger.info(f"Evaluation completed with safety score: {result['safety_score']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during evaluation: {str(e)}"
        )

@router.get(
    "/health",
    response_model=HealthCheck,
    summary="Check service health",
    description="Returns the current health status of the evaluation service."
)
async def health_check() -> HealthCheck:
    """Check the health of the evaluation service and its models."""
    try:
        # Check if all models are loaded
        model_status = {
            "toxicity": "loaded" if evaluation_service.evaluators["toxicity"]._is_initialized else "not loaded",
            "pii": "loaded" if evaluation_service.evaluators["pii"]._is_initialized else "not loaded",
            "bias": "loaded" if evaluation_service.evaluators["bias"]._is_initialized else "not loaded",
            "hallucination": "loaded" if evaluation_service.evaluators["hallucination"]._is_initialized else "not loaded",
        }
        
        return HealthCheck(
            status="healthy",
            version="1.0.0",
            model_status=model_status
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthCheck(
            status="unhealthy",
            version="1.0.0",
            model_status={"error": str(e)}
        )
