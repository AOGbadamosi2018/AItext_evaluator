from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class EvaluationRequest(BaseModel):
    """Request schema for text evaluation."""
    text: str = Field(..., description="The text to evaluate")
    context: Optional[str] = Field(
        None, 
        description="Optional context for hallucination detection"
    )
    evaluations: Optional[List[str]] = Field(
        None,
        description="List of specific evaluations to run (default: all)",
        example=["toxicity", "pii", "bias", "hallucination"]
    )

class EvaluationResult(BaseModel):
    """Base result schema for an individual evaluation."""
    score: float = Field(..., ge=0.0, le=1.0, description="Evaluation score (0-1)")
    evaluation_type: str = Field(..., description="Type of evaluation")
    details: Optional[Dict[str, Any]] = Field(
        None, 
        description="Additional details specific to the evaluation type"
    )

class EvaluationResponse(BaseModel):
    """Response schema for evaluation results."""
    safety_score: float = Field(
        ..., 
        ge=0.0, 
        le=100.0,
        description="Overall safety score (0-100), higher is better"
    )
    text: str = Field(..., description="The evaluated text")
    context: Optional[str] = Field(None, description="Context used for evaluation")
    evaluations: Dict[str, EvaluationResult] = Field(
        ...,
        description="Dictionary of evaluation results by evaluation type"
    )

class HealthCheck(BaseModel):
    """Health check response schema."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    model_status: Dict[str, str] = Field(
        ...,
        description="Status of each evaluation model"
    )
