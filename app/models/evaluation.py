from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.sql import func
from .base import Base

class EvaluationResult(Base):
    __tablename__ = "evaluation_results"
    
    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    evaluation_type = Column(String(50), nullable=False)  # toxicity, pii, bias, hallucination, safety
    score = Column(Float, nullable=False)
    details = Column(JSON, nullable=True)  # Store additional model-specific details
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<EvaluationResult(id={self.id}, type={self.evaluation_type}, score={self.score})>"
