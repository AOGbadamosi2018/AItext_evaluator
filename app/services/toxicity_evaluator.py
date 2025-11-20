import numpy as np
from typing import Dict, Any, List
from transformers import pipeline
from loguru import logger
from .base_evaluator import BaseEvaluator

class ToxicityEvaluator(BaseEvaluator):
    """Evaluator for detecting toxic content in text."""
    
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        super().__init__(model_name)
        self.labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    
    async def _load_model(self):
        """Load the toxicity classification model."""
        try:
            self.pipeline = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=-1  # Use CPU by default
            )
        except Exception as e:
            logger.error(f"Error loading toxicity model: {e}")
            raise
    
    async def evaluate(self, text: str) -> Dict[str, Any]:
        """
        Evaluate text for toxicity.
        
        Args:
            text: The text to evaluate
            
        Returns:
            Dict containing toxicity scores and classification
        """
        if not self._is_initialized:
            await self.initialize()
        
        try:
            result = self.pipeline(
                text,
                candidate_labels=self.labels,
                multi_label=True
            )
            
            # Process results
            scores = {
                label: float(score) 
                for label, score in zip(result["labels"], result["scores"])
            }
            
            # Calculate overall toxicity score (max of all toxicity types)
            overall_score = max(scores.values()) if scores else 0.0
            
            return {
                "score": overall_score,
                "scores": scores,
                "is_toxic": overall_score > 0.5,
                "evaluation_type": "toxicity"
            }
            
        except Exception as e:
            logger.error(f"Error in toxicity evaluation: {e}")
            return {
                "score": 0.0,
                "scores": {label: 0.0 for label in self.labels},
                "is_toxic": False,
                "error": str(e),
                "evaluation_type": "toxicity"
            }
