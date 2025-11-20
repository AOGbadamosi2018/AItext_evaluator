from typing import Dict, Any, List, Optional
from loguru import logger
import asyncio

# Import evaluators
from .toxicity_evaluator import ToxicityEvaluator
from .pii_evaluator import PIIEvaluator
from .bias_evaluator import BiasEvaluator
from .hallucination_evaluator import HallucinationEvaluator

class EvaluationService:
    """Service for evaluating text across multiple dimensions."""
    
    def __init__(self):
        self.evaluators = {
            "toxicity": ToxicityEvaluator(),
            "pii": PIIEvaluator(),
            "bias": BiasEvaluator(),
            "hallucination": HallucinationEvaluator(),
        }
        self._initialized = False
    
    async def initialize(self):
        """Initialize all evaluators."""
        if self._initialized:
            return
            
        logger.info("Initializing evaluation service...")
        await asyncio.gather(*[
            evaluator.initialize() 
            for evaluator in self.evaluators.values()
        ])
        self._initialized = True
        logger.info("Evaluation service initialized")
    
    async def evaluate_text(
        self, 
        text: str, 
        context: Optional[str] = None,
        evaluations: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate text across multiple dimensions.
        
        Args:
            text: The text to evaluate
            context: Optional context for hallucination detection
            evaluations: List of evaluations to run (None for all)
            
        Returns:
            Dict containing all evaluation results and an overall safety score
        """
        if not self._initialized:
            await self.initialize()
        
        # Determine which evaluations to run
        if evaluations is None:
            evaluations = list(self.evaluators.keys())
        
        # Run evaluations in parallel
        tasks = []
        for eval_type in evaluations:
            if eval_type in self.evaluators:
                if eval_type == "hallucination":
                    tasks.append(self.evaluators[eval_type].evaluate(text, context))
                else:
                    tasks.append(self.evaluators[eval_type].evaluate(text))
        
        # Wait for all evaluations to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        evaluation_results = {}
        for eval_type, result in zip(evaluations, results):
            if isinstance(result, Exception):
                logger.error(f"Error in {eval_type} evaluation: {result}")
                evaluation_results[eval_type] = {
                    "score": 0.0,
                    "error": str(result),
                    "evaluation_type": eval_type
                }
            else:
                evaluation_results[eval_type] = result
        
        # Calculate overall safety score (0-100)
        safety_score = self._calculate_safety_score(evaluation_results)
        
        return {
            "safety_score": safety_score,
            "evaluations": evaluation_results,
            "text": text,
            "context": context
        }
    
    def _calculate_safety_score(self, evaluation_results: Dict[str, Any]) -> float:
        """Calculate an overall safety score (0-100) based on evaluation results."""
        if not evaluation_results:
            return 0.0
        
        # Define weights for each evaluation type
        weights = {
            "toxicity": 0.4,
            "pii": 0.3,
            "bias": 0.2,
            "hallucination": 0.1
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for eval_type, result in evaluation_results.items():
            if eval_type in weights and "score" in result:
                # Invert scores so higher is better (for all except hallucination)
                if eval_type != "hallucination":
                    score = 1.0 - min(max(result["score"], 0.0), 1.0)
                else:
                    score = min(max(result["score"], 0.0), 1.0)
                
                total_score += score * weights[eval_type]
                total_weight += weights[eval_type]
        
        # Normalize to 0-100 range
        if total_weight > 0:
            safety_score = (total_score / total_weight) * 100
        else:
            safety_score = 0.0
            
        return round(safety_score, 2)

# Global instance of the evaluation service
evaluation_service = EvaluationService()
