from typing import Dict, Any, List, Optional
import numpy as np
from transformers import pipeline
from sentence_transformers import CrossEncoder
from .base_evaluator import BaseEvaluator
from loguru import logger
import re

class HallucinationEvaluator(BaseEvaluator):
    """Evaluator for detecting hallucinations and factual inconsistencies in text."""
    
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-large"):
        super().__init__(model_name)
        self.entailment_threshold = 0.7
        self.contradiction_threshold = 0.7
    
    async def _load_model(self):
        """Load the NLI model for hallucination detection."""
        try:
            # Using a cross-encoder for better accuracy
            self.model = CrossEncoder(self.model_name)
        except Exception as e:
            logger.error(f"Error loading hallucination detection model: {e}")
            raise
    
    async def evaluate(self, text: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate text for potential hallucinations or factual inconsistencies.
        
        Args:
            text: The text to evaluate
            context: Optional context to check against (e.g., source document)
            
        Returns:
            Dict containing hallucination detection results
        """
        if not self._is_initialized:
            await self.initialize()
        
        try:
            # If no context is provided, use a simple heuristic-based approach
            if not context:
                return await self._evaluate_without_context(text)
            
            # Otherwise, use the NLI model to check for contradictions
            return await self._evaluate_with_context(text, context)
            
        except Exception as e:
            logger.error(f"Error in hallucination evaluation: {e}")
            return {
                "score": 0.0,
                "is_hallucinated": False,
                "confidence": 0.0,
                "error": str(e),
                "evaluation_type": "hallucination"
            }
    
    async def _evaluate_without_context(self, text: str) -> Dict[str, Any]:
        """Evaluate text for hallucinations without external context."""
        # Simple heuristics for when no context is available
        score = 0.0
        flags = []
        
        # Check for vague language
        vague_phrases = [
            "some people say", "many believe", "it is known",
            "experts agree", "studies show", "research indicates"
        ]
        
        if any(phrase in text.lower() for phrase in vague_phrases):
            score += 0.3
            flags.append("vague_language")
        
        # Check for numerical claims without sources
        if re.search(r'\d+ (percent|%)', text.lower()):
            score += 0.2
            flags.append("unsourced_statistic")
        
        # Check for superlatives and extreme claims
        extreme_words = ["always", "never", "everyone", "no one", "best", "worst"]
        if any(word in text.lower() for word in extreme_words):
            score += 0.2
            flags.append("extreme_claim")
        
        # Cap the score at 1.0
        score = min(1.0, score)
        
        return {
            "score": score,
            "is_hallucinated": score > 0.5,
            "confidence": score,
            "flags": flags,
            "evaluation_type": "hallucination",
            "note": "Evaluation without context is less reliable"
        }
    
    async def _evaluate_with_context(self, text: str, context: str) -> Dict[str, Any]:
        """Evaluate text against provided context using NLI."""
        # Split text into sentences
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return {
                "score": 0.0,
                "is_hallucinated": False,
                "confidence": 0.0,
                "evaluation_type": "hallucination"
            }
        
        # Prepare sentence pairs for NLI
        sentence_pairs = [(sentence, context) for sentence in sentences]
        
        # Get NLI scores
        nli_scores = self.model.predict(sentence_pairs)
        
        # Process scores (assuming the model returns [contradiction, neutral, entailment] scores)
        contradiction_scores = []
        for score in nli_scores:
            if len(score) == 3:  # contradiction, neutral, entailment
                contradiction_scores.append(score[0])  # Contradiction score
            else:
                # Handle different output formats if needed
                contradiction_scores.append(0.0)
        
        # Calculate overall hallucination score
        avg_contradiction = np.mean(contradiction_scores) if contradiction_scores else 0.0
        is_hallucinated = avg_contradiction > self.contradiction_threshold
        
        # Get problematic sentences
        problematic_sentences = [
            {"sentence": sent, "contradiction_score": float(score)}
            for sent, score in zip(sentences, contradiction_scores)
            if score > self.contradiction_threshold
        ]
        
        return {
            "score": float(avg_contradiction),
            "is_hallucinated": is_hallucinated,
            "confidence": float(avg_contradiction),
            "problematic_sentences": problematic_sentences,
            "evaluation_type": "hallucination"
        }
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Simple sentence splitter."""
        # This is a basic implementation. Consider using a more robust sentence splitter.
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        return sentences
