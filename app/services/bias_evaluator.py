from typing import Dict, Any, List, Tuple
import numpy as np
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from .base_evaluator import BaseEvaluator
from loguru import logger
import tensorflow as tf

class BiasEvaluator(BaseEvaluator):
    """Evaluator for detecting bias in text."""
    
    def __init__(self, model_name: str = "d4data/bias-detection-model"):
        super().__init__(model_name)
        self.bias_categories = [
            "gender", "race_ethnicity", "religion", "age", 
            "nationality", "sexual_orientation", "disability", 
            "social_class", "political", "other"
        ]
    
    async def _load_model(self):
        """Load the bias detection model using TensorFlow."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # Load the TensorFlow model (no need for from_tf=True with TFAutoModelForSequenceClassification)
            self.model = TFAutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
        except Exception as e:
            logger.error(f"Error loading bias detection model: {e}")
            raise
    
    async def evaluate(self, text: str) -> Dict[str, Any]:
        """
        Evaluate text for potential biases.
        
        Args:
            text: The text to evaluate
            
        Returns:
            Dict containing bias detection results
        """
        if not self._is_initialized:
            await self.initialize()
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length"
            )
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.sigmoid(logits).squeeze().numpy()
            
            # Process results
            bias_scores = {
                cat: float(score) 
                for cat, score in zip(self.bias_categories, probs)
            }
            
            # Calculate overall bias score (max of all bias categories)
            overall_score = max(bias_scores.values()) if bias_scores else 0.0
            
            # Get top bias categories
            sorted_biases = sorted(
                bias_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            return {
                "score": overall_score,
                "bias_scores": bias_scores,
                "top_biases": [{"category": k, "score": v} for k, v in sorted_biases[:3]],
                "has_bias": overall_score > 0.5,
                "evaluation_type": "bias"
            }
            
        except Exception as e:
            logger.error(f"Error in bias evaluation: {e}")
            return {
                "score": 0.0,
                "bias_scores": {cat: 0.0 for cat in self.bias_categories},
                "top_biases": [],
                "has_bias": False,
                "error": str(e),
                "evaluation_type": "bias"
            }
    
    def _extract_biased_phrases(self, text: str, category: str) -> List[Dict[str, Any]]:
        """Extract phrases that might be contributing to the bias score.
        
        This is a simplified implementation. In a production environment, you might
        want to use more sophisticated techniques like LIME or SHAP for explanation.
        """
        # Simple implementation: split into sentences and score each
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        phrase_scores = []
        for sentence in sentences:
            if not sentence:
                continue
                
            # Get score for this sentence
            try:
                inputs = self.tokenizer(
                    sentence,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding="max_length"
                )
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probs = torch.sigmoid(logits).squeeze().numpy()
                
                # Get the score for the requested category
                category_idx = self.bias_categories.index(category)
                score = float(probs[category_idx])
                
                if score > 0.5:  # Only include if score is above threshold
                    phrase_scores.append({
                        "text": sentence,
                        "score": score
                    })
                    
            except Exception as e:
                logger.warning(f"Error processing sentence for bias: {e}")
                continue
        
        # Sort by score descending
        return sorted(phrase_scores, key=lambda x: x["score"], reverse=True)[:3]
