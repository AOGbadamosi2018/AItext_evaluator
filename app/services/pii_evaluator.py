from typing import Dict, Any, List, Tuple
import re
from transformers import pipeline
from .base_evaluator import BaseEvaluator
from loguru import logger

class PIIEvaluator(BaseEvaluator):
    """Evaluator for detecting Personally Identifiable Information (PII) in text."""
    
    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        super().__init__(model_name)
        self.pii_types = [
            "PERSON", "ORGANIZATION", "LOCATION", "DATE",
            "PHONE", "EMAIL", "IP_ADDRESS", "CREDIT_CARD",
            "SSN", "DRIVER_LICENSE"
        ]
        self.regex_patterns = {
            "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "PHONE": r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            "IP_ADDRESS": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            "CREDIT_CARD": r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11})\b',
            "SSN": r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b',
            "DRIVER_LICENSE": r'\b[A-Za-z]\d{4,8}\b',
        }
    
    async def _load_model(self):
        """Load the NER model for PII detection."""
        try:
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model_name,
                aggregation_strategy="simple",
                device=-1  # Use CPU by default
            )
        except Exception as e:
            logger.error(f"Error loading PII model: {e}")
            raise
    
    async def evaluate(self, text: str) -> Dict[str, Any]:
        """
        Evaluate text for PII content.
        
        Args:
            text: The text to evaluate
            
        Returns:
            Dict containing PII detection results
        """
        if not self._is_initialized:
            await self.initialize()
        
        try:
            # Use NER model for entity detection
            ner_results = self.ner_pipeline(text)
            
            # Check for PII using regex patterns
            regex_matches = self._check_with_regex(text)
            
            # Combine results
            detected_entities = self._process_ner_results(ner_results)
            detected_entities.update(regex_matches)
            
            # Calculate overall PII score
            pii_score = min(1.0, len(detected_entities) * 0.2)  # Cap at 1.0
            
            return {
                "score": pii_score,
                "detected_pii": detected_entities,
                "has_pii": len(detected_entities) > 0,
                "evaluation_type": "pii"
            }
            
        except Exception as e:
            logger.error(f"Error in PII evaluation: {e}")
            return {
                "score": 0.0,
                "detected_pii": {},
                "has_pii": False,
                "error": str(e),
                "evaluation_type": "pii"
            }
    
    def _check_with_regex(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Check for PII using regex patterns."""
        results = {}
        for pii_type, pattern in self.regex_patterns.items():
            matches = [
                {"text": match.group(), "start": match.start(), "end": match.end()}
                for match in re.finditer(pattern, text, re.IGNORECASE)
            ]
            if matches:
                results[pii_type] = matches
        return results
    
    def _process_ner_results(self, ner_results: List[Dict]) -> Dict[str, List[Dict[str, Any]]]:
        """Process NER model results into a structured format."""
        entities = {}
        for entity in ner_results:
            entity_type = entity["entity_group"]
            if entity_type in self.pii_types:
                if entity_type not in entities:
                    entities[entity_type] = []
                entities[entity_type].append({
                    "text": entity["word"],
                    "score": float(entity["score"]),
                    "start": entity["start"],
                    "end": entity["end"]
                })
        return entities
