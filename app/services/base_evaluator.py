from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from loguru import logger

class BaseEvaluator(ABC):
    """Base class for all text evaluators."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._is_initialized = False
    
    async def initialize(self):
        """Initialize the model and tokenizer."""
        if not self._is_initialized:
            logger.info(f"Initializing {self.__class__.__name__} with model {self.model_name}")
            await self._load_model()
            self._is_initialized = True
    
    @abstractmethod
    async def _load_model(self):
        """Load the model and tokenizer."""
        pass
    
    @abstractmethod
    async def evaluate(self, text: str) -> Dict[str, Any]:
        """Evaluate the given text and return a dictionary with results."""
        pass
    
    def _calculate_score(self, result: Dict[str, Any]) -> float:
        """Calculate a normalized score (0-1) from the evaluation result."""
        # Default implementation, can be overridden by subclasses
        return result.get('score', 0.0)
