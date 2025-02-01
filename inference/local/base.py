from abc import ABC, abstractmethod
from typing import List, Dict, Union
import torch

class BaseModel(ABC):
    """Abstract base class for all models"""
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.processor = None
        
    @abstractmethod
    def _load(self) -> None:
        """Load model and processor"""
        pass
    
    @abstractmethod
    def inference(self, msgs: List[Dict], num_samples: int = 1) -> Union[str, List[str]]:
        """Run inference"""
        pass
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()