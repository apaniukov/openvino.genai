"""
LLM Inference Engine for Research Organizer
Uses OpenVINO GenAI for efficient local inference.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import openvino_genai as ov_genai


class LLM:
    """
    Global LLM instance for the research organizer.
    Handles model initialization, inference, and structured output generation.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "CPU",
        **generation_config
    ):
        """
        Initialize the LLM with OpenVINO GenAI.
        
        Args:
            model_path: Path to the model directory
            device: Device to run inference on (CPU, GPU, AUTO)
            **generation_config: Additional generation parameters
        """
        self.model_path = Path(model_path)
        self.device = device
        
        if not self.model_path.exists():
            raise ValueError(
                f"Model path does not exist: {self.model_path}"
            )
        
        # Initialize OpenVINO GenAI pipeline
        self.pipe = ov_genai.LLMPipeline(
            str(self.model_path),
            device=self.device
        )
        
        # Default generation configuration
        self.default_config = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True,
        }
        self.default_config.update(generation_config)
        
        print(f"✓ LLM initialized: {self.model_path.name}")
        print(f"   Device: {self.device}")
    
    def generate(
        self,
        prompt: Union[List[str], ov_genai.ChatHistory],
        generation_config: Optional[ov_genai.GenerationConfig] = None,
    ) -> ov_genai.GenerationResult:
        return self.pipe.generate(prompt, generation_config=generation_config)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_path": str(self.model_path),
            "model_name": self.model_path.name,
            "device": self.device,
            "default_config": self.default_config,
        }


# Global LLM instance
_llm_instance: Optional[LLM] = None


def init_llm(model_path: str, device: str = "CPU", **kwargs) -> LLM:
    """
    Initialize the global LLM instance.
    
    Args:
        model_path: Path to model directory
        device: Device to use (CPU, GPU, AUTO)
        **kwargs: Additional generation config
    
    Returns:
        Initialized LLM instance
    """
    global _llm_instance
    _llm_instance = LLM(model_path, device, **kwargs)
    return _llm_instance


def get_llm() -> LLM:
    """
    Get the global LLM instance.
    
    Returns:
        LLM instance or None if not initialized
    """
    if _llm_instance is None:
        raise RuntimeError(
            "LLM not initialized. Call init_llm() first."
        )
    return _llm_instance


def is_llm_initialized() -> bool:
    """
    Check if LLM is initialized.
    
    Returns:
        True if LLM is ready
    """
    return _llm_instance is not None
