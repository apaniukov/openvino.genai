"""
Agents Package - LLM and Agent Infrastructure

This package contains LLM-related functionality:
- LLM inference engine (OpenVINO GenAI)
- Prompt templates
- Structured output schemas
- Agent implementations
"""

from .llm import LLM, init_llm, get_llm, is_llm_initialized
from .topic_agent import TopicAgent
from .summarizer_agent import SummarizerAgent
from .orchestrator_agent import OrchestratorAgent

__all__ = [
    # LLM management
    'LLM',
    'init_llm',
    'get_llm',
    'is_llm_initialized',
    # Agents
    'TopicAgent',
    'SummarizerAgent',
    'OrchestratorAgent',
]
