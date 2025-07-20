from abc import ABC, abstractmethod
import logging
from typing import Any, List, Dict, Optional, TypedDict, Annotated
import operator

from ..services.rag_service import RAGService
from ..config.settings import get_settings
from ..utils.prompts import AgentPrompts

logger = logging.getLogger(__name__)

# State management
class AgentState(TypedDict):
    """Central state shared across all agents in the graph"""
    user_query: str
    intent: Optional[str]
    intent_confidence: float
    agent_responses: Dict[str, Any]
    context_data: List[Dict[str, Any]]
    clatifications_needed: List[str]
    processing_steps: List[Dict[str, Any]]
    final_response: str
    metadata: Dict[str, Any]
    # Reducer for agent_responses to accumulate results
    agent_responses: Annotated[Dict, operator.add]

# class BaseAgent(ABC):
#     """
#     Defines the common interface and shared functionality that all agents inherit.
#     Ensures consistency while allowing for domain-specific customization.
#     Receives communication from the intent_detection router.

#     Key responsibilities:
#     1. Task decomposition and workflow management
#     2. RAG integratyion for grounded repsonses
#     3. Multi-component prompting for complex reasoning
#     4. Quality assessment and confidence scoring
#     5. Content management and memory
#     """
#     def __init__(self, agent_type: str, rag_service: RAGService):
#         self.agent_type = agent_type
#         self.rag_service = rag_service
#         self.settings = get_settings()
#         self.prompts = AgentPrompts()
        
#         # Agent-specific configuration
#         self.max_context_length = 4000  # Maximum context to include in prompts
#         self.confidence_threshold = 0.6  # Minimum confidence for responses
        
#         # Initialize agent-specific settings
#         self._initialize_agent()
        
#         logger.info(f"{self.agent_type} agent initialized")