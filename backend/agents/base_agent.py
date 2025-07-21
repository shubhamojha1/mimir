from abc import ABC, abstractmethod
import logging
from enum import Enum
from typing import Any, List, Dict, Optional, TypedDict, Annotated
from dataclasses import dataclass, asdict
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

class IntentType(Enum):
    """Supported Intent Types"""
    QUESTION_ANSWER = "question_answer"
    SUMMARIZE = "summarize" 
    ANALYZE = "analyze"
    COMPARE = "compare"
    STUDY_GUIDE = "study_guide"
    PRACTICE_PROBLEMS = "practice_problems"
    MATH_SOLVE = "math_solve"
    CODE_REVIEW = "code_review"
    RESEARCH = "research"
    CLARIFICATION = "clarification"
    # TODO: Add more intents (podcast generation / etc.)

@dataclass 
class AgentConfig:
    """Configuration for individual agents"""
    agent_type: str
    supported_intents: List[IntentType]
    parallal_capable: bool = True
    dependencies: List[str] = None
    confidence_threshold: float = 0.7

class IntentClassifier:
    """Intent classification with confidence scoring"""
    # TODO: will add intent classifier with finetuned Setfit model
    def __init__(self):
        # Intent patterns and keywords
        self.intent_patterns = {
            IntentType.QUESTION_ANSWER: [
                "what", "how", "why", "when", "where", "explain", "tell me"
            ],
            IntentType.SUMMARIZE: [
                "summarize", "summary", "key points", "main ideas", "overview"
            ],
            IntentType.ANALYZE: [
                "analyze", "analysis", "examine", "evaluate", "assess", "critique"
            ],
            IntentType.COMPARE: [
                "compare", "contrast", "difference", "similarity", "versus", "vs"
            ],
            IntentType.STUDY_GUIDE: [
                "study guide", "notes", "prepare for", "exam", "test", "review"
            ],
            IntentType.PRACTICE_PROBLEMS: [
                "practice", "problems", "exercises", "quiz", "questions"
            ],
            IntentType.MATH_SOLVE: [
                "solve", "calculate", "equation", "formula", "derivative", "integral"
            ],
            IntentType.CODE_REVIEW: [
                "code", "debug", "review", "programming", "function", "algorithm"
            ]
        }
    
    def classify(self, query: str) -> tuple[IntentType, float]:
        """Classify intent with confidence score"""
        query_lower = query.lower()
        scores = {}
        
        for intent, keywords in self.intent_patterns.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                scores[intent] = score / len(keywords)
        
        if not scores:
            return IntentType.QUESTION_ANSWER, 0.3
        
        best_intent = max(scores, key=scores.get)
        confidence = min(scores[best_intent] * 2, 1.0)  # Scale confidence
        
        return best_intent, confidence

class SupervisorAgent:
    """
    Supervisor agent that orchestrates the entire workflow
    """
    def __init__(self, rag_service, available_agents: Dict[str, AgentConfig]):
        self.rag_service = rag_service
        self.available_agents = available_agents
        self.intent_classifier = IntentClassifier()
        self.confidence_threshold = 0.6

        
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