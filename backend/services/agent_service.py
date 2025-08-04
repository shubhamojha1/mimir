from ..agents.base_agent import SupervisorAgent, AgentConfig, IntentType
from .rag_service import RAGService

from typing import Dict

class AgentService:
    """Service to manage agent instances and configurations"""

    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        self.available_agents = self._configure_agents()
        self.supervisor = SupervisorAgent(rag_service, self.available_agents)

    def _configure_agents(self) -> Dict[str, AgentConfig]:
        """Configure available agents and their capabilities"""
        return {
            "general_agent": AgentConfig(
                agent_type="general",
                supported_intents=[IntentType.QUESTION_ANSWER]
            ),
            "rag_agent": AgentConfig(
                agent_type="rag",
                supported_intents=[
                    IntentType.QUESTION_ANSWER,
                    IntentType.SUMMARIZE,
                    IntentType.ANALYZE
                ]
            ),
            # TODO: Add other agents as needed in future
        }
    
    async def process_chat_query(self, query: str, context: Dict = None) -> Dict:
        """Process chat query through supervisor agent"""
        return await self.supervisor.process_query(query, context)