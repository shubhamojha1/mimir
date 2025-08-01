from abc import ABC, abstractmethod
import logging
from enum import Enum
from typing import Any, List, Dict, Optional, TypedDict, Annotated
from dataclasses import dataclass, asdict
from datetime import datetime
import operator

from ..services.rag_service import RAGService
from ..config.settings import get_settings
from ..utils.prompts import SupervisorPrompts

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langgraph.checkpoint.memory import MemorySaver

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

        # Build the langgraph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the langgraph state machine"""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("classify_intent", self.classify_intent)
        workflow.add_node("llm_fallback", self.llm_fallback_classification)
        workflow.add_node("route_agents", self.route_to_agents)
        workflow.add_node("execute_agents", self.execute_agents)
        workflow.add_node("handle_clarification", self.handle_clarification)
        workflow.add_node("validate_output", self.validate_output)
        workflow.add_node("consolidate_response", self.consolidate_response)

        # Add edges with conditional routing
        workflow.set_entry_point("classify_intent")

        workflow.add_conditional_edges(
            "classify_intent",
            self.should_use_llm_fallback,
            {
                "use_llm": "llm_fallback",
                "proceed": "route_agents"
            }
        )

        workflow.add_edge("llm_fallback", "route_agents")
        workflow.add_edge("route_agents", "execute_agents")

        workflow.add_conditional_edges(
            "execute_agents",
            self.needs_clarification,
            {
                "clarify": "handle_clarification",
                "validate": "validate_output"
            }
        )

        workflow.add_edge("handle_clarification", "execute_agents")
        workflow.add_edge("validate_output", "consolidate_response")
        workflow.add_edge("consolidate_response", END)

        # memory for state persistence
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    async def classify_intent(self, state: AgentState) -> AgentState:
        """Node: Classify user intent"""
        user_query = state["user_query"]
        intent, confidence = self.intent_classifier.classify(user_query)

        state["intent"] = intent.value
        state["intent_confidence"] = confidence
        state["processing_steps"].append({
            "step": "intent_classification",
            "intent": intent.value,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })

        logger.info(f"Classified intent: {intent.value} (confidence: {confidence:.2f})")
        return state
    
    async def llm_fallback_classification(self, state: AgentState) -> AgentState:
        """Node: Use LLM for better intent classification when confidence is low"""
        user_query = state["user_query"]
        
        # using chain of thought prompting
        cot_prompt = SupervisorPrompts.chain_of_thought_prompt(user_query)

        return Any
    
    async def route_to_agents(self, state: AgentState) -> AgentState:
        """Node: Determine which agents to use and their execution plan"""
        intent = IntentType(state["intent"])

        # mapping intent to agents
        agent_mapping = {
            IntentType.QUESTION_ANSWER: ["general_agent", "rag_agent"],
            IntentType.SUMMARIZE: ["summarization_agent", "rag_agent"],
            IntentType.ANALYZE: ["analysis_agent", "rag_agent"],
            IntentType.COMPARE: ["comparison_agent", "rag_agent"],
            IntentType.STUDY_GUIDE: ["study_guide_agent", "content_agent"],
            IntentType.PRACTICE_PROBLEMS: ["problem_generator_agent"],
            IntentType.MATH_SOLVE: ["math_agent", "solver_agent"],
            IntentType.CODE_REVIEW: ["code_agent", "review_agent"]
        }

        required_agents = agent_mapping.get(intent, ["general_agent"])

        # filter available agents
        available_agents = [
            agent for agent in required_agents 
            if agent in self.available_agents
        ]

        # determine execution strategy (parallel vs sequential)
        execution_plan = self._create_execution_plan(available_agents, intent)

        state["metadata"]["required_agents"] = available_agents
        state["metadata"]["execution_plan"] = execution_plan
        state["processing_steps"].append({
            "step": "agent_routing",
            "agents": available_agents,
            "execution_plan": execution_plan,
            "timestamp": datetime.now().isoformat()
        })

        return state
    
    async def execute_agents(self, state: AgentState) -> AgentState:
        """Node: Execute the required agentrs according to the plan"""
        agents = state["metadata"]["required_agents"]
        execution_plan = state["metadata"]["execution_plan"]

        # Retrieve context for agents
        contexts = await self._retrieve_context_for_agents()
        state["context_data"] = contexts

        agent_responses = {}

        if execution_plan["type"] == "parallel":
            # execute agents in parallel
            agent_responses = await self._execute_agents_parallel(
                agents, state, contexts
            )
        else:
            # execute agents sequentially
            agent_responses = await self._execute_agents_sequential(
                agents, state, contexts, execution_plan["sequence"]
            )
        
        state["agent_responses"].update(agent_responses)
        state["processing_steps"].append({
            "step": "agent_execution",
            "executed_agents": list(agent_responses.keys()),
            "timestamp": datetime.now().isoformat()
        })

        return state
    
    async def handle_clarification(self, state: AgentState) -> AgentState:
        """Node: Handle clarification requests from agents"""
        clarifications = state["clarifications_needed"]
        
        # Generate clarification response using LLM
        clarification_prompt = SupervisorPrompts.clarification_prompt(
                                        state['user_query'], clarifications)
        
        try:
            clarification_response = await self.rag_service._generate_llm_response(
                "You help resolve ambiguities in user queries.",
                clarification_prompt
            )
            
            # Store clarification for use in next agent execution
            state["metadata"]["clarifications_resolved"] = clarification_response
            state["clarifications_needed"] = []  # Clear clarifications
            
            state["processing_steps"].append({
                "step": "clarification_handling",
                "clarifications": clarifications,
                "resolution": clarification_response[:200] + "...",
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error handling clarifications: {e}")
            state["clarifications_needed"] = []  # Clear to prevent loop
        
        return state
    
    async def validate_output(self, state: AgentState) -> AgentState:
        """Node: Validate the combined agent outputs against user intent"""
        intent = state["intent"]
        agent_responses = state["agent_responses"]
        
        # Basic validation checks
        validation_results = {
            "has_responses": len(agent_responses) > 0,
            "intent_addressed": True,  # More sophisticated check would go here
            "quality_threshold_met": True,  # Quality assessment would go here
            "completeness_check": True  # Completeness assessment would go here
        }
        
        state["metadata"]["validation_results"] = validation_results
        state["processing_steps"].append({
            "step": "output_validation",
            "validation_results": validation_results,
            "timestamp": datetime.now().isoformat()
        })
        
        return state
    
    async def consolidate_response(self, state: AgentState) -> AgentState:
        """Node: Consolidate all agent responses into final response"""
        agent_responses = state["agent_responses"]
        intent = state["intent"]
        
        # Create consolidation prompt
        consolidation_prompt = SupervisorPrompts.consolidation_prompt(state['user_query'], intent)
        
        for agent_name, response in agent_responses.items():
            consolidation_prompt += f"\n{agent_name}: {response}\n"
        
        consolidation_prompt += SupervisorPrompts.additonal_consoludation_prompt()
        
        try:
            final_response = await self.rag_service._generate_llm_response(
                "You are expert at consolidating multiple perspectives into coherent responses.",
                consolidation_prompt
            )
            
            state["final_response"] = final_response
            state["processing_steps"].append({
                "step": "response_consolidation",
                "agents_consolidated": list(agent_responses.keys()),
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error consolidating responses: {e}")
            # Fallback: return primary agent response
            primary_response = next(iter(agent_responses.values()))
            state["final_response"] = primary_response
        
        return state
    
    # helper functions
    def _create_execution_plan(self, agents: List[str], intent: IntentType) -> Dict[str, Any]:
        """Create execution plan for agents"""
        # TODO: simple logic for now.. need to implement better one
        if len(agents) <= 2:
            return {"type": "parallel"}
        else:
            return {
                "type": "sequential",
                "sequence": agents # TODO: could (how?) implement dependency based ordering
            }
        
    async def _retrieve_context_for_agents(self, state: AgentState) -> List[Dict[str, Any]]:
        """Retrieve relevant context for agent execution"""
        # TODO: need to decide what context to retrieve
        # could involve communicating with other agents
        # not just rag retrieval
        try:
            contexts = await self.rag_service.retrieve_relevant_context(
                query=state["user_query"],
                filters={},
                max_results=5
            )
            return [asdict(ctx) for ctx in contexts]
        except Exception as e:
            logger.error(f"Error retrieving context")

    async def _execute_agents_in_parallel(self,
                                          agents: List[str],
                                          state: AgentState,
                                          contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute agents in parallel"""
        # Implementation would use asyncio.gather or similar
        responses = {}
        for agent in agents:
            try:
                response = await self._execute_single_agent(agent, state, contexts)
                responses[agent] = response
            except Exception as e:
                logger.error(f"Error executing agent {agent}: {e}")
                responses[agent] = f"Error: {str(e)}"
        
        return responses
    
    async def _execute_agents_sequential(self, 
                                       agents: List[str], 
                                       state: AgentState, 
                                       contexts: List[Dict[str, Any]],
                                       sequence: List[str]) -> Dict[str, str]:
        """Execute agents sequentially"""
        responses = {}
        accumulated_context = contexts.copy()
        
        for agent in sequence:
            try:
                response = await self._execute_single_agent(agent, state, accumulated_context)
                responses[agent] = response
                
                # Add previous responses to context for next agent
                accumulated_context.append({
                    "source": f"agent_{agent}",
                    "content": response,
                    "type": "agent_response"
                })
                
            except Exception as e:
                logger.error(f"Error executing agent {agent}: {e}")
                responses[agent] = f"Error: {str(e)}"
        
        return responses
    
    async def _execute_single_agent(self, 
                                   agent_name: str, 
                                   state: AgentState, 
                                   contexts: List[Dict[str, Any]]) -> str:
        """Execute a single agent"""
        # This would interface with your existing agent implementations
        # For now, using RAG service as placeholder
        
        agent_prompt = f"""
        You are the {agent_name} specialized for handling {state['intent']} tasks.
        
        User Query: {state['user_query']}
        Intent: {state['intent']}
        
        Available Context:
        {contexts}
        
        Please provide your specialized response to this query.
        """
        
        try:
            response = await self.rag_service._generate_llm_response(
                f"You are {agent_name}, a specialized agent.",
                agent_prompt
            )
            return response
        except Exception as e:
            # Check if this agent needs clarification
            if "unclear" in str(e).lower() or "ambiguous" in str(e).lower():
                state["clarifications_needed"].append(f"{agent_name} needs clarification: {str(e)}")
            raise e
    
    # Main execution method
    async def process_query(self, user_query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process user query through the LangGraph workflow"""
        initial_state = AgentState(
            user_query=user_query,
            intent=None,
            intent_confidence=0.0,
            agent_responses={},
            context_data=[],
            clarifications_needed=[],
            processing_steps=[],
            final_response="",
            metadata=context or {}
        )
        
        # Execute the graph
        config = {"thread_id": f"user_session_{datetime.now().timestamp()}"}
        final_state = await self.graph.ainvoke(initial_state, config)
        
        return {
            "response": final_state["final_response"],
            "intent": final_state["intent"],
            "confidence": final_state["intent_confidence"],
            "processing_steps": final_state["processing_steps"],
            "metadata": final_state["metadata"]
        }
    
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
#         self.prompts = SupervisorPrompts()
        
#         # Agent-specific configuration
#         self.max_context_length = 4000  # Maximum context to include in prompts
#         self.confidence_threshold = 0.6  # Minimum confidence for responses
        
#         # Initialize agent-specific settings
#         self._initialize_agent()
        
#         logger.info(f"{self.agent_type} agent initialized")