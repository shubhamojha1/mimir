from typing import Optional


class RAGPrompts:
    """
    Manages prompt templates for the RAG system.
    """
    def get_rag_system_prompt(self, agent_type: Optional[str] = None) -> str:
        """
        Generate the system prompt for RAG response generation.
        
        Args:
            agent_type: Optional specialization (e.g., "math", "science", "legal", or something else)
            
        Returns:
            str: Customized system prompt
        """
        base_prompt = """You are a knowledgeable AI assistant that helps users understand their documents.
                    Your responses must be:
                    1. Accurate and grounded in the provided context
                    2. Clear and concise
                    3. Properly cited with source references
                    4. Professional in tone

                    When responding:
                    - If the context contains the information, use it to provide specific, detailed answers
                    - If the context doesn't contain enough information, clearly state this limitation
                    - If you're unsure, express your uncertainty rather than making assumptions
                    - Always cite your sources using the provided reference numbers

                    Never make up information or rely on knowledge outside the provided context."""
        
        if agent_type:
            specialization_prompts = {
                "math": "\nYou specialize in mathematical concepts and calculations. Explain mathematical ideas clearly and step-by-step.",
                "science": "\nYou specialize in scientific topics. Use precise scientific terminology and explain complex concepts clearly.",
                "legal": "\nYou specialize in legal documents. Use appropriate legal terminology while making concepts accessible.",
                "technical": "\nYou specialize in technical documentation. Break down complex technical concepts into understandable parts."
            }
            base_prompt += specialization_prompts.get(agent_type, 
                                                      f"\nYou specialize in {agent_type} related topics.")

        return base_prompt
    
    def get_rag_user_prompt(self, query: str, context: str) -> str:
        """
        Generate the user prompt combining the query and retrieved context.
        
        Args:
            query: User's question
            context: Retrieved document chunks with source information
            
        Returns:
            str: Formatted prompt for the LLM
        """
        return f"""Please answer the following question based on the provided context.

                Context:
                {context}

                Question:
                {query}

                Instructions:
                1. Answer the question using only information from the provided context
                2. Cite sources using [Source X] notation when referencing specific information
                3. If the context doesn't contain relevant information, say so clearly
                4. Keep your response clear and concise

                Answer:"""
    
class SupervisorPrompts:
    """Manages prompt templates for all the agents"""
    def chain_of_thought_prompt(query) -> str:
        cot_prompt = f"""
                I need to carefully analyze this user query to understand their intent. Let me think through this step by step.

                Query: "{query}"

                Let me break this down:

                Step 1: What are the key words and phrases in this query?
                - Identify the main action words (verbs)
                - Identify the subject matter (nouns)
                - Look for intent indicators (question words, request patterns)

                Step 2: What is the user trying to accomplish?
                - Are they seeking information? (question_answer)
                - Do they want content condensed? (summarize)
                - Do they need detailed examination? (analyze)
                - Are they comparing things? (compare)
                - Do they need study materials? (study_guide)
                - Do they want practice exercises? (practice_problems)
                - Are they solving math problems? (math_solve)
                - Do they need code help? (code_review)
                - Are they doing research? (research)
                - Do they need clarification? (clarification)

                Step 3: What evidence supports each possible intent?
                - List specific words/phrases that indicate each intent
                - Consider the overall context and tone

                Step 4: What intent best matches the evidence?
                - Compare the strength of evidence for each intent
                - Consider which intent would best serve the user's needs

                Step 5: How confident am I in this classification?
                - Strong evidence and clear intent = high confidence (0.8-1.0)
                - Some evidence but could be multiple intents = medium confidence (0.5-0.7)
                - Weak evidence or very ambiguous = low confidence (0.3-0.5)

                Now let me work through these steps:
            
            Step 1 - Key words and phrases:
            [Analyze the query here]
            
            Step 2 - User's goal:
            [Determine what user wants to accomplish]
            
            Step 3 - Evidence for each intent:
            [List evidence for different intents]
            
            Step 4 - Best matching intent:
            [Choose the best intent with reasoning]
            
            Step 5 - Confidence assessment:
            [Assess confidence with reasoning]
            
            Final answer: [intent_type],[confidence_score]        
        """
        return cot_prompt
    
    def clarification_prompt(query, clarifications) -> str:
        prompt = f"""
        The following clarifications are needed to better answer the user's question:
        
        Original Query: {query}
        Clarifications Needed: {clarifications}
        
        Please provide reasonable assumptions or default interpretations for these clarifications
        so we can proceed with generating a response.
        """
        return prompt

    def consolidation_prompt(query, intent) -> str:
        prompt = f"""
        Consolidate the following agent responses into a single, coherent response 
        that addresses the user's original query.
        
        Original Query: {query}
        Intent: {intent}
        
        Agent Responses:
        """
        return prompt
    
    def additional_consolidation_prompt() -> str:
        prompt = """
                
                Please create a unified response that:
                1. Directly addresses the user's question
                2. Integrates insights from all agents
                3. Is well-structured and coherent
                4. Cites sources appropriately
                5. Provides actionable information where relevant
                """
        return prompt