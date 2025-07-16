from typing import Optional


class RAGPrompts:
    """
    Manages prompt templates for the entire project.
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