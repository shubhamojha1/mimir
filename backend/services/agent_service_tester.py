from backend.services.agent_service import AgentService
from backend.services.rag_service import RAGService
import asyncio
import logging

# Command for testing: python -m backend.services.agent_service_tester

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler("agent_service_test.log")
                    ])
logger = logging.getLogger(__name__)

class AgentServiceTester:
    """
    Test suite for the AgentService to ensure the supervisor and agent orchestration
    works as expected.
    """
    def __init__(self):
        # Initialize the necessary services
        self.rag_service = RAGService()
        self.agent_service = AgentService(self.rag_service)
        
    async def test_process_chat_query(self):
        """
        Test the main entry point of the agent service with various queries
        to trigger different intents and workflows.
        """
        test_queries = {
            "Question Answering": "What is the difference between supervised and unsupervised learning?",
            "Summarization": "Summarize the key concepts of Large Language Models.",
            "Analysis": "Analyze the impact of AI on software development.",
            "Unsupported Intent": "Generate a picture of a cat."
        }

        for intent, query in test_queries.items():
            logger.info(f"------------------------------------------------ Testing Intent: {intent} ------------------------------------------------")
            logger.info(f"Query: {query}")
            
            try:
                # Process the query using the agent service
                result = await self.agent_service.process_chat_query(query)
                
                # Log the detailed results
                logger.info(f"Final Response: {result.get('response')}")
                logger.info(f"Detected Intent: {result.get('intent')}")
                logger.info(f"Intent Confidence: {result.get('confidence')}")
                
                logger.info("Processing Steps:")
                for step in result.get('processing_steps', []):
                    logger.info(f"  - {step}")
                
                logger.info(f"--- Test for '{intent}' complete ---")

            except Exception as e:
                logger.error(f"An error occurred during the test for intent '{intent}': {e}", exc_info=True)

async def main():
    """
    Main function to run the agent service tests.
    """
    logger.info("Initializing Agent Service Test Suite...")
    tester = AgentServiceTester()
    
    logger.info("Starting tests for process_chat_query...")
    await tester.test_process_chat_query()
    logger.info("All tests complete.")

if __name__ == "__main__":
    # This allows running the tester as a standalone script
    asyncio.run(main())
