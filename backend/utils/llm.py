from typing import Optional
import logging

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Object-oriented wrapper for generating LLM responses using different providers.
    Supports OpenAI, Anthropic, and Ollama.
    """

    def __init__(
        self,
        provider: str,
        openai_client: Optional[object] = None,
        anthropic_client: Optional[object] = None,
        ollama_host: Optional[str] = None,
    ):
        self.provider = provider.lower()
        self.openai_client = openai_client
        self.anthropic_client = anthropic_client
        self.ollama_host = ollama_host

    async def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate a response using the configured LLM provider.
        """
        if self.provider == "openai":
            return await self._generate_openai_response(system_prompt, user_prompt)
        elif self.provider == "anthropic":
            return await self._generate_anthropic_response(system_prompt, user_prompt)
        elif self.provider == "ollama":
            return await self._generate_ollama_response(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    async def _generate_openai_response(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    async def _generate_anthropic_response(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = await self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    async def _generate_ollama_response(self, system_prompt: str, user_prompt: str) -> str:
        import aiohttp

        url = f"{self.ollama_host}/api/generate"
        payload = {
            "model": "llama3.2",
            "prompt": f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:",
            "stream": False
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    return data.get("response", "")
        except Exception as e:
            logger.error(f"Ollama API error: {e}")