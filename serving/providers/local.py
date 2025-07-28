import aiohttp
from typing import Dict, Any

from ..base import LLMProvider, LLMRequest, LLMResponse


class LocalProvider(LLMProvider):
    """Provider for local GPU serving (vLLM or SGLang)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:8000")
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        payload = {
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "model": request.model
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/completions",
                json=payload
            ) as response:
                response.raise_for_status()
                data = await response.json()
        
        return LLMResponse(
            text=data["choices"][0]["text"],
            model=request.model,
            provider="local"
        )