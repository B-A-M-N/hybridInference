import aiohttp
from typing import Dict, Any

from ..base import LLMProvider, LLMRequest, LLMResponse


class LlamaProvider(LLMProvider):
    """Provider for Meta's Llama API"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config["api_key"]
        self.base_url = config.get("base_url", "https://api.llama.com/compat/v1/")
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers
            ) as response:
                response.raise_for_status()
                data = await response.json()
        
        return LLMResponse(
            text=data["choices"][0]["message"]["content"],
            model=request.model,
            provider="llama"
        )