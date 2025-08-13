import aiohttp
from typing import Dict, Any, Optional, List

from ..base import LLMProvider, LLMRequest, LLMResponse


class OpenRouterProvider(LLMProvider):
    """Provider client for OpenRouter-compatible API.

    Expects base_url to include the version segment, e.g.:
      - https://openrouter.ai/api/v1
      - http://localhost:8080/v1
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url: str = config.get("base_url", "http://localhost:8080/v1").rstrip("/")
        self.api_key: Optional[str] = config.get("api_key")
        # Optional attribution headers used by OpenRouter; harmless for local
        self.http_referer: Optional[str] = config.get("http_referer")
        self.x_title: Optional[str] = config.get("x_title")

    async def generate(self, request: LLMRequest) -> LLMResponse:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.x_title:
            headers["X-Title"] = self.x_title

        payload = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            ) as response:
                response.raise_for_status()
                data = await response.json()

        return LLMResponse(
            text=data["choices"][0]["message"]["content"],
            model=request.model,
            provider="openrouter",
        )
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from OpenRouter.
        
        Returns:
            List of model metadata dictionaries following OpenRouter schema
        """
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # For OpenRouter API, models endpoint is at /api/v1/models
        # For our local server, it's at /openrouter/models
        models_url = f"{self.base_url}/models"
        if "localhost" in self.base_url or "127.0.0.1" in self.base_url:
            # Use our custom endpoint for local server
            models_url = self.base_url.replace("/v1", "") + "/openrouter/models"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(models_url, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
        
        # OpenRouter returns {"data": [...models...]}
        return data.get("data", [])

