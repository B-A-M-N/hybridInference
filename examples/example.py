import asyncio
import sys
sys.path.append('..')  # Add parent directory to path

from serving.base import LLMRequest
from serving.config import get_config
from serving.providers.llama import LlamaProvider


async def main():
    # Load configuration
    config = get_config()
    
    # Initialize provider
    llama_provider = LlamaProvider(config["llama"])
    
    # Create a request
    request = LLMRequest(
        prompt="What is the capital of France?",
        model="Llama-3.3-8B-Instruct",
        max_tokens=50,
        temperature=0.7
    )
    
    # Generate using llama provider directly
    response = await llama_provider.generate(request)
    print(f"Provider: {response.provider}")
    print(f"Response: {response.text}")


if __name__ == "__main__":
    asyncio.run(main())