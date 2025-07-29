import asyncio
import sys
sys.path.append('..')  # Add parent directory to path

from serving.manager import ServiceManager
from serving.base import LLMRequest
from serving.config import get_config


async def main():
    # Load configuration
    config = get_config()
    
    # Initialize service manager
    manager = ServiceManager(config)
    
    # Create a request
    request = LLMRequest(
        prompt="What is the capital of France?",
        model="Llama-3.3-8B-Instruct",
        max_tokens=50,
        temperature=0.7
    )
    
    # Generate using llama provider directly
    response = await manager.generate(request, provider="llama")
    print(f"Provider: {response.provider}")
    print(f"Response: {response.text}")


if __name__ == "__main__":
    asyncio.run(main())