# OpenRouter Integration

## Overview

HybridInference can act as an OpenRouter-compatible API provider, allowing you to serve your local models through the OpenRouter API format.

## Components

### 1. OpenRouter Server (`serving/servers/openrouter.py`)

A FastAPI server that:
- Provides OpenRouter-compatible endpoints
- Forwards requests to your backend servers (vLLM, SGLang, etc.)
- Supports basic failover between multiple backends

**Endpoints:**
- `GET /openrouter/models` - Returns available models list
- `POST /v1/chat/completions` - Chat completions (OpenAI format)
- `POST /v1/completions` - Text completions (OpenAI format)

### 2. OpenRouter Client (`serving/providers/openrouter.py`)

A client that can connect to:
- The official OpenRouter API (openrouter.ai)
- Your local OpenRouter server

**Features:**
- Generate completions via `generate()` method
- List available models via `list_models()` method

### 3. Models Configuration (`config/openrouter_models.json`)

Defines which models your server advertises as available:

```json
{
  "data": [
    {
      "id": "meta-llama/llama-3.1-8b-instruct",
      "name": "Meta Llama 3.1 8B Instruct",
      "context_length": 8192,
      "pricing": {
        "prompt": "0",
        "completion": "0"
      },
      // ... other OpenRouter-standard fields
    }
  ]
}
```

## Quick Start

### Running the Server

```bash
# Set your backend server URLs (comma-separated for multiple)
export UPSTREAM_API_BASES="http://localhost:8001/v1"

# Start the server
python -m serving.servers.openrouter

# Server runs on http://localhost:8080
```

### Using the Client

```python
from serving.providers.openrouter import OpenRouterProvider
from serving.base import LLMRequest

# Connect to your local server
provider = OpenRouterProvider({
    "base_url": "http://localhost:8080/v1"
})

# List models
models = await provider.list_models()

# Generate text
request = LLMRequest(
    prompt="Hello!",
    model="meta-llama/llama-3.1-8b-instruct",
    max_tokens=100
)
response = await provider.generate(request)
print(response.text)
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `UPSTREAM_API_BASES` | Backend server URLs (comma-separated) | `http://localhost:8001/v1` |
| `OPENROUTER_MODELS_PATH` | Path to models.json | `config/openrouter_models.json` |
| `PORT` | Server port | `8080` |

### Configuring Models

Edit `config/openrouter_models.json` to match your deployed models. Key fields:

- `id`: Model identifier (must match your backend)
- `context_length`: Maximum context size
- `pricing`: Set to "0" for internal use
- `quantization`: Your model's quantization (bf16, int8, etc.)

## Testing

```bash
# Test the models endpoint
curl http://localhost:8080/openrouter/models

# Test chat completion
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```