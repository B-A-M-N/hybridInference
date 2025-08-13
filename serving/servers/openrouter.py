import os
import json
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request  # type: ignore
from fastapi.responses import JSONResponse  # type: ignore
import aiohttp


def _get_upstream_bases() -> List[str]:
    """Comma-separated list of upstream OpenAI-compatible API base URLs.

    Example: "http://localhost:8001/v1,http://10.0.0.5:8001/v1"
    """
    bases = os.getenv("UPSTREAM_API_BASES", "http://localhost:8001/v1")
    return [b.strip().rstrip("/") for b in bases.split(",") if b.strip()]


def _get_models_path() -> str:
    return os.getenv(
        "OPENROUTER_MODELS_PATH",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "openrouter_models.json"),
    )


app = FastAPI(title="HybridInference OpenRouter Provider")


@app.get("/openrouter/models")
async def list_models():
    """OpenRouter List Models endpoint.

    Returns a payload with key "data" containing model metadata per OpenRouter schema.
    """
    path = _get_models_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"Models file not found: {path}")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid models JSON: {e}")

    if not isinstance(data, dict) or "data" not in data:
        raise HTTPException(status_code=500, detail="Models JSON must contain top-level 'data' array")

    return JSONResponse(content=data)


async def _proxy_post_json(url: str, payload: dict, headers: Optional[dict] = None) -> JSONResponse:
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            text = await resp.text()
            # Try parse JSON; if fails, forward as text
            try:
                body = json.loads(text)
                return JSONResponse(content=body, status_code=resp.status)
            except json.JSONDecodeError:
                return JSONResponse(content=text, status_code=resp.status, media_type="application/json")


async def _forward_with_failover(path_suffix: str, body: dict) -> JSONResponse:
    bases = _get_upstream_bases()
    if not bases:
        raise HTTPException(status_code=500, detail="No upstream API bases configured")

    last_exc: Optional[Exception] = None
    # Minimal failover: try each base in order
    for base in bases:
        url = f"{base}{path_suffix}"
        try:
            return await _proxy_post_json(url, body, headers={"Content-Type": "application/json"})
        except aiohttp.ClientError as e:
            last_exc = e
            continue

    # All attempts failed
    raise HTTPException(status_code=502, detail=f"Upstream failure: {last_exc}")


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    """OpenAI-compatible Chat Completions endpoint.

    Body is forwarded to an upstream OpenAI-compatible server (e.g., vLLM/sglang).
    """
    try:
        body = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")

    # Optional: validate required fields minimally
    if "model" not in body:
        raise HTTPException(status_code=400, detail="Field 'model' is required")

    return await _forward_with_failover(path_suffix="/chat/completions", body=body)


@app.post("/v1/completions")
async def completions(req: Request):
    """OpenAI-compatible Completions endpoint.

    Body is forwarded to an upstream OpenAI-compatible server (e.g., vLLM/sglang).
    """
    try:
        body = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")

    # Optional: validate required fields minimally
    if "model" not in body:
        raise HTTPException(status_code=400, detail="Field 'model' is required")

    return await _forward_with_failover(path_suffix="/completions", body=body)


if __name__ == "__main__":
    import uvicorn  # type: ignore

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("serving.servers.openrouter:app", host=host, port=port, reload=False)

