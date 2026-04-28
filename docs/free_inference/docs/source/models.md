# Available Models

FreeInference provides access to multiple state-of-the-art LLM models for coding agents and IDEs.

## Model Overview

| Model ID | Name | Context Length | Max Output | Features |
|----------|------|----------------|------------|----------|
| `glm-5.1` | GLM-5.1 | 200K tokens | 128K tokens | Function calling, Structured output, Bilingual (Chinese/English), Thinking mode |
| `glm-5` | GLM-5 | 200K tokens | 128K tokens | Function calling, Structured output, Bilingual (Chinese/English), Thinking mode |
| `glm-4.7` | GLM-4.7 | 200K tokens | 128K tokens | Function calling, Structured output, Bilingual (Chinese/English), Thinking mode |
| `glm-4.7-flash` | GLM-4.7-Flash | 200K tokens | 128K tokens | Function calling, Structured output, Bilingual (Chinese/English), Thinking mode |
| `minimax-m2.5` | MiniMax M2.5 | 1M tokens | 128K tokens | Function calling, Structured output, Thinking mode, Multimodal (text+image) |
| `minimax-m2` | MiniMax M2 | 196K tokens | 8K tokens | Function calling, Structured output |
| `qwen3-coder-30b` | Qwen3 Coder 30B | 32K tokens | 8K tokens | Function calling, Structured output |
| `llama-3.3-70b-instruct` | Llama 3.3 70B Instruct | 131K tokens | 8K tokens | Function calling, Structured output |
| `llama-4-scout` | Llama 4 Scout | 128K tokens | 16K tokens | Function calling, Structured output |
| `llama-4-maverick` | Llama 4 Maverick | 128K tokens | 16K tokens | Function calling, Structured output, Multimodal (text+image) |

> **Note:** Llama models are available with limited capacity. Availability may vary during peak usage.

### Embedding Models

| Model ID | Name | Dimensions | Context Length | Use Case |
|----------|------|------------|----------------|----------|
| `bge-m3` | BGE-M3 | 1024 | 8K tokens | Codebase indexing, semantic search |

---

## Model Details

### GLM-5.1

**Model ID:** `glm-5.1`

- Context length: 200,000 tokens
- Max output: 128,000 tokens
- Quantization: fp8
- Input modalities: text
- Output modalities: text
- Language support: Chinese, English
- Function calling: Yes
- Structured output: Yes
- Thinking mode: Yes
- Tool streaming: Yes

---

### GLM-5

**Model ID:** `glm-5`

- Context length: 200,000 tokens
- Max output: 128,000 tokens
- Architecture: 745B MoE (44B active parameters)
- Quantization: fp8
- Input modalities: text
- Output modalities: text
- Language support: Chinese, English
- Function calling: Yes
- Structured output: Yes
- Thinking mode: Yes
- Tool streaming: Yes

---

### GLM-4.7

**Model ID:** `glm-4.7`

- Context length: 200,000 tokens
- Max output: 128,000 tokens
- Quantization: fp8
- Input modalities: text
- Output modalities: text
- Language support: Chinese, English
- Function calling: Yes
- Structured output: Yes
- Thinking mode: Yes
- Tool streaming: Yes

---

### GLM-4.7-Flash

**Model ID:** `glm-4.7-flash`

- Context length: 200,000 tokens
- Max output: 128,000 tokens
- Quantization: fp8
- Input modalities: text
- Output modalities: text
- Language support: Chinese, English
- Function calling: Yes
- Structured output: Yes
- Thinking mode: Yes
- Tool streaming: Yes

---

### MiniMax M2.5

**Model ID:** `minimax-m2.5`

- Context length: 1,000,000 tokens
- Max output: 131,072 tokens
- Architecture: 230B MoE (10B active parameters)
- Quantization: bf16
- Input modalities: text, image
- Output modalities: text
- Function calling: Yes
- Structured output: Yes
- Thinking mode: Yes

---

### MiniMax M2

**Model ID:** `minimax-m2`

- Context length: 196,608 tokens
- Max output: 8,192 tokens
- Quantization: bf16
- Input modalities: text
- Output modalities: text
- Function calling: Yes
- Structured output: Yes

---

### Qwen3 Coder 30B

**Model ID:** `qwen3-coder-30b`

- Context length: 32,768 tokens
- Max output: 8,192 tokens
- Quantization: bf16
- Input modalities: text
- Output modalities: text
- Function calling: Yes
- Structured output: Yes

---

### Llama 3.3 70B Instruct (Limited Capacity)

**Model ID:** `llama-3.3-70b-instruct`

- Context length: 131,072 tokens
- Max output: 8,192 tokens
- Quantization: bf16
- Input modalities: text
- Output modalities: text
- Function calling: Yes
- Structured output: Yes

---

### Llama 4 Scout (Limited Capacity)

**Model ID:** `llama-4-scout`

- Context length: 128,000 tokens
- Max output: 16,384 tokens
- Quantization: fp8
- Input modalities: text
- Output modalities: text
- Function calling: Yes
- Structured output: Yes

---

### Llama 4 Maverick (Limited Capacity)

**Model ID:** `llama-4-maverick`

- Context length: 128,000 tokens
- Max output: 16,384 tokens
- Quantization: fp8
- Input modalities: text, image
- Output modalities: text
- Function calling: Yes
- Structured output: Yes

---

### BGE-M3 (Embedding)

**Model ID:** `bge-m3`

- Type: Embedding
- Dimensions: 1024
- Context length: 8,192 tokens
- Quantization: fp16
- Input modalities: text
- Output modalities: embedding
- Multilingual: Yes (100+ languages)

Use this model for codebase indexing in Roo Code, Kilo Code, and other tools that support semantic code search. See the [integration guide](integrations.md) for setup instructions.

---

## Switching Models

To use different models, change the model name in your IDE configuration:

**Cursor:** Select from the dropdown in settings

**Codex:** Edit `~/.codex/config.toml`:
```toml
model = "glm-5"  # Change to any model ID
```

**Roo Code / Kilo Code:** Select from the dropdown in extension settings
