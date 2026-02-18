# M²A: Multimodal Memory Agent with Dual-Layer Hybrid Memory for Long-Term Personalized Interactions - Official Implementation

This repository contains the official implementation for the paper: **M²A: Multimodal Memory Agent with Dual-Layer Hybrid Memory for Long-Term Personalized Interactions**.

## 1. Installation

### Quick Start

```bash
# Create virtual environment (using uvu
uv venv

# Activate environment
source .venv/bin/activate

# Install dependencies
pip install -e agent/
```

<!-- ### Requirements

*   **OS**: Ubuntu 22.04 (Tested)
*   **Python**: 3.9+
*   **LLM Backend**: An OpenAI-compatible inference server (e.g., vLLM or SGLang) or OpenAI API -->

## 2. Configuration

The system supports flexible configuration through environment variables or a JSON config file.

### Quick Configuration

For development/testing, you can set key environment variables:

```bash
# For local LLM server (vLLM)
export M2A_LLM_PROVIDER=custom
export M2A_LLM_BASE_URL=http://localhost:9000/v1
export M2A_LLM_API_KEY=EMPTY

# For OpenAI API
export M2A_LLM_PROVIDER=openai
export M2A_LLM_MODEL=gpt-4o-mini
export M2A_LLM_API_KEY=sk-...
export M2A_LLM_BASE_URL=https://api.openai.com/v1

# For image format
export M2A_IMAGE_USE_BASE64=false  # vLLM file:// format
# export M2A_IMAGE_USE_BASE64=true   # OpenAI base64 format

# For embedding services
export M2A_TEXT_EMBEDDING_PROVIDER=local
export M2A_TEXT_EMBEDDING_BASE_URL=http://localhost:8100/v1
export M2A_IMAGE_EMBEDDING_PROVIDER=local
export M2A_IMAGE_EMBEDDING_BASE_URL=http://localhost:8050/v1
```

### Using a Config File

For more complex configurations, use a JSON config file:

```bash
# Create config file
cat > m2a_config.json << EOF
{
  "llm": {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "api_key": "sk-...",
    "base_url": "https://api.openai.com/v1",
    "temperature": 0.7,
    "max_tokens": 1200,
    "timeout": 20
  },
  "embedding": {
    "text_provider": "local",
    "text_base_url": "http://localhost:8100/v1"
  },
  "image_format": {
    "use_base64": false
  }
}
EOF

# Load config file
python -c "
from agent.m2a import create_m2a_from_file
m2a = create_m2a_from_file('m2a_config.json')
"
```

### Configuration Reference

See `agent/config.py` for complete configuration schema.

Key configuration options:

| Category | Environment Variable | Default | Description |
|----------|-------------------|----------|-------------|
| LLM Provider | `M2A_LLM_PROVIDER` | `openai` | `openai` or `custom` |
| LLM Model | `M2A_LLM_MODEL` | `gpt-4o-mini` | Model name |
| LLM API Key | `M2A_LLM_API_KEY` | - | API key for LLM |
| LLM Base URL | `M2A_LLM_BASE_URL` | `https://api.openai.com/v1` | API endpoint |
| Text Embedding Provider | `M2A_TEXT_EMBEDDING_PROVIDER` | `local` | `local` or `openai` |
| Image Embedding Provider | `M2A_IMAGE_EMBEDDING_PROVIDER` | `local` | `local` or `openai` |
| Image Format | `M2A_IMAGE_USE_BASE64` | `false` | `true` for OpenAI, `false` for vLLM |
| Max Query Iterations | `M2A_CHAT_MAX_QUERY_ITER` | `5` | ChatAgent query iteration limit |
| Max Update Iterations | `M2A_CHAT_MAX_UPDATE_ITER` | `5` | ChatAgent update iteration limit |
| Context Window | `M2A_CHAT_CONTEXT_WINDOW` | `5` | Memory manager context size |
| Memory Max Iterations | `M2A_MEMORY_MAX_ITER` | `15` | MemoryManager iteration limit |

## 3. Data Preparation

### Download Datasets

Please download the LoCoMo dataset from [here](https://github.com/snap-research/LoCoMo) and the YoLLaVA dataset from [here](https://github.com/WisconsinAIVision/YoLLaVA).

For local evaluation, all images must be downloaded to your machine, and image URLs in the datasets should be replaced with local file paths.

### Preprocessing

To construct the multimodally enhanced LoCoMo dataset, please refer to `data/yollava_generate_and_merge.py`.

This script injects visual-centric QA pairs from YoLLaVA into LoCoMo following the pipeline described in the paper.

## 4. Chat with M²A

### Basic Usage

```python
from agent.m2a import M2ASystem, create_m2a_from_env

# Create with default config (from environment)
m2a = M2ASystem()

# Chat
response = m2a.chat_once(text="Hello")
print(response)

# With image
response = m2a.chat_once(text="What's in this image?", image="/path/to/image.jpg")
print(response)
```

### Advanced Configuration

```python
from agent.m2a import M2ASystem, M2AConfig

# Create with custom config
config = M2AConfig(
    llm={
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key": "sk-...",
        "base_url": "https://api.openai.com/v1",
    },
    embedding={
        "text_provider": "local",
        "text_base_url": "http://localhost:8100/v1",
    },
    image_format={
        "use_base64": False,  # Use vLLM file:// format
    },
)

m2a = M2ASystem(config=config)
```

## 5. Evaluation

The system includes an evaluation wrapper for running experiments on the enhanced LoCoMo dataset.

### Running Evaluation

```python
from eval_wrapper import create_evaluation_wrapper
from data.evaluator import Evaluator

# Create evaluation wrapper
wrapper = create_evaluation_wrapper(db_dir="eval_results")

# Initialize for each conversation
wrapper.start_conversation(conv_info={
    "conv_idx": 0,
    "speaker_0": "Alice",
    "speaker_1": "Bob"
})

# Process dialogue history (builds memory)
wrapper.chat(dialogue=[
    {"speaker": 0, "dia_id": "msg1", "images": [], "text": "Hello", "timestamp": "2024-01-01 10:00"},
    # ... more messages
])

# Answer a question
answer = wrapper.question(text="What did we discuss?")
print(answer)

# Save results after conversation
wrapper.over(results=results_dict, summary=summary_dict)
```

### Using with Evaluator

The `eval_wrapper.py` module provides a wrapper class that implements the interface expected by `data/evaluator.py`:

- `start_conversation(conv_info)` - Initialize for a new conversation
- `chat(dialogue)` - Process dialogue history (builds memory)
- `question(text, image)` - Answer a question about the conversation
- `over(**kwargs)` - Save evaluation results and dump databases

```python
from data.evaluator import Evaluator

ev = Evaluator(methods=[wrapper])
ev.evaluate_file(dataset_path)
```

This evaluation protocol follows the setup described in the paper.

## 6. File Structure

```
M2A_agent_submit/
├── agent/                          # M²A system implementation
│   ├── __init__.py
│   ├── config.py              # Configuration system
│   ├── m2a.py                # Main system entry point
│   ├── agents/                # LangGraph agents
│   │   ├── __init__.py
│   │   ├── chat_agent.py
│   │   └── memory_manager.py
│   ├── stores/                # Storage layers
│   │   ├── __init__.py
│   │   ├── raw.py
│   │   ├── semantic.py
│   │   └── image_manager.py
│   ├── embeddings/            # Embedding services
│   │   ├── __init__.py
│   └── local_embeddings.py
│   └── utils/                # Utilities
├── data/                             # Dataset code
│   ├── yollava_generate_and_merge.py  # Dataset preprocessing
│   └── evaluator.py                 # Evaluation pipeline
├── eval_wrapper.py                # Evaluation wrapper
├── pyproject.toml                  # Dependencies (in agent/)
└── CLAUDE.md                     # This file
```


## License

This project is licensed under the MIT License.