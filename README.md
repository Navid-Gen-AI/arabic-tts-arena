# 🎙️ Arabic TTS Arena

A community-driven platform to evaluate Arabic text-to-speech models through blind A/B voting. Users listen to two anonymous TTS outputs side by side, vote for the better one, and an ELO ranking system builds a community leaderboard.

**Frontend** → [Hugging Face Space](https://huggingface.co/spaces/Navid-AI/Arabic-TTS-Arena) (Gradio)
**Backend** → This repo, deployed on [Modal](https://modal.com) (GPU inference + vote storage)

## Architecture

```
User → Gradio (HF Space) → Modal Backend (GPU inference per model)
                          → ArenaService (vote recording → Modal Volume)
                          ← leaderboard.json ← Daily cron (Modal → HF Space)
```

## Current Models

| Model | GPU | Source |
|-------|-----|--------|
| XTTS-v2 | T4 | [coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2) |
| Chatterbox | T4 | [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox) |
| Fish Speech S1-mini | T4 | [fishaudio/s1-mini](https://huggingface.co/fishaudio/s1-mini) |
| MOSS-TTS | A10G | [OpenMOSS-Team/MOSS-TTS](https://huggingface.co/OpenMOSS-Team/MOSS-TTS) |

## Setup

### Prerequisites

- Python 3.10+
- A [Modal](https://modal.com) account
- A [Hugging Face](https://huggingface.co) token (for leaderboard cron + model downloads)

### Deploy

```bash
pip install modal
modal setup          # authenticate
modal deploy app.py
```

### Secrets (Modal Dashboard)

| Secret Name | Keys |
|-------------|------|
| `huggingface` | `HF_TOKEN` |

## Contributing a New Model

We welcome new Arabic TTS models! Here's how:

### 1. Create a model file

#### Open-source GPU model

Add `models/your_model.py`:

```python
import modal
from models import BaseTTSModel, register_model
from app import app, base_gpu_image

# Extend the shared GPU base with your model's dependencies
your_model_image = base_gpu_image.uv_pip_install("your-model-package")

@register_model
@app.cls(image=your_model_image, gpu="T4", scaledown_window=300,
         secrets=[modal.Secret.from_name("huggingface")])
class YourModel(BaseTTSModel):
    model_id = "your_model"          # unique, lowercase, underscores
    display_name = "Your Model"      # shown on leaderboard

    @modal.enter()
    def load_model(self):
        # Load weights once when the container starts
        ...

    @modal.method()
    def synthesize(self, text: str) -> dict:
        try:
            wav = ...  # numpy array
            sample_rate = 24000
            audio_base64 = self.audio_to_base64(wav, sample_rate)
            return self.success_response(audio_base64, sample_rate)
        except Exception as e:
            return self.error_response(e)
```

#### Closed-source / API-based model

See [`models/example_api_model.py`](models/example_api_model.py) for a full template.
Use `base_api_image` instead (lightweight, no GPU).
After your PR is merged, DM the maintainer your API keys — they run one command to store them securely in Modal.

### 2. Requirements

- ✅ Supports Arabic text input
- ✅ Returns WAV audio via `success_response()`
- ✅ Handles errors via `error_response()`
- ✅ Unique `model_id`
- ⚡ Inference under 30 seconds

### 3. Per-model images

Each model defines its own image by extending the appropriate base:

```python
# Open-source model → extend base_gpu_image
from app import base_gpu_image
your_image = base_gpu_image.uv_pip_install("your-package")

# API-based model → extend base_api_image (no GPU)
from app import base_api_image
your_image = base_api_image.pip_install("your-sdk")
```

### 4. Submit a PR

Push to a fork and open a Pull Request. The CI will automatically deploy to Modal on merge.

## License

Apache 2.0
