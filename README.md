# 🎙️ Arabic TTS Arena

Blind A/B voting platform for Arabic text-to-speech models. Users hear two anonymous outputs, pick the better one, and an ELO system ranks all models on a public [leaderboard](https://huggingface.co/spaces/Navid-AI/Arabic-TTS-Arena).

**We welcome all Arabic TTS models — open-source or proprietary.**

---

## Add Your Model

### Open-source model

1. Create `models/your_model.py`:

```python
import modal
from models import BaseTTSModel, register_model
from app import app, base_gpu_image

your_image = base_gpu_image.uv_pip_install("your-package")

@register_model
@app.cls(image=your_image, gpu="T4", scaledown_window=300,
         secrets=[modal.Secret.from_name("huggingface")])
class YourModel(BaseTTSModel):
    model_id = "your_model"
    display_name = "Your Model"

    @modal.enter()
    def load_model(self):
        ...  # load weights once on container start

    @modal.method()
    def synthesize(self, text: str) -> dict:
        try:
            wav = ...  # numpy array
            return self.success_response(self.audio_to_base64(wav, 24000), 24000)
        except Exception as e:
            return self.error_response(e)
```

2. Open a PR.

### API-based / closed-source model

1. Copy [`models/example_api_model.py`](models/example_api_model.py) and adapt it. Uses `base_api_image` — lightweight, no GPU.
2. Open a PR with your model code.
3. DM the maintainer your API keys. We store them in an encrypted vault (never in git).

### Rules

- Must support Arabic text input
- Return audio via `success_response()` / errors via `error_response()`
- Unique `model_id` (lowercase, underscores)
- Inference under 30 seconds

CI auto-deploys to Modal on merge.

## License

Apache 2.0
