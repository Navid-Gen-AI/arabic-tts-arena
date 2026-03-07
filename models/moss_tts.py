import modal
from models import BaseTTSModel, register_model
from app import app, LOCAL_MODULES

moss_tts_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("ffmpeg", "libsndfile1", "espeak-ng", "git")
    .uv_pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy",
        "soundfile",
        "transformers>=4.36.0",
        "accelerate",
    )
    .run_commands(
        "pip install --upgrade pip && "
        "git clone https://github.com/OpenMOSS/MOSS-TTS.git /tmp/moss-tts && "
        "cd /tmp/moss-tts && pip install . && "
        "rm -rf /tmp/moss-tts",
        # Introspect the installed package so we can find the right API
        "python3 -c \""
        "import moss_tts; print('moss_tts dir:', [x for x in dir(moss_tts) if not x.startswith('_')]); "
        "import inspect, moss_tts; "
        "[print(f'{name}') for name, obj in inspect.getmembers(moss_tts) if inspect.isclass(obj) or inspect.isfunction(obj)]"
        "\" || true",
        "python3 -c \""
        "import importlib, pkgutil; "
        "import moss_tts; "
        "for info in pkgutil.walk_packages(moss_tts.__path__, moss_tts.__name__ + '.'): "
        "    print(info.name)"
        "\" || true",
    )
    .add_local_python_source(*LOCAL_MODULES)
)


@register_model
@app.cls(
    image=moss_tts_image,
    gpu="A10G",  # 8B model needs more VRAM than a T4
    scaledown_window=300,
    retries=0,
    secrets=[modal.Secret.from_name("hf-ar-tts-arena")],
)
class MossTTSModel(BaseTTSModel):
    """OpenMOSS MOSS-TTS — 8B parameter multilingual TTS with Arabic support."""

    model_id = "moss_tts"
    display_name = "MOSS-TTS"
    model_url = "https://huggingface.co/OpenMOSS-Team/MOSS-TTS"

    @modal.enter()
    def load_model(self):
        """Load the MOSS-TTS model when container starts."""
        import torch
        from transformers import AutoModel, AutoProcessor, pipeline

        # Required for MOSS-TTS inference
        torch.backends.cuda.enable_cudnn_sdp(False)

        model_name = "OpenMOSS-Team/MOSS-TTS"

        # Try the HF pipeline first — it handles the full generate→decode flow
        try:
            self.pipe = pipeline(
                "text-to-audio",
                model=model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device="cuda",
            )
            self.use_pipeline = True
            self.sample_rate = getattr(
                self.pipe.model.config, 'sampling_rate', 24000
            )
            print(f"✅ MOSS-TTS loaded via pipeline (sr={self.sample_rate})")
        except Exception as e:
            print(f"  ⚠️ Pipeline failed ({e}), falling back to manual loading")
            self.use_pipeline = False

            self.processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=True,
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            ).to("cuda")

            sr = 24000
            if hasattr(self.model.config, 'sampling_rate'):
                sr = self.model.config.sampling_rate
            self.sample_rate = sr
            print(f"✅ MOSS-TTS loaded manually (sr={self.sample_rate})")

    @modal.method()
    def synthesize(self, text: str) -> dict:
        """Synthesize Arabic text to speech."""
        try:
            import torch
            import numpy as np

            # ── Path A: HF pipeline (handles everything) ──
            if self.use_pipeline:
                result = self.pipe(text)
                # Pipeline returns {"audio": ndarray, "sampling_rate": int}
                wav = result["audio"]
                sr = result.get("sampling_rate", self.sample_rate)
                if isinstance(wav, torch.Tensor):
                    wav = wav.squeeze().cpu().float().numpy()
                wav = np.asarray(wav, dtype=np.float32).squeeze()
                audio_base64 = self.audio_to_base64(wav, sr)
                return self.success_response(audio_base64, sr)

            # ── Path B: Manual model.generate() ──
            conversations = [{"role": "user", "content": text}]
            inputs = self.processor(
                conversations=conversations,
                return_tensors="pt",
            ).to("cuda")

            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=4096)

            # Deep-inspect the output structure for diagnostics
            def describe(obj, depth=0):
                t = type(obj).__name__
                if isinstance(obj, torch.Tensor):
                    return f"Tensor(shape={list(obj.shape)}, dtype={obj.dtype})"
                if isinstance(obj, np.ndarray):
                    return f"ndarray(shape={list(obj.shape)}, dtype={obj.dtype})"
                if isinstance(obj, (list, tuple)):
                    items = [describe(x, depth+1) for x in obj[:4]]
                    return f"{t}(len={len(obj)}, [{', '.join(items)}])"
                if hasattr(obj, '__dict__'):
                    keys = [k for k in obj.__dict__ if not k.startswith('_')]
                    return f"{t}(attrs={keys})"
                return f"{t}({str(obj)[:80]})"

            desc = describe(outputs)
            print(f"  generate() → {desc}")

            # Walk the nested structure to find tensors
            def find_tensors(obj, path=""):
                found = []
                if isinstance(obj, torch.Tensor):
                    found.append((path, obj))
                elif isinstance(obj, (list, tuple)):
                    for i, item in enumerate(obj):
                        found.extend(find_tensors(item, f"{path}[{i}]"))
                elif hasattr(obj, '__dict__'):
                    for k, v in obj.__dict__.items():
                        if not k.startswith('_'):
                            found.extend(find_tensors(v, f"{path}.{k}"))
                return found

            tensors = find_tensors(outputs, "outputs")
            for path, t in tensors:
                print(f"  {path}: shape={list(t.shape)} dtype={t.dtype}")

            wav = None

            # Extract the audio codes tensor: outputs[0][1] is shape [42, 33]
            # (42 codebooks × 33 timesteps)
            audio_codes = outputs[0][1] if isinstance(outputs[0], tuple) else None

            if audio_codes is not None and audio_codes.dim() >= 2:
                print(f"  audio_codes shape: {list(audio_codes.shape)}")

                # Try every available decode method
                decode_attempts = []

                # 1. processor.codes_to_wav
                if hasattr(self.processor, 'codes_to_wav'):
                    try:
                        wav = self.processor.codes_to_wav(audio_codes)
                        decode_attempts.append("processor.codes_to_wav ✓")
                    except Exception as e:
                        decode_attempts.append(f"processor.codes_to_wav ✗: {e}")

                # 2. processor.decode with audio codes
                if wav is None and hasattr(self.processor, 'decode'):
                    for fmt in [audio_codes, audio_codes.unsqueeze(0), [audio_codes]]:
                        try:
                            result = self.processor.decode(fmt)
                            if isinstance(result, torch.Tensor):
                                wav = result
                            elif hasattr(result, 'audio'):
                                wav = result.audio
                            elif isinstance(result, dict) and 'audio' in result:
                                wav = result['audio']
                            if wav is not None:
                                decode_attempts.append("processor.decode ✓")
                                break
                        except Exception as e:
                            decode_attempts.append(f"processor.decode ✗: {e}")

                # 3. model.decode / model.decoder
                if wav is None:
                    decoder = getattr(self.model, 'decode', None) or getattr(self.model, 'decoder', None)
                    if decoder is not None:
                        for fmt in [audio_codes.unsqueeze(0), audio_codes]:
                            try:
                                wav = decoder(fmt) if callable(decoder) else decoder.decode(fmt)
                                decode_attempts.append("model.decode ✓")
                                break
                            except Exception as e:
                                decode_attempts.append(f"model.decode ✗: {e}")

                # 4. model.codes_to_audio / model.decode_codes
                if wav is None:
                    for method_name in ['codes_to_audio', 'decode_codes', 'vocoder', 'audio_decoder']:
                        method = getattr(self.model, method_name, None)
                        if method is not None and callable(method):
                            try:
                                wav = method(audio_codes.unsqueeze(0))
                                decode_attempts.append(f"model.{method_name} ✓")
                                break
                            except Exception as e:
                                decode_attempts.append(f"model.{method_name} ✗: {e}")

                # 5. Log all model & processor public methods for debugging
                if wav is None:
                    proc_methods = [m for m in dir(self.processor) if not m.startswith('_') and callable(getattr(self.processor, m, None))]
                    model_methods = [m for m in dir(self.model) if not m.startswith('_') and callable(getattr(self.model, m, None))]
                    return self.error_response(
                        f"All decode attempts failed: {decode_attempts}. "
                        f"Processor methods: {proc_methods[:30]}. "
                        f"Model methods: {model_methods[:30]}"
                    )

            if wav is None:
                return self.error_response(f"No audio codes found. Structure: {desc}")

            if isinstance(wav, torch.Tensor):
                wav = wav.squeeze().cpu().float().numpy()
            elif not isinstance(wav, np.ndarray):
                wav = np.array(wav, dtype=np.float32)

            audio_base64 = self.audio_to_base64(wav, self.sample_rate)
            return self.success_response(audio_base64, self.sample_rate)
        except Exception as e:
            return self.error_response(e)
