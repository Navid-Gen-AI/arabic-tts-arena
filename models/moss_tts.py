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
    )
    .add_local_python_source(*LOCAL_MODULES)
)


@register_model
@app.cls(
    image=moss_tts_image,
    gpu="A10G",
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
        import inspect
        import torch
        from transformers import AutoModel, AutoProcessor

        torch.backends.cuda.enable_cudnn_sdp(False)

        model_name = "OpenMOSS-Team/MOSS-TTS"

        # Load processor and model
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
        for attr in ('sampling_rate', 'sample_rate', 'audio_sample_rate'):
            if hasattr(self.model.config, attr):
                sr = getattr(self.model.config, attr)
                break
        self.sample_rate = sr

        # ── Runtime introspection ──
        # Read the actual source code of the model and processor classes
        self._model_source = ""
        self._proc_source = ""
        try:
            src_file = inspect.getfile(type(self.model))
            with open(src_file) as f:
                self._model_source = f.read()
            print(f"   Model source: {src_file} ({len(self._model_source)} chars)")
        except Exception as e:
            print(f"   Could not read model source: {e}")
        try:
            src_file = inspect.getfile(type(self.processor))
            with open(src_file) as f:
                self._proc_source = f.read()
            print(f"   Processor source: {src_file} ({len(self._proc_source)} chars)")
        except Exception as e:
            print(f"   Could not read processor source: {e}")

        # Dump model children (sub-modules)
        children = [(n, type(c).__name__) for n, c in self.model.named_children()]
        print(f"   Model children: {children}")

        # Audio-related methods
        for obj_name, obj in [("model", self.model), ("processor", self.processor)]:
            audio_methods = [m for m in dir(obj)
                             if not m.startswith('_')
                             and callable(getattr(obj, m, None))
                             and any(kw in m.lower() for kw in
                                     ['audio', 'wav', 'decode', 'synth', 'vocod',
                                      'codec', 'codes', 'generate', 'speech'])]
            print(f"   {obj_name} audio-related methods: {audio_methods}")

        print(f"✅ MOSS-TTS loaded (sr={self.sample_rate})")
        print(f"   model type:     {type(self.model).__name__}")
        print(f"   processor type: {type(self.processor).__name__}")
        print(f"   model module:   {type(self.model).__module__}")

    @modal.method()
    def synthesize(self, text: str) -> dict:
        """Synthesize Arabic text to speech."""
        try:
            import torch
            import numpy as np

            model_src = self._model_source
            proc_src = self._proc_source

            # ─────────────────────────────────────────────────
            # Strategy 1: Direct synthesis methods on model
            # ─────────────────────────────────────────────────
            for method_name in ['synthesize', 'tts', 'text_to_speech',
                                'inference', 'infer', 'predict']:
                method = getattr(self.model, method_name, None)
                if method is not None and callable(method):
                    try:
                        result = method(text)
                        wav = self._extract_audio(result)
                        if wav is not None:
                            print(f"  ✅ model.{method_name}() worked")
                            audio_base64 = self.audio_to_base64(wav, self.sample_rate)
                            return self.success_response(audio_base64, self.sample_rate)
                    except Exception as e:
                        print(f"  ✗ model.{method_name}(): {e}")

            # ─────────────────────────────────────────────────
            # Strategy 2: Processor-based synthesis
            # ─────────────────────────────────────────────────
            for method_name in ['synthesize', 'tts', 'text_to_speech',
                                'text_to_audio', '__call__']:
                method = getattr(self.processor, method_name, None)
                if method is not None and callable(method):
                    try:
                        if method_name == '__call__':
                            result = self.processor(text, return_tensors="pt")
                        else:
                            result = method(text)
                        wav = self._extract_audio(result)
                        if wav is not None:
                            print(f"  ✅ processor.{method_name}() worked")
                            audio_base64 = self.audio_to_base64(wav, self.sample_rate)
                            return self.success_response(audio_base64, self.sample_rate)
                    except Exception as e:
                        print(f"  ✗ processor.{method_name}(): {e}")

            # ─────────────────────────────────────────────────
            # Strategy 3: generate() with processor
            # ─────────────────────────────────────────────────
            # Try conversation-style input first, then plain text
            input_formats = [
                ("conversations", {"conversations": [{"role": "user", "content": text}]}),
                ("text", {"text": text}),
            ]

            for fmt_name, fmt_kwargs in input_formats:
                try:
                    inputs = self.processor(
                        **fmt_kwargs,
                        return_tensors="pt",
                    ).to("cuda")

                    with torch.no_grad():
                        outputs = self.model.generate(**inputs, max_new_tokens=4096)

                    desc = self._describe(outputs)
                    print(f"  generate({fmt_name}) → {desc}")

                    # Try to decode the output into audio
                    wav = self._decode_generate_output(outputs, inputs)
                    if wav is not None:
                        print(f"  ✅ generate({fmt_name}) decoded successfully")
                        audio_base64 = self.audio_to_base64(wav, self.sample_rate)
                        return self.success_response(audio_base64, self.sample_rate)

                except Exception as e:
                    print(f"  ✗ generate({fmt_name}): {e}")

            # ─────────────────────────────────────────────────
            # All strategies failed — return full diagnostic
            # ─────────────────────────────────────────────────
            proc_methods = sorted([m for m in dir(self.processor)
                                   if not m.startswith('_')
                                   and callable(getattr(self.processor, m, None))])
            model_methods = sorted([m for m in dir(self.model)
                                    if not m.startswith('_')
                                    and callable(getattr(self.model, m, None))])
            model_children = [(n, type(c).__name__)
                              for n, c in self.model.named_children()]

            # Dump the first 3000 chars of model source for debugging
            src_snippet = model_src[:3000] if model_src else "N/A"
            proc_snippet = proc_src[:2000] if proc_src else "N/A"

            return self.error_response(
                f"All synthesis strategies failed.\n"
                f"Model type: {type(self.model).__name__}\n"
                f"Processor type: {type(self.processor).__name__}\n"
                f"Processor methods: {proc_methods}\n"
                f"Model methods: {model_methods}\n"
                f"Model children: {model_children}\n"
                f"Config sampling_rate keys: "
                f"{[k for k in dir(self.model.config) if 'sampl' in k.lower() or 'rate' in k.lower() or 'sr' in k.lower()]}\n"
                f"Model source (first 3000):\n{src_snippet}\n"
                f"Processor source (first 2000):\n{proc_snippet}"
            )
        except Exception as e:
            return self.error_response(e)

    def _describe(self, obj, depth=0):
        """Recursively describe a nested object for diagnostics."""
        import torch
        import numpy as np
        t = type(obj).__name__
        if isinstance(obj, torch.Tensor):
            return f"Tensor(shape={list(obj.shape)}, dtype={obj.dtype})"
        if isinstance(obj, np.ndarray):
            return f"ndarray(shape={list(obj.shape)}, dtype={obj.dtype})"
        if isinstance(obj, (list, tuple)):
            items = [self._describe(x, depth+1) for x in obj[:4]]
            return f"{t}(len={len(obj)}, [{', '.join(items)}])"
        if isinstance(obj, dict):
            keys = list(obj.keys())[:10]
            return f"dict(keys={keys})"
        if hasattr(obj, '__dict__'):
            keys = [k for k in obj.__dict__ if not k.startswith('_')][:10]
            return f"{t}(attrs={keys})"
        return f"{t}({str(obj)[:80]})"

    def _extract_audio(self, result):
        """Try to extract a numpy audio waveform from various result types."""
        import torch
        import numpy as np

        if result is None:
            return None

        # dict with 'audio' key
        if isinstance(result, dict):
            for key in ('audio', 'waveform', 'wav', 'speech'):
                if key in result:
                    return self._to_numpy(result[key])

        # tensor directly
        if isinstance(result, torch.Tensor) and result.dim() >= 1:
            # Only if it looks like audio (float, many samples)
            if result.dtype in (torch.float32, torch.float16, torch.bfloat16):
                return self._to_numpy(result)

        # numpy array
        if isinstance(result, np.ndarray) and result.dtype in (np.float32, np.float64):
            return result.squeeze().astype(np.float32)

        # Named tuple or object with audio attribute
        for attr in ('audio', 'waveform', 'wav', 'speech', 'audios'):
            if hasattr(result, attr):
                v = getattr(result, attr)
                arr = self._to_numpy(v)
                if arr is not None:
                    return arr

        return None

    def _to_numpy(self, v):
        """Convert tensor/ndarray/list to float32 numpy."""
        import torch
        import numpy as np

        if v is None:
            return None
        if isinstance(v, torch.Tensor):
            if v.numel() < 100:  # too small to be audio
                return None
            return v.squeeze().cpu().float().numpy()
        if isinstance(v, np.ndarray):
            if v.size < 100:
                return None
            return v.squeeze().astype(np.float32)
        if isinstance(v, list):
            arr = np.array(v, dtype=np.float32).squeeze()
            if arr.size < 100:
                return None
            return arr
        return None

    def _decode_generate_output(self, outputs, inputs):
        """Try every possible way to decode generate() output into audio."""
        import torch
        import numpy as np

        # 1. If outputs is already a tensor (standard generate output)
        if isinstance(outputs, torch.Tensor):
            # Remove input tokens to get only generated tokens
            if hasattr(inputs, 'input_ids'):
                input_len = inputs.input_ids.shape[-1]
                generated = outputs[:, input_len:]
            else:
                generated = outputs
            return self._try_decode_tokens(generated)

        # 2. If outputs is a list of tuples (observed structure)
        if isinstance(outputs, (list, tuple)):
            all_tensors = []
            self._collect_tensors(outputs, all_tensors, "out")

            for path, tensor in all_tensors:
                print(f"    tensor {path}: shape={list(tensor.shape)} dtype={tensor.dtype}")
                # Float tensor with many elements → probably audio
                if tensor.dtype in (torch.float32, torch.float16, torch.bfloat16) and tensor.numel() >= 100:
                    return tensor.squeeze().cpu().float().numpy()

            # No float tensors — try decode on int64 tensors
            for path, tensor in all_tensors:
                if tensor.dtype in (torch.int64, torch.int32, torch.long) and tensor.dim() >= 1:
                    wav = self._try_decode_tokens(tensor)
                    if wav is not None:
                        return wav

        # 3. Object with attributes
        if hasattr(outputs, '__dict__'):
            for attr_name in ('sequences', 'audio_values', 'audio', 'waveform',
                              'generated_ids', 'logits'):
                v = getattr(outputs, attr_name, None)
                if v is not None:
                    print(f"    outputs.{attr_name}: {self._describe(v)}")
                    wav = self._extract_audio(v)
                    if wav is not None:
                        return wav

        return None

    def _collect_tensors(self, obj, result, path):
        """Recursively collect all tensors from a nested structure."""
        import torch
        if isinstance(obj, torch.Tensor):
            result.append((path, obj))
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                self._collect_tensors(item, result, f"{path}[{i}]")
        elif hasattr(obj, '__dict__'):
            for k, v in obj.__dict__.items():
                if not k.startswith('_'):
                    self._collect_tensors(v, result, f"{path}.{k}")

    def _try_decode_tokens(self, token_ids):
        """Try decoding integer token IDs into audio waveform."""
        import torch
        import numpy as np

        attempts = []

        # a. processor.decode → might return text with embedded audio
        if hasattr(self.processor, 'decode'):
            try:
                if token_ids.dim() == 1:
                    decoded = self.processor.decode(token_ids, skip_special_tokens=True)
                else:
                    decoded = self.processor.batch_decode(token_ids, skip_special_tokens=True)
                print(f"    processor.decode → {type(decoded).__name__}: {str(decoded)[:200]}")
                # If decode returns audio data (unlikely but check)
                wav = self._extract_audio(decoded)
                if wav is not None:
                    return wav
            except Exception as e:
                attempts.append(f"processor.decode: {e}")

        # b. Check if model has a codec / vocoder sub-module
        for sub_name in ['codec', 'vocoder', 'audio_encoder', 'audio_decoder',
                         'speech_decoder', 'codec_model', 'audio_codec']:
            sub = getattr(self.model, sub_name, None)
            if sub is not None:
                print(f"    Found sub-module: model.{sub_name} = {type(sub).__name__}")
                # Try decode
                for method in ['decode', 'decode_code', '__call__']:
                    fn = getattr(sub, method, None)
                    if fn is not None and callable(fn):
                        for fmt in [token_ids.unsqueeze(0), token_ids]:
                            try:
                                result = fn(fmt)
                                wav = self._extract_audio(result)
                                if wav is not None:
                                    print(f"    ✅ model.{sub_name}.{method}() → audio")
                                    return wav
                            except Exception as e:
                                attempts.append(f"{sub_name}.{method}: {e}")

        # c. model.generate returns codes → processor decodes them
        for method_name in ['codes_to_wav', 'decode_audio', 'codes_to_audio',
                            'convert_tokens_to_audio', 'extract_audio']:
            method = getattr(self.processor, method_name, None)
            if method is not None and callable(method):
                for fmt in [token_ids, token_ids.unsqueeze(0),
                            token_ids.cpu().numpy()]:
                    try:
                        result = method(fmt)
                        wav = self._extract_audio(result)
                        if wav is not None:
                            print(f"    ✅ processor.{method_name}() → audio")
                            return wav
                    except Exception as e:
                        attempts.append(f"processor.{method_name}: {e}")

        if attempts:
            print(f"    decode attempts: {attempts}")
        return None
