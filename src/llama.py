"""
Wraps the served LLM (Llama by default).

Used for:
  1) generating r samples per prompt to get median length labels
  2) pulling last-token hidden states for ProD-M
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.utils import get_hf_token


def _resolve_device(requested):
    """Use CUDA only if this torch build actually supports it."""
    requested = (requested or "cuda").lower()
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError(
        "CUDA was requested but this PyTorch build has no GPU support.\n"
        "Common Colab cause: `pip install torch` replaced Colab's CUDA torch "
        "with a CPU-only wheel.\n\n"
        "Fix in Colab:\n"
        "  1) Runtime → Change runtime type → GPU (T4)\n"
        "  2) Runtime → Restart session\n"
        "  3) Reinstall CUDA torch, then our deps WITHOUT torch:\n"
        "       !pip install -q --upgrade torch --index-url "
        "https://download.pytorch.org/whl/cu121\n"
        "       !pip install -q -r requirements.txt\n"
        "  4) Check: import torch; print(torch.cuda.is_available())  # True"
    )


class LlamaServer:
    def __init__(self, model_name, device="cuda", load_in_4bit=True, max_prompt_tokens=4096):
        self.model_name = model_name
        self.device = _resolve_device(device)
        self.max_prompt_tokens = max_prompt_tokens
        token = get_hf_token()

        print(f"Loading {model_name} (device={self.device}, 4bit={load_in_4bit})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = {"token": token}
        if load_in_4bit and self.device != "cpu":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            kwargs["device_map"] = "auto"
            kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = torch.float16 if self.device != "cpu" else torch.float32
            if self.device != "cpu":
                kwargs["device_map"] = self.device

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()
        self.hidden_dim = self.model.config.hidden_size

    def _format(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _input_device(self):
        # quantized / device_map models expose .device on the root module
        if hasattr(self.model, "device"):
            return self.model.device
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device(self.device if self.device != "cuda" else "cuda:0")

    def _to_device(self, batch):
        device = self._input_device()
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def generate_lengths(self, prompt, num_samples, max_new_tokens, temperature, top_p):
        """Run the model num_samples times and return output token counts."""
        text = self._format(prompt)
        inputs = self._to_device(self.tokenizer(text, return_tensors="pt"))
        prompt_len = inputs["input_ids"].shape[1]
        lengths = []

        for _ in range(num_samples):
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            lengths.append(int(out.shape[1] - prompt_len))
        return lengths

    @torch.no_grad()
    def encode(self, prompts, batch_size=4):
        """
        Last-layer hidden state at the last prompt token.
        This is what ProD-M uses as input features.
        """
        all_h = []
        for start in range(0, len(prompts), batch_size):
            batch_prompts = [self._format(p) for p in prompts[start : start + batch_size]]
            batch = self.tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=self.max_prompt_tokens,
                return_tensors="pt",
            )
            batch = self._to_device(batch)
            outputs = self.model(**batch, output_hidden_states=True)
            last = outputs.hidden_states[-1]
            idx = batch["attention_mask"].sum(dim=1) - 1
            rows = torch.arange(last.size(0), device=last.device)
            all_h.append(last[rows, idx, :].float().cpu())
        return torch.cat(all_h, dim=0)
