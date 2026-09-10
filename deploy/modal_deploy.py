import os, modal

app = modal.App("unified-valuation-engine-rev2")

MODEL_REVISION = os.getenv("HF_MODEL_REVISION", "main")
MODEL_ID = os.getenv("HF_FINE_TUNED_MODEL", "faxoi/llama-3.2-3b-price-engine-Rev2")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers>=4.48.0", "accelerate", "bitsandbytes", "huggingface_hub")
    .add_local_python_source("prompt_config")
)

@app.cls(
    image=image,
    gpu="T4",
    secrets=[modal.Secret.from_dotenv()],
    scaledown_window=300,
)
class ModelInferenceService:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import prompt_config

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        kwargs = {"token": token} if token else {}
        revision = MODEL_REVISION or "main"

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision, **kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=revision,
            torch_dtype=torch.float16,
            device_map="auto",
            **kwargs,
        )
        self.model.eval()

    @modal.method()
    def predict(self, product_description):
        import torch, prompt_config
        messages = prompt_config.build_messages(product_description.strip())
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=16, do_sample=False, use_cache=True
            )
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True
        ).strip()
        value = prompt_config.parse_price(response)
        if value is None or value <= 0:
            raise RuntimeError(f"Invalid QLoRA output: {response!r}")
        return float(value)
