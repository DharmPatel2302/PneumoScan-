"""
MedVQA Model Loader
────────────────────
Loads LLaVA-Med base model + LoRA adapter weights and exposes
a simple .answer(image_pil, question) interface used by the
FastAPI /medvqa endpoint.

Requires:
  model/medvqa_adapter/
      adapter_config.json
      adapter_model.bin   (or adapter_model.safetensors)
      tokenizer files
      training_config.json

Install deps (already in requirements.txt):
  pip install transformers peft bitsandbytes accelerate
"""

import json
import os
import torch
from PIL import Image


# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────

BASE_MODEL_ID  = "microsoft/llava-med-v1.5-mistral-7b"
ADAPTER_DIR    = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model", "medvqa_adapter"
)
IMAGE_SIZE     = 224
MAX_NEW_TOKENS = 64

SYSTEM_PROMPT = (
    "You are a medical AI assistant specialized in chest X-ray analysis. "
    "Answer the following question based on the provided chest X-ray image. "
    "Be concise and clinically accurate."
)


# ──────────────────────────────────────────────────────────────
# LOADER CLASS
# ──────────────────────────────────────────────────────────────

class MedVQAModel:
    """
    Wraps LLaVA-Med + LoRA adapter for single-call inference.

    Usage:
        model = MedVQAModel()
        model.load()
        result = model.answer(pil_image, "Is there pneumothorax?")
    """

    def __init__(self, adapter_dir: str = ADAPTER_DIR):
        self.adapter_dir = adapter_dir
        self.model       = None
        self.processor   = None
        self.device      = "cuda" if torch.cuda.is_available() else "cpu"
        self._loaded     = False

    # ──────────────────────────────────────────────────────────
    def load(self):
        """
        Load base model in 4-bit + merge LoRA adapter.
        Call once at startup (takes ~60-90s on CPU, ~15s on GPU).
        """
        if self._loaded:
            return

        # Check adapter exists
        if not os.path.isdir(self.adapter_dir):
            raise FileNotFoundError(
                f"MedVQA adapter not found at: {self.adapter_dir}\n"
                "Please download medvqa_adapter.zip from Kaggle and extract\n"
                "it to model/medvqa_adapter/"
            )

        print(f"[MedVQA] Loading base model: {BASE_MODEL_ID}")
        print(f"[MedVQA] Adapter dir: {self.adapter_dir}")
        print(f"[MedVQA] Device: {self.device}")

        from transformers import (
            LlavaForConditionalGeneration,
            AutoProcessor,
            BitsAndBytesConfig,
        )
        from peft import PeftModel

        # 4-bit quantization — keeps VRAM low enough for inference
        bnb_config = BitsAndBytesConfig(
            load_in_4bit              = True,
            bnb_4bit_quant_type       = "nf4",
            bnb_4bit_compute_dtype    = torch.float16,
            bnb_4bit_use_double_quant = True,
        )

        # Load processor (tokenizer + image processor)
        self.processor = AutoProcessor.from_pretrained(
            self.adapter_dir,          # saved here during training
            trust_remote_code=True,
        )
        self.processor.tokenizer.padding_side = "left"

        # Load base model quantized
        base_model = LlavaForConditionalGeneration.from_pretrained(
            BASE_MODEL_ID,
            quantization_config = bnb_config,
            device_map          = "auto",
            trust_remote_code   = True,
            torch_dtype         = torch.float16,
        )

        # Merge LoRA adapter on top
        self.model = PeftModel.from_pretrained(
            base_model,
            self.adapter_dir,
            is_trainable=False,
        )
        self.model.eval()

        # Read training config for reference
        cfg_path = os.path.join(self.adapter_dir, "training_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            print(f"[MedVQA] Trained on {cfg.get('training_samples', '?')} samples")
            print(f"[MedVQA] Final train loss: {cfg.get('final_train_loss', '?')}")

        self._loaded = True
        print("[MedVQA] Model ready.")

    # ──────────────────────────────────────────────────────────
    def answer(
        self,
        image: Image.Image,
        question: str,
        question_type: str = "auto",
    ) -> dict:
        """
        Run inference for one (image, question) pair.

        Args:
            image         : PIL Image (any size, auto-resized)
            question      : plain text question string
            question_type : hint — 'closed', 'open', or 'auto'

        Returns:
            {
              "answer"       : str,   predicted answer text
              "confidence"   : float, softmax prob of top token (0-1)
              "answer_type"  : str,   'closed' or 'open'
              "question"     : str,   echo of input question
            }
        """
        if not self._loaded:
            self.load()

        # Preprocess image
        image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))

        # Build prompt
        prompt = (
            f"[INST] <image>\n{SYSTEM_PROMPT}\n\n"
            f"Question: {question} [/INST] Answer:"
        )

        # Tokenize
        inputs = self.processor(
            text          = prompt,
            images        = image,
            return_tensors = "pt",
            padding        = True,
        ).to(self.device)

        # Infer closed vs open from question text if not given
        if question_type == "auto":
            closed_hints = [
                "is there", "are there", "does", "do ", "can ", "has ",
                "have ", "is this", "was there", "is it", "is the"
            ]
            q_lower = question.lower()
            question_type = (
                "closed"
                if any(q_lower.startswith(h) for h in closed_hints)
                else "open"
            )

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens = MAX_NEW_TOKENS,
                do_sample      = False,             # greedy — deterministic
                temperature    = 1.0,
                pad_token_id   = self.processor.tokenizer.eos_token_id,
                output_scores  = True,
                return_dict_in_generate = True,
            )

        # Decode answer (only new tokens, not prompt)
        sequences  = output_ids.sequences
        new_tokens = sequences[0][inputs["input_ids"].shape[-1]:]
        raw_answer = self.processor.tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip()

        # Clean up common artifacts
        answer = raw_answer.split("\n")[0].strip()
        answer = answer.split("[/INST]")[0].strip()
        if answer.lower().startswith("answer:"):
            answer = answer[7:].strip()

        # Estimate confidence from first-token probability
        confidence = 0.0
        if output_ids.scores:
            first_logits = output_ids.scores[0][0]           # [vocab_size]
            first_probs  = torch.softmax(first_logits, dim=-1)
            confidence   = float(first_probs.max().item())

        return {
            "answer"       : answer,
            "confidence"   : round(confidence, 4),
            "answer_type"  : question_type,
            "question"     : question,
        }

    # ──────────────────────────────────────────────────────────
    def is_loaded(self) -> bool:
        return self._loaded


# ──────────────────────────────────────────────────────────────
# SINGLETON — one instance shared across all requests
# ──────────────────────────────────────────────────────────────

_medvqa_model: MedVQAModel | None = None


def get_medvqa_model() -> MedVQAModel:
    global _medvqa_model
    if _medvqa_model is None:
        _medvqa_model = MedVQAModel()
    return _medvqa_model


# ──────────────────────────────────────────────────────────────
# QUICK TEST (run this file directly to verify)
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("MedVQA Loader — Quick Test")
    print("=" * 50)

    adapter_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "model", "medvqa_adapter"
    )

    if not os.path.isdir(adapter_path):
        print(f"Adapter not found at: {adapter_path}")
        print("Download medvqa_adapter.zip from Kaggle first.")
        sys.exit(1)

    model = MedVQAModel()
    model.load()

    # Use a blank test image
    test_image = Image.new("RGB", (224, 224), color=(128, 128, 128))

    test_questions = [
        ("Is there pneumothorax in this chest X-ray?", "closed"),
        ("What abnormality is visible in this image?", "open"),
        ("Are the lungs normal appearing?", "closed"),
    ]

    for question, qtype in test_questions:
        result = model.answer(test_image, question, question_type=qtype)
        print(f"\nQ [{qtype}]: {question}")
        print(f"A           : {result['answer']}")
        print(f"Confidence  : {result['confidence']:.4f}")
