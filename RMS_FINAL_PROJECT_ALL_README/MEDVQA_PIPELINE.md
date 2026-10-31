# MedVQA Fine-Tuning Pipeline
### LLaVA-Med + VQA-RAD + Custom Dataset → PneumoScan /medvqa endpoint
*Parth Lathiya (23BCE215) | Dharm Patel (23BCE224) | Nirma University*

---

## Your Data — What You Have

```
medvqa/
├── VQA_RAD Dataset Public.json     ← 2,248 QA pairs (original VQA-RAD)
├── VQA_RAD Image Folder/           ← 315 chest/head/abdomen X-ray images
├── medvqa_dataset.csv              ← YOUR CUSTOM DATASET (218,652 QA pairs!)
│   Columns: qa_id, split, image_path, image_id, source,
│            patient_key, question, answer, question_type,
│            answer_type, disease, options, labels
│   Sources: CheXpert (152,219) + NIH (66,433)
│   Split:   train(153,315) / val(32,887) / test(32,450)
│   Types:   closed(184,891) + open(33,761)
└── VQA_RAD Dataset Public.xlsx / .xml
```

### Why Your Custom Dataset is Powerful

Your `medvqa_dataset.csv` is far better than raw VQA-RAD because:
- **218,652 QA pairs** vs VQA-RAD's 2,248 — nearly 100x more data
- **6 question types**: presence, normal_abnormal, identification,
  multiple_choice, body_system, two_findings
- **Multi-disease labels**: Pneumonia, Effusion, Atelectasis,
  Edema, Cardiomegaly, Consolidation, Pneumothorax (matches our 11-class model!)
- Already split into train/val/test — ready to use directly
- Images from CheXpert + NIH — same sources as our DenseNet121 training

---

## The Full Pipeline — Bird's Eye View

```
STEP 1          STEP 2           STEP 3            STEP 4          STEP 5
────────        ────────         ────────          ────────        ────────
Prepare    →    Upload to   →    Fine-tune    →    Download   →    Integrate
Dataset         Kaggle           LLaVA-Med         Model           into
(local)         (GPU cloud)      on Kaggle         (local)         FastAPI
```

---

## STEP 1 — Prepare & Understand Your Dataset (Local)

### 1.1 Dataset Statistics

| Property | Value |
|---|---|
| Total QA pairs | 218,652 |
| Training pairs | 153,315 |
| Validation pairs | 32,887 |
| Test pairs | 32,450 |
| Closed-ended | 184,891 (84.6%) |
| Open-ended | 33,761 (15.4%) |
| Unique diseases | Pneumonia, Effusion, Atelectasis, Edema, Cardiomegaly, Consolidation, Pneumothorax |

### 1.2 Question Types in Your Dataset

| Type | Count | Example |
|---|---|---|
| `presence` | 79,463 | "Is there pneumothorax in this chest X-ray?" → yes/no |
| `normal_abnormal` | 33,761 | "Are there any abnormal findings?" → yes/no |
| `identification` | 33,761 | "What abnormality is visible?" → pneumothorax |
| `multiple_choice` | 33,761 | "Which finding? (A) atelectasis (B)..." → pneumothorax |
| `body_system` | 28,510 | "Which body system is affected?" → pulmonary |
| `two_findings` | 9,396 | "Are both X and Y present?" → yes/no |

### 1.3 Important: Image Paths Issue

Your CSV has Kaggle paths like:
```
/kaggle/input/datasets/ashery/chexpert/train/patient00078/...
```
These don't exist locally. **Solution:** On Kaggle, add the original
CheXpert + NIH datasets as input datasets — the paths will match exactly.

---

## STEP 2 — What is LLaVA-Med?

### 2.1 Architecture

```
Chest X-ray Image (224×224 pixels)
          │
          ▼
  ┌───────────────────┐
  │  CLIP ViT-L/14    │  ← Vision encoder, converts image to 256 visual tokens
  │  (Vision Encoder) │    Each token = 1024-dim vector representing image patch
  └────────┬──────────┘
           │  256 visual tokens × 1024 dim
           ▼
  ┌───────────────────┐
  │  MLP Projector    │  ← 2-layer MLP maps visual tokens to LLM space
  │  (1024 → 4096)    │    This is the "bridge" between vision and language
  └────────┬──────────┘
           │  256 visual tokens × 4096 dim (LLM embedding size)
           │
           │    Question Text: "Is there pneumonia?"
           │          │
           │          ▼
           │   ┌─────────────┐
           │   │  Tokenizer  │  ← Converts text to token IDs
           │   └──────┬──────┘
           │          │  Text tokens × 4096 dim
           │          │
           └────┬─────┘
                │  Combined: [visual tokens] + [text tokens]
                ▼
  ┌───────────────────────┐
  │   LLaMA-2 7B / 13B   │  ← Main language model (7 billion parameters)
  │   (Language Model)    │    Processes combined vision + text tokens
  │                       │    Auto-regressively generates answer
  └────────┬──────────────┘
           │
           ▼
      Generated Answer
   "yes" / "no" / "pneumothorax" / full sentence
```

### 2.2 Why LLaVA-Med specifically?

| Model | Medical Pre-training | Vision | Size | Free? |
|---|---|---|---|---|
| LLaVA-Med | Yes — 60k PubMed pairs | CLIP ViT | 7B | Yes (HuggingFace) |
| LLaVA 1.5 | No | CLIP ViT | 7B/13B | Yes |
| GPT-4V | Partial | Proprietary | Unknown | No ($$$) |
| BioViL | Yes — MIMIC-CXR | ResNet | 400M | Yes |
| Med-Flamingo | Yes | NFNet | 80M | Yes |

LLaVA-Med is the best choice because:
- Already understands medical images from PubMed pre-training
- Open source, free to use
- Runs on a single T4 GPU with LoRA fine-tuning
- Architecture matches exactly what we need

### 2.3 What is LoRA Fine-Tuning?

Full fine-tuning of 7B parameters requires 8× A100 GPUs (impossible on free Kaggle).

**LoRA (Low-Rank Adaptation)** is a technique that:
- Freezes ALL original model weights (7B params — not trained)
- Adds tiny "adapter" matrices at key layers (only ~4M params trained)
- Achieves 95% of full fine-tuning quality at 1% of the cost
- Works on a single T4 GPU (16GB VRAM)

```
Original weight matrix W (frozen):
┌─────────────────────┐
│  7B parameters      │  ← NOT updated during training
│  (FROZEN)           │
└─────────────────────┘

LoRA adds two small matrices:
┌──────┐   ┌──────┐
│  A   │ × │  B   │  ← Only these ~4M params are trained
│(r×d) │   │(d×r) │    rank r=8, much smaller than original
└──────┘   └──────┘

Output = W·x + (A·B)·x × scale
              ↑ LoRA contribution (small correction)
```

---

## STEP 3 — Kaggle Setup

### 3.1 What to Upload to Kaggle as Datasets

You need 3 Kaggle datasets:

**Dataset 1: Your MedVQA CSV**
- Upload `medvqa/medvqa_dataset.csv`
- Name: `medvqa-custom-dataset`
- This has all 218k QA pairs with train/val/test splits

**Dataset 2: CheXpert Images**
- Search Kaggle: "CheXpert" — use existing public dataset
- `ashery/chexpert` — already on Kaggle, just add as input
- Paths in your CSV match this dataset exactly

**Dataset 3: NIH Chest X-ray Images**
- Search Kaggle: "NIH Chest X-rays"
- `nih-chest-xrays/data` — already on Kaggle, add as input

### 3.2 Kaggle Notebook Settings

When creating the notebook:
- **Accelerator**: GPU T4 x2 (free, 2× 16GB = 32GB total)
- **Internet**: ON (needed to download LLaVA-Med from HuggingFace)
- **Persistence**: Save & Run All
- **Session timeout**: 12 hours

### 3.3 Add Datasets as Inputs

In your Kaggle notebook, add:
```
+ Add Data → Search "chexpert" → ashery/chexpert
+ Add Data → Search "nih chest" → nih-chest-xrays
+ Add Data → Your Dataset → medvqa-custom-dataset
```

---

## STEP 4 — Training Strategy

### 4.1 Why Not Train on All 218k Pairs at Once?

218k × multiple epochs on a T4 = ~48 hours. Too long for free Kaggle (9h limit).

### 4.2 Recommended Training Subsets

**Phase A: Quick Proof of Concept (fits in 9h Kaggle session)**
- Use only CHEST images from your dataset
- Only `presence` + `normal_abnormal` question types
- Estimated: ~40,000 pairs → 2-3 hours on T4

**Phase B: Full Training (for final submission)**
- All 153,315 training pairs
- All question types
- Run in 2 Kaggle sessions (save checkpoint between)
- Estimated: ~15-20 hours total

### 4.3 Training Hyperparameters

```python
# LoRA config
lora_r          = 8        # rank — higher = more capacity, more memory
lora_alpha      = 16       # scaling factor (usually 2× rank)
lora_dropout    = 0.05     # regularization
target_modules  = ["q_proj", "v_proj"]  # which layers get LoRA adapters

# Training config
learning_rate   = 2e-4     # higher than full fine-tuning (LoRA can handle it)
batch_size      = 4        # per GPU (small due to image + text)
grad_accum      = 8        # effective batch = 4 × 8 = 32
epochs          = 3        # 3 passes through training data
warmup_ratio    = 0.03     # 3% of steps for warmup
max_seq_length  = 512      # max tokens for question + answer
```

### 4.4 Prompt Format

LLaVA-Med uses a specific prompt template — your data must be formatted this way:

```
For CLOSED questions:
──────────────────────────────────────────────
<image>
USER: Is there pneumothorax in this chest X-ray?
ASSISTANT: yes
──────────────────────────────────────────────

For OPEN questions:
──────────────────────────────────────────────
<image>
USER: What abnormality is visible in this chest X-ray?
ASSISTANT: pneumothorax
──────────────────────────────────────────────

For MULTIPLE CHOICE questions:
──────────────────────────────────────────────
<image>
USER: Which one of these findings can be seen in this radiograph?
Options: (A) atelectasis (B) pleural effusion (C) pneumothorax (D) cardiomegaly
ASSISTANT: (C) pneumothorax
──────────────────────────────────────────────
```

---

## STEP 5 — What the Training Notebook Does

The notebook `MEDVQA_TRAIN/medvqa_training.ipynb` does:

```
Cell 1: Install dependencies
        pip install transformers peft bitsandbytes accelerate
        pip install Pillow requests tqdm pandas

Cell 2: Load dataset CSV, filter & preprocess
        - Fix image paths for Kaggle
        - Format into LLaVA prompt template
        - Split: train / val

Cell 3: Download LLaVA-Med from HuggingFace
        microsoft/llava-med-v1.5-mistral-7b
        Load in 4-bit quantization (saves 75% VRAM)

Cell 4: Apply LoRA adapters
        Only q_proj, v_proj layers get adapters
        ~4M trainable params out of 7B total

Cell 5: Training loop
        Trainer with evaluation every 500 steps
        Save best checkpoint based on val loss

Cell 6: Evaluate on test set
        Accuracy for closed questions
        BLEU score for open questions

Cell 7: Save & export model
        Save LoRA adapter weights only (~80MB)
        Download medvqa_adapter.zip
```

---

## STEP 6 — After Training (Local Integration)

### 6.1 What You Download from Kaggle

```
medvqa_adapter/
├── adapter_config.json      ← LoRA configuration
├── adapter_model.bin        ← Trained adapter weights (~80MB)
└── tokenizer files
```

### 6.2 Where to Put It in Project

```
Lathiyo - Copy/
├── model/
│   ├── final_model.pt           ← existing DenseNet121
│   └── medvqa_adapter/          ← NEW: put downloaded adapter here
│       ├── adapter_config.json
│       └── adapter_model.bin
```

### 6.3 New Backend Endpoint

A new `/medvqa` endpoint is added to `backend/main.py`:

```
POST /medvqa
Input:  multipart/form-data
        - file: chest X-ray image
        - question: text question

Output: {
  "question": "Is there pneumothorax?",
  "answer": "yes",
  "confidence": 0.94,
  "answer_type": "closed",
  "attention_image": "<base64 PNG>"  ← optional
}
```

### 6.4 Frontend Integration

In `ChatBot.tsx`, questions are routed:
```
User types question
        │
        ▼
Is image uploaded? AND is question image-specific?
        │
   YES  │  NO
        │  └──→ Regular /chat (Groq LLM)
        ▼
   /medvqa endpoint
   (LLaVA-Med sees actual image)
        │
        ▼
Answer shown in chatbot
```

---

## STEP 7 — Evaluation Targets

| Metric | Closed QA Target | Open QA Target |
|---|---|---|
| Accuracy | ≥ 75% | N/A |
| BLEU-1 | N/A | ≥ 0.45 |
| BLEU-4 | N/A | ≥ 0.25 |
| BERTScore | ≥ 0.80 | ≥ 0.75 |

Published LLaVA-Med benchmarks on VQA-RAD:
- Closed accuracy: ~84%
- Open accuracy: ~66%

With your larger custom dataset these should improve.

---

## Complete File Structure After Everything

```
Lathiyo - Copy/
├── medvqa/
│   ├── medvqa_dataset.csv          ← Your 218k QA dataset
│   ├── VQA_RAD Dataset Public.json ← Original VQA-RAD
│   └── VQA_RAD Image Folder/       ← 315 original images
├── MEDVQA_TRAIN/
│   └── medvqa_training.ipynb       ← Kaggle training notebook (upload this)
├── model/
│   ├── final_model.pt              ← DenseNet121 (existing)
│   └── medvqa_adapter/             ← Download from Kaggle after training
│       ├── adapter_config.json
│       └── adapter_model.bin
├── utils/
│   ├── model_loader.py             ← existing
│   ├── gradcam.py                  ← existing
│   ├── preprocessing.py            ← existing
│   └── medvqa_loader.py            ← NEW: loads LLaVA-Med + adapter
├── backend/
│   ├── main.py                     ← add /medvqa endpoint
│   └── requirements.txt            ← add new deps
└── frontend/
    └── src/components/
        └── ChatBot.tsx             ← add image-question routing
```

---

## Quick Reference — Commands

```powershell
# Check your dataset locally
.\venv\Scripts\python.exe -c "
import pandas as pd
df = pd.read_csv('medvqa/medvqa_dataset.csv')
print(df.groupby(['split','answer_type']).size())
"

# After downloading adapter from Kaggle, test locally
.\venv\Scripts\python.exe utils/medvqa_loader.py
```

---

## Summary

| Step | Where | Time |
|---|---|---|
| 1. Understand dataset | Local — done | ✅ Done |
| 2. Upload CSV to Kaggle dataset | Kaggle website | 10 min |
| 3. Add CheXpert + NIH as inputs | Kaggle notebook | 5 min |
| 4. Upload & run training notebook | Kaggle — GPU T4 | 3–20 hours |
| 5. Download adapter weights | Kaggle → local | 5 min |
| 6. Place in model/medvqa_adapter/ | Local | 2 min |
| 7. Restart backend | Local terminal | 1 min |
| 8. Test /medvqa endpoint | Browser/Postman | 10 min |

*Total active work time: ~30 minutes. Waiting time: 3–20 hours (Kaggle trains while you sleep).*
