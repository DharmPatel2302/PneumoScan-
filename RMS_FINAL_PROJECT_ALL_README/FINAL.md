# PneumoScan — Complete Project Documentation
### Explainable and Interactive Medical AI Assistant for Chest X-Ray Diagnosis
**Parth Lathiya (23BCE215) | Dharm Patel (23BCE224)**
**B.Tech Computer Science & Engineering — Nirma University**
**Research Methodology and Seminar (4FT901CC24)**

---

# TABLE OF CONTENTS

1. [Project Overview](#1-project-overview)
2. [Dataset Collection](#2-dataset-collection)
3. [Model 1 — DenseNet121 Multiclass Classification](#3-model-1--densenet121-multiclass-classification)
4. [Explainable AI — Grad-CAM](#4-explainable-ai--grad-cam)
5. [Model 2 — MedVQA with LLaVA-Med](#5-model-2--medvqa-with-llava-med)
6. [AI Chatbot — PneumoBot](#6-ai-chatbot--pneumobot)
7. [Backend Architecture — FastAPI](#7-backend-architecture--fastapi)
8. [Frontend Architecture — Next.js](#8-frontend-architecture--nextjs)
9. [Complete System Architecture](#9-complete-system-architecture)
10. [API Reference](#10-api-reference)
11. [How to Run](#11-how-to-run)

---

# 1. Project Overview

PneumoScan is a full-stack AI-powered chest X-ray analysis platform that combines:

- **Deep Learning** — DenseNet121 trained to detect 11 chest conditions simultaneously
- **Explainable AI (XAI)** — Grad-CAM heatmaps showing which regions the model focused on
- **Medical Visual Question Answering (MedVQA)** — LLaVA-Med fine-tuned model that visually examines the X-ray and answers image-specific questions
- **AI Chatbot** — Groq-powered LLM assistant (PneumoBot) for general medical knowledge
- **PDF Report Generation** — Downloadable clinical-style report with findings, heatmaps, and chat history

### What makes it different from a basic classifier

| Feature | Basic Classifier | PneumoScan |
|---|---|---|
| Output | Single label (Pneumonia/Normal) | 11 simultaneous disease probabilities |
| Explanation | None | Grad-CAM heatmap + XAI overlay |
| Interactivity | None | Full AI chatbot with scan context |
| Image QA | None | MedVQA model answers visual questions |
| Report | None | Downloadable PDF with full analysis |

---

# 2. Dataset Collection

## 2.1 DenseNet121 Training Dataset

### Sources Used

| Source | Images | Access |
|---|---|---|
| NIH ChestX-ray14 | 110,375 | Kaggle public dataset |
| CheXpert | 201,248 | Kaggle (ashery/chexpert) |
| **Total** | **311,623** | Merged CSV |

### Dataset Link
```
https://www.kaggle.com/datasets/dp2302/final-merged-dataset-csv
```

### CSV Structure
```
image_path, source, no_finding, Atelectasis, Cardiomegaly, Effusion,
Consolidation, Edema, Pneumonia, Pneumothorax, Infiltration,
Lung_Lesion, Pleural_Thickening
```

Each row is one image with binary labels (0/1) for each of the 11 classes.
One image can have multiple 1s — this is **multi-label classification**.

### Class Distribution

| Class | NIH Count | CheXpert Count | Total | % of Dataset |
|---|---|---|---|---|
| No Finding | 60,361 | 22,419 | 82,780 | 26.6% |
| Infiltration | 19,894 | 105,707 | 125,601 | 40.3% |
| Effusion | 13,317 | 86,254 | 99,571 | 32.0% |
| Edema | 2,303 | 52,291 | 54,594 | 17.5% |
| Atelectasis | 11,559 | 33,456 | 45,015 | 14.4% |
| Cardiomegaly | 2,776 | 27,068 | 29,844 | 9.6% |
| Pneumothorax | 5,302 | 19,456 | 24,758 | 7.9% |
| Lung_Lesion | 11,207 | 9,187 | 20,394 | 6.5% |
| Consolidation | 4,667 | 14,816 | 19,483 | 6.3% |
| Pneumonia | 1,431 | 6,047 | 7,478 | 2.4% |
| Pleural_Thickening | 3,385 | 3,524 | 6,909 | 2.2% |

### Multi-label Statistics

| Labels per Image | Count | % |
|---|---|---|
| 1 label | 175,528 | 56.3% |
| 2 labels | 81,697 | 26.2% |
| 3 labels | 41,837 | 13.4% |
| 4+ labels | 12,561 | 4.0% |

---

## 2.2 MedVQA Training Dataset

### Sources Used

| Source | QA Pairs |
|---|---|
| VQA-RAD (original) | 2,248 |
| Custom generated (CheXpert + NIH) | 218,652 |
| **Total** | **220,900** |

### Custom MedVQA Dataset Structure
```
qa_id, split, image_path, image_id, source, patient_key,
question, answer, question_type, answer_type, disease, options, labels
```

### Split
```
Train : 153,315 (70%)
Val   :  32,887 (15%)
Test  :  32,450 (15%)
```

### Question Types

| Type | Count | Example |
|---|---|---|
| presence | 79,463 | "Is there pneumothorax in this chest X-ray?" |
| normal_abnormal | 33,761 | "Are there any abnormal findings?" |
| identification | 33,761 | "What abnormality is visible?" |
| multiple_choice | 33,761 | "Which finding? (A) atelectasis (B)..." |
| body_system | 28,510 | "Which body system is affected?" |
| two_findings | 9,396 | "Are both X and Y present?" |

---

# 3. Model 1 — DenseNet121 Multiclass Classification

## 3.1 Why DenseNet121?

DenseNet121 (Densely Connected Convolutional Network) was chosen because:
- Each layer connects to ALL subsequent layers — maximum feature reuse
- Very strong on medical imaging tasks (used in original CheXNet paper)
- 121 layers deep but parameter-efficient due to dense connections
- Pre-trained on ImageNet — excellent transfer learning base
- Produces 1024-dimensional feature vector before classifier

## 3.2 Architecture

```
Input Image (224 × 224 × 3)
        │
        ▼
┌───────────────────────────────┐
│     DenseNet121 Backbone      │
│                               │
│  Initial Conv (7×7, stride 2) │
│         │                     │
│  MaxPool (3×3, stride 2)      │
│         │                     │
│  Dense Block 1 (6 layers)     │ ← 256 feature maps
│  Transition Layer 1           │
│         │                     │
│  Dense Block 2 (12 layers)    │ ← 512 feature maps
│  Transition Layer 2           │
│         │                     │
│  Dense Block 3 (24 layers)    │ ← 1024 feature maps
│  Transition Layer 3           │
│         │                     │
│  Dense Block 4 (16 layers)    │ ← 1024 feature maps  ← GradCAM target
│         │                     │
│  Global Average Pooling       │ → [1024] vector
└───────────────────────────────┘
        │
        ▼ 1024-dim feature vector
┌───────────────────────────────┐
│     Custom Classifier Head    │
│                               │
│  Linear(1024 → 512)           │
│  BatchNorm1d(512)             │
│  ReLU()                       │
│  Dropout(0.3)                 │
│  Linear(512 → 11)             │ ← 11 output logits
└───────────────────────────────┘
        │
        ▼
  Sigmoid Activation            ← independent probability per class
        │
        ▼
  [p0, p1, p2, ..., p10]        ← 11 probabilities (0.0 to 1.0)
  no_finding, Atelectasis, ...
```

### Key Detail: Sigmoid vs Softmax

Binary classification uses **Softmax** (probabilities sum to 1 — mutually exclusive).
This model uses **Sigmoid** — each class is independent.
This is critical for multi-label: a patient can have BOTH Pneumonia AND Effusion simultaneously.

## 3.3 Training Strategy (Two Phases)

### Phase 1 — Feature Extraction (Frozen Base)
```
Epochs     : 5
LR         : 1e-3
Frozen     : All DenseNet layers (only classifier trained)
Purpose    : Learn basic task mapping without destroying ImageNet features
```

### Phase 2 — Fine-Tuning (Unfreeze Top 20%)
```
Epochs     : 20
LR         : 1e-5
Unfrozen   : Top 20% of DenseNet layers + classifier
Purpose    : Fine-tune high-level features for chest X-ray domain
```

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | Binary Cross Entropy with Logits (BCEWithLogitsLoss) |
| Batch Size | 32 |
| Image Size | 224 × 224 |
| Dropout | 0.3 |
| Early Stopping | Patience = 5 (monitor: val_auc) |
| Train/Val/Test | 70% / 15% / 15% |
| Class Weights | Computed per-class to handle imbalance |

### Why BCEWithLogitsLoss?

Standard CrossEntropy assumes one correct answer per image.
BCEWithLogitsLoss treats each of the 11 outputs as a separate binary problem:
- Is there Pneumonia? → BCE loss
- Is there Effusion? → separate BCE loss
- ... (11 separate binary losses, averaged together)

## 3.4 Model Checkpoint

```
model/final_model.pt

Checkpoint keys:
  model_state_dict   ← weights
  epoch              ← 10
  best_val_auc       ← 0.7789
  disease_cols       ← ['no_finding', 'Atelectasis', ...]
  num_classes        ← 11
  img_size           ← 224
  architecture       ← 'DenseNet121'
  mean               ← [0.485, 0.456, 0.406]
  std                ← [0.229, 0.224, 0.225]
```

## 3.5 Inference Pipeline

```python
# 1. Load image
image_pil = Image.open(file).convert("RGB")

# 2. Preprocess
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std =[0.229, 0.224, 0.225])
])
input_tensor = transform(image_pil).unsqueeze(0)  # [1, 3, 224, 224]

# 3. Forward pass
logits = model(input_tensor)          # [1, 11]
probs  = torch.sigmoid(logits[0])     # [11] → values 0.0 to 1.0

# 4. Threshold at 0.5 → detected/not detected
for label, prob in zip(DISEASE_CLASSES, probs):
    detected = prob >= 0.5
```

## 3.6 Output Schema

```json
{
  "primary_label":       "Consolidation",
  "primary_display":     "Consolidation",
  "primary_probability": 72.0,
  "all_findings": [
    {"label": "Consolidation", "probability": 72.0, "detected": true},
    {"label": "Atelectasis",   "probability": 71.6, "detected": true},
    {"label": "Pneumonia",     "probability": 36.9, "detected": false},
    ...
  ],
  "detected_diseases": [...],
  "gradcam_class":   "Consolidation",
  "gradcam_display": "Consolidation",
  "original_image":  "<base64 PNG>",
  "gradcam_heatmap": "<base64 PNG>",
  "overlay_image":   "<base64 PNG>"
}
```

---

# 4. Explainable AI — Grad-CAM

## 4.1 What is Grad-CAM?

Gradient-weighted Class Activation Mapping (Grad-CAM) answers:
**"Which regions of the X-ray did the model look at to make this prediction?"**

Without XAI, a model is a black box — it gives a prediction but no reason.
Grad-CAM makes it interpretable by creating a heatmap over the image.

## 4.2 How Grad-CAM Works — Step by Step

```
Step 1: Forward Pass
        Input image → model → logit for target class (e.g. Consolidation)

Step 2: Backward Pass
        Compute gradients of the target class score
        with respect to the feature maps in Dense Block 4
        ∂score_consolidation / ∂A^k   (for each of k feature maps)

Step 3: Global Average Pooling of Gradients
        α^k = (1/Z) Σ_ij (∂score / ∂A^k_ij)
        This gives importance weight for each feature map k

Step 4: Weighted Combination
        L_GradCAM = ReLU(Σ_k α^k · A^k)
        ReLU removes negative activations (only positive influence matters)

Step 5: Upsample to 224×224
        Bilinear interpolation from feature map size → input size

Step 6: Normalize 0→1 and apply JET colormap
        Blue = low attention, Red = high attention
```

## 4.3 Target Layer

```python
target_layer = model.densenet.features.denseblock4
```

Dense Block 4 is chosen because:
- Last convolutional block before Global Average Pooling
- Contains the highest-level, most semantically meaningful features
- Spatial information is still preserved (not yet pooled away)
- Feature maps here correspond directly to image regions

## 4.4 Implementation

```python
class GradCAM:
    def __init__(self, model, target_layer):
        # Register two hooks:
        # 1. Forward hook — captures activations A^k
        # 2. Full backward hook — captures gradients ∂score/∂A^k
        self._fwd_handle = target_layer.register_forward_hook(forward_hook)
        self._bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class_idx=None):
        # If no class specified, use highest probability class
        if target_class_idx is None:
            target_class_idx = probs.argmax()

        # Backward on specific class score
        score = output[0, target_class_idx]
        score.backward()

        # Weighted combination + ReLU
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = F.relu(torch.sum(weights * activations, dim=1))

        # Resize and normalize
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear')
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return heatmap, target_class_idx
```

## 4.5 Three Output Images

| Image | Description | How Generated |
|---|---|---|
| Original Image | De-normalized X-ray | Reverse ImageNet normalization |
| Grad-CAM Heatmap | Pure attention map (JET colormap) | `cv2.applyColorMap(heatmap, COLORMAP_JET)` |
| XAI Overlay | X-ray + heatmap blended | `cv2.addWeighted(image, 0.55, heatmap_color, 0.45, 0)` |

---

# 5. Model 2 — MedVQA with LLaVA-Med

## 5.1 What is MedVQA?

Medical Visual Question Answering — an AI task where the system:
- Takes a **chest X-ray image** as input
- Takes a **natural language question** as input
- Generates a **natural language answer** based on what it visually sees

Unlike the chatbot (which only knows predicted labels), MedVQA actually **looks at image pixels** to answer.

```
Example:
  Image   : Chest X-ray
  Question: "Is there consolidation visible in the right lung?"
  Answer  : "yes"  ← model examined the actual image, not just labels
```

## 5.2 Base Model — LLaVA-Med

**LLaVA-Med** (Large Language and Vision Assistant for Medicine) by Microsoft:
- Pre-trained on **60,000 biomedical image-text pairs** from PubMed
- Architecture: CLIP ViT-L/14 (vision) + MLP projector + LLaMA/Mistral-7B (language)
- HuggingFace: `microsoft/llava-med-v1.5-mistral-7b`

### LLaVA-Med Architecture

```
Chest X-ray (224×224)
        │
        ▼
┌─────────────────────┐
│   CLIP ViT-L/14     │  ← Vision Encoder
│   (Vision Encoder)  │    Splits image into 16×16 patches
│                     │    Each patch → 1024-dim vector
│   Output: 256 visual│    256 visual tokens total
│          tokens     │
└──────────┬──────────┘
           │  [256, 1024]
           ▼
┌─────────────────────┐
│   MLP Projector     │  ← Bridge between vision and language
│   Linear(1024→4096) │    Maps visual tokens to LLM embedding space
│   GELU activation   │    So LLM can "understand" image patches
│   Linear(4096→4096) │
└──────────┬──────────┘
           │  [256, 4096]  ← visual tokens in LLM space
           │
           │    Question: "Is there consolidation?"
           │          │
           │    ┌──────────┐
           │    │ Tokenizer│ → [32, 4096] text tokens
           │    └──────────┘
           │          │
           └────┬─────┘
                │  [288, 4096]  ← combined visual + text tokens
                ▼
┌─────────────────────────┐
│   Mistral-7B LLM        │  ← Language Model Decoder
│   32 transformer layers │    Auto-regressively generates answer
│   4096 hidden dim       │    token by token
│   32 attention heads    │
└──────────┬──────────────┘
           │
           ▼
     Generated Answer
     "yes" / "no" / "consolidation" / full sentence
```

## 5.3 Fine-Tuning with LoRA

### Why LoRA?

Full fine-tuning of 7B parameters requires ~56GB VRAM (4× A100).
LoRA (Low-Rank Adaptation) trains only ~4M parameters instead of 7B.

### How LoRA Works

```
Original weight matrix W (7B total params — FROZEN):
┌─────────────────────────────────────┐
│  W  (d × d) — NOT updated           │
└─────────────────────────────────────┘

LoRA adds two small matrices (TRAINABLE):
┌───────┐   ┌───────┐
│   A   │ × │   B   │   rank r=8
│(r × d)│   │(d × r)│   only 2 × r × d params per layer
└───────┘   └───────┘

Forward pass: output = W·x  +  (B · A) · x · (α/r)
                              ↑ LoRA correction (tiny but effective)
```

### LoRA Configuration

```python
lora_config = LoraConfig(
    r              = 8,           # rank of adaptation matrices
    lora_alpha     = 16,          # scaling = alpha/r = 2.0
    lora_dropout   = 0.05,
    target_modules = ['q_proj', 'v_proj'],  # attention query & value
    bias           = 'none',
    task_type      = 'CAUSAL_LM',
)
# Result: 4,194,304 trainable / 7,067,861,504 total = 0.06%
```

## 5.4 Training Dataset

```
medvqa/medvqa_dataset.csv
  Total: 218,652 QA pairs
  Train: 153,315
  Val:    32,887
  Test:   32,450

For 30-min training (quick demo):
  max_train_samples = 3,000
  answer_type       = 'closed' only
  epochs            = 1
  max_seq_length    = 256
```

## 5.5 Prompt Format

```
[INST] <image>
You are a medical AI assistant specialized in chest X-ray analysis.
Answer the following question based on the provided chest X-ray image.

Question: Is there pneumothorax in this chest X-ray? [/INST] Answer: yes
```

The model only learns to predict the **answer tokens** — the prompt is masked
with label `-100` so loss is computed only on the answer.

## 5.6 Training Hyperparameters

| Parameter | Quick (30-min) | Full Training |
|---|---|---|
| Training samples | 3,000 | 153,315 |
| Epochs | 1 | 3 |
| Batch size | 4 | 4 |
| Gradient accumulation | 4 | 8 |
| Effective batch | 16 | 32 |
| Learning rate | 2e-4 | 2e-4 |
| Max sequence length | 256 | 512 |
| LoRA r | 8 | 8 |
| GPU | T4 (16GB) | T4 x2 (32GB) |
| Est. time | ~25 min | ~15-20 hours |

## 5.7 Inference Pipeline

```python
# 1. Load X-ray image
image = Image.open(file).convert("RGB").resize((224, 224))

# 2. Build prompt
prompt = (
    "[INST] <image>\n"
    "You are a medical AI assistant...\n\n"
    f"Question: {question} [/INST] Answer:"
)

# 3. Tokenize image + text together
inputs = processor(text=prompt, images=image, return_tensors='pt')

# 4. Generate answer (greedy decoding)
output_ids = model.generate(
    **inputs,
    max_new_tokens = 64,
    do_sample      = False,   # deterministic
)

# 5. Decode only new tokens (not the prompt)
new_tokens = output_ids[0][inputs['input_ids'].shape[-1]:]
answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
# Output: "yes" or "pneumothorax" or full sentence
```

## 5.8 Current Implementation Status

| Component | Status |
|---|---|
| Training notebook | Done (`MEDVQA_TRAIN/medvqa_training.ipynb`) |
| FastAPI endpoint `/medvqa` | Done (`backend/main.py`) |
| Model loader | Done (`utils/medvqa_loader.py`) |
| Frontend routing | Done (`ChatContext.tsx`) |
| Actual trained adapter | Pending Kaggle training |
| Demo mode (LLM simulates MedVQA) | **Active** |

### Demo Mode Explanation

While the MedVQA adapter is being trained on Kaggle, all questions go to `/chat`.
The backend's `classify_question()` function detects image-specific questions and
switches to a **MedVQA system prompt** — the LLM answers as if it is examining the
actual X-ray. The blue `● MedVQA Vision Model` badge differentiates this from normal
LLM responses visually.

Once adapter is ready: change `MEDVQA_READY = false` → `MEDVQA_READY = true`
in `frontend/src/context/ChatContext.tsx`.

---

# 6. AI Chatbot — PneumoBot

## 6.1 Architecture

```
User types message
        │
        ▼
classify_question(message)
        │
        ├── Image keywords detected? ──YES──→ mode = "medvqa"
        │                                      System: MedVQA-Bot persona
        │                                      Tone: visual examination
        │
        └── No image keywords ──────────→  mode = "llm"
                                           System: PneumoBot persona
                                           Tone: general medical knowledge
        │
        ▼
Groq API (qwen/qwen3.8-27b model)
        │
        ▼
response.source = "medvqa" | "llm"
        │
        ▼
Frontend renders:
  "medvqa" → Blue bubble + "● MedVQA Vision Model" badge
  "llm"    → Green bubble + "● PneumoBot LLM" badge
```

## 6.2 Question Classification

```python
IMAGE_KEYWORDS = [
    "this x-ray", "this image", "this scan", "this chest x-ray",
    "in the image", "in the scan", "in the x-ray",
    "left lung", "right lung", "highlighted", "heatmap", "overlay",
    "describe this image", "explain this image",
    "what is shown in", "what abnormality is visible",
    ...
]

def classify_question(message):
    q = message.lower()
    return "medvqa" if any(kw in q for kw in IMAGE_KEYWORDS) else "llm"
```

## 6.3 LLM Provider

```
Provider : Groq
Model    : qwen/qwen3.8-27b
Speed    : ~1-2 second response time
Free tier: Yes
API key  : GROQ_API_KEY in backend/.env
```

## 6.4 Chat Features

- **Persistent across pages** — chat history survives navigation (React Context in layout.tsx)
- **Scan context injection** — when a scan is uploaded, all_findings are injected into system prompt
- **Unread notification** — toast popup when reply arrives while chat is closed
- **Unread badge** — red count on chat bubble
- **Chat history in PDF** — all messages included in downloaded report
- **Suggested questions** — shown on first load based on whether scan is uploaded
- **Clear chat** — resets messages and scan context together

---

# 7. Backend Architecture — FastAPI

## 7.1 File Structure

```
backend/
├── main.py           ← All API endpoints
└── .env              ← GROQ_API_KEY

utils/
├── model_loader.py   ← ChestXRayModel class + load_model()
├── preprocessing.py  ← transform pipeline + denormalize()
├── gradcam.py        ← GradCAM class + create_overlay()
└── medvqa_loader.py  ← MedVQAModel class + get_medvqa_model()

model/
├── final_model.pt    ← DenseNet121 trained weights
└── medvqa_adapter/   ← LLaVA-Med LoRA adapter (after Kaggle training)
```

## 7.2 Startup Sequence

```python
@app.on_event("startup")
async def load_model_on_startup():
    1. Load DenseNet121 from model/final_model.pt
    2. Register GradCAM hooks on denseblock4
    3. Check GROQ_API_KEY in .env
    4. Check if medvqa_adapter/ exists
       → If yes: load LLaVA-Med + merge LoRA adapter
       → If no:  skip (endpoint returns 503)
```

## 7.3 API Endpoints

| Method | URL | Purpose |
|---|---|---|
| GET | `/health` | System health + loaded classes |
| POST | `/predict` | X-ray analysis (DenseNet + GradCAM) |
| POST | `/chat` | AI chatbot (Groq LLM) |
| GET | `/medvqa/status` | Check if MedVQA adapter is loaded |
| POST | `/medvqa` | Visual QA (LLaVA-Med inference) |

## 7.4 /predict Endpoint — Detailed Flow

```
1. Receive image file (JPEG/PNG)
2. PIL.Image.open() → convert to RGB
3. transform() → [1, 3, 224, 224] tensor
4. model(tensor) → [1, 11] logits
5. sigmoid(logits) → [11] probabilities (0-1)
6. Build ClassResult for each of 11 classes
7. Sort by probability descending
8. Find detected diseases (prob >= 0.5, exclude no_finding)
9. Pick top detected disease as GradCAM target
10. GradCAM.generate(tensor, target_class_idx)
    → heatmap [224, 224], used_class_idx
11. denormalize(tensor) → original RGB image numpy
12. create_overlay(image, heatmap) → blended image
13. Convert all 3 images to base64 PNG
14. Return PredictionResponse JSON
```

---

# 8. Frontend Architecture — Next.js

## 8.1 File Structure

```
frontend/src/
├── app/
│   ├── layout.tsx          ← Root layout (ChatProvider + ChatBot rendered here)
│   ├── page.tsx            ← Home page (upload + results)
│   ├── about/page.tsx      ← About page
│   └── gallery/page.tsx    ← Gallery page
├── components/
│   ├── Navbar.tsx          ← Navigation bar
│   ├── Footer.tsx          ← Footer
│   ├── UploadSection.tsx   ← Drag & drop X-ray upload
│   ├── LoadingModal.tsx    ← Full-screen loading animation
│   ├── ResultsSection.tsx  ← Multiclass results + XAI images
│   ├── ChatBot.tsx         ← Floating chat widget
│   ├── GalleryGrid.tsx     ← Gallery page image grid
│   └── PageTransition.tsx  ← Page transition animation
├── context/
│   └── ChatContext.tsx     ← Global state (persists across pages)
└── utils/
    └── generateReport.ts   ← jsPDF report generator
```

## 8.2 Global State (ChatContext)

Everything that must survive page navigation lives in `ChatContext`:

```typescript
// State stored in ChatContext (never reset on navigation)
messages:           Message[]           // full chat history
open:               boolean             // chat window open/closed
minimised:          boolean             // chat minimised
unread:             number              // unread message count
loading:            boolean             // waiting for API response
predictionContext:  PredictionContext   // scan result summary for chatbot
lastReply:          string              // for notification toast
result:             PredictionResult    // FULL scan result (11 classes + images)
apiError:           string              // upload error message
```

## 8.3 ResultsSection — What It Shows

```
┌────────────────────────────────────────────┐
│  PRIMARY FINDING BANNER                    │
│  Consolidation  72.0%  [chip] [chip] ...   │
│  Detected: Consolidation, Atelectasis      │
└────────────────────────────────────────────┘
┌────────────────────────────────────────────┐
│  ALL CLASSIFICATION FINDINGS               │
│  ● Consolidation  ████████████░  72.0%     │
│  ● Atelectasis    ████████████░  71.6%     │
│  ● Infiltration   ████████░░░░░  39.6%     │
│  ● Pneumonia      ███████░░░░░░  36.9%     │
│  ... (all 11 classes with bars)            │
└────────────────────────────────────────────┘
┌────────────────────────────────────────────┐
│  INTERPRETABILITY ANALYSIS (XAI)           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │ Original │ │ GradCAM  │ │ Overlay  │   │
│  │  Image   │ │ Heatmap  │ │          │   │
│  └──────────┘ └──────────┘ └──────────┘   │
│  Click any → full-size modal + download    │
└────────────────────────────────────────────┘
```

## 8.4 ChatBot Widget — States

```
CLOSED STATE:
  Green circle (56×56px) bottom-right
  Red badge shows unread count
  Toast notification for new messages

OPEN STATE:
  390px wide, 600px tall panel
  Header: PneumoBot name + status + clear/minimise/close
  Context pill: shows current scan findings
  Messages: scrollable list of bubbles
  Suggested questions: shown when chat is empty
  Input: textarea + send button

MESSAGE BUBBLE THEMES:
  User message    → Green bubble (right-aligned)
  LLM response    → Dark grey bubble + green "● PneumoBot LLM" badge
  MedVQA response → Dark blue bubble + blue pulsing "● MedVQA Vision Model" badge
  Error message   → Dark red bubble
```

## 8.5 PDF Report Contents

```
Page 1:
  Header: PneumoScan logo + report ID + timestamp
  Section 01: Diagnosis Result
    - Primary finding badge + confidence score
    - Confidence bar
    - Summary text

  Section 02: Imaging Analysis (Grad-CAM XAI)
    - Three images side by side: Original, Heatmap, Overlay
    - Labels and captions

  Section 03: Clinical Recommendations
    - 5-7 bullet points based on diagnosis

  Section 04: PneumoBot Chat Summary (if messages exist)
    - All messages in colour-coded bubbles (blue=user, green=bot)

  Section 05: Disclaimer
    - Research/educational use only warning

Footer on every page: Report ID + page number
```

---

# 9. Complete System Architecture

## 9.1 End-to-End Flow

```
USER ACTION                   FRONTEND                BACKEND               AI MODELS
──────────────                ────────────────────    ───────────────────   ──────────────────

1. Open browser          →    Next.js loads           FastAPI starts
   localhost:3000              layout.tsx              loads DenseNet121
                               ChatProvider mounts     registers GradCAM hooks
                               ChatBot renders         checks GROQ_API_KEY
                               on every page           checks medvqa_adapter

2. User drags           →    UploadSection.tsx        POST /predict
   X-ray image               FormData with image       ├── PIL decode
                                                        ├── transform [1,3,224,224]
                                                        ├── DenseNet121 forward
                                                        │   logits [1,11]
                                                        ├── sigmoid → probs [11]
                                                        ├── build ClassResult ×11
                                                        ├── GradCAM.generate()
                                                        │   target: top disease
                                                        ├── denormalize image
                                                        ├── create_overlay()
                                                        └── return JSON
                              setResult(data)
                              setPredictionContext()

3. ResultsSection       →    Renders:
   shown                      Primary finding banner
                               11-class findings panel
                               XAI image cards (×3)
                               Download report button

4. User clicks image    →    XAIImageCard modal
                               dark modal, no backdrop
                               download button works
                               badge shows prediction

5. User types          →    ChatBot.tsx             POST /chat
   "what is pneumonia?"       sendViaChat()          classify_question()
                                                      → mode: "llm"
                                                      build_system_prompt(llm)
                                                      Groq API call
                                                      return {reply, source:"llm"}
                              source="llm"
                              → green bubble rendered

6. User types          →    ChatBot.tsx             POST /chat
   "what do you see           sendViaChat()          classify_question()
    in this x-ray?"                                  → mode: "medvqa"
                                                      build_system_prompt(medvqa)
                                                      Groq API as MedVQA persona
                                                      return {reply, source:"medvqa"}
                              source="medvqa"
                              → blue bubble rendered

7. User navigates      →    ChatContext persists     (no API call)
   to Gallery/About          result, messages,
   then back to Home         predictionContext
                              → same results shown

8. User clicks         →    generateReport.ts        (no API call)
   Download Report            jsPDF builds PDF
                               adds images, chat history
                               saves to disk
```

## 9.2 Technology Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| **Frontend** | Next.js | 14 | React framework with App Router |
| **Frontend** | TypeScript | 5 | Type safety |
| **Frontend** | Tailwind CSS | 3 | Styling |
| **Frontend** | Lucide React | latest | Icons |
| **Frontend** | jsPDF | latest | PDF generation |
| **Backend** | FastAPI | 0.111 | REST API |
| **Backend** | Python | 3.12 | Runtime |
| **Backend** | Uvicorn | 0.29 | ASGI server |
| **ML** | PyTorch | 2.x | Deep learning |
| **ML** | Torchvision | 0.x | Image transforms |
| **ML** | DenseNet121 | ImageNet pretrained | X-ray classification |
| **XAI** | OpenCV | 4.x | Heatmap colormap + overlay |
| **MedVQA** | Transformers | 4.40 | LLaVA-Med loading |
| **MedVQA** | PEFT | 0.10 | LoRA adapters |
| **MedVQA** | BitsAndBytes | 0.43 | 4-bit quantization |
| **LLM** | Groq | latest | Fast LLM inference |
| **LLM** | Qwen 3.8-27B | hosted | Chat responses |

## 9.3 Data Flow Diagram

```
                    ┌─────────────────────────────┐
                    │         BROWSER             │
                    │   localhost:3000            │
                    │                             │
                    │  ┌───────────────────────┐  │
                    │  │    Next.js Frontend    │  │
                    │  │                        │  │
                    │  │  layout.tsx            │  │
                    │  │    └─ ChatProvider     │  │
                    │  │         └─ ChatBot     │  │
                    │  │                        │  │
                    │  │  page.tsx              │  │
                    │  │    ├─ UploadSection    │  │
                    │  │    └─ ResultsSection   │  │
                    │  └───────────┬────────────┘  │
                    └─────────────┼───────────────┘
                                  │ HTTP (localhost)
                    ┌─────────────▼───────────────┐
                    │      FastAPI Backend         │
                    │      localhost:8000          │
                    │                              │
                    │  POST /predict               │
                    │    └─ DenseNet121 ──────────►│ model/final_model.pt
                    │    └─ GradCAM                │
                    │                              │
                    │  POST /chat                  │
                    │    └─ classify_question()    │
                    │    └─ Groq API ─────────────►│ api.groq.com
                    │                              │
                    │  POST /medvqa                │
                    │    └─ LLaVA-Med ────────────►│ model/medvqa_adapter/
                    │    └─ LoRA inference         │ (after Kaggle training)
                    └──────────────────────────────┘
```

## 9.4 Key Design Decisions

| Decision | Why |
|---|---|
| **Sigmoid not Softmax** | Multi-label — patient can have multiple diseases simultaneously |
| **GradCAM on denseblock4** | Last conv layer with spatial info — best for localization |
| **Global React Context** | Scan result + chat must survive page navigation |
| **ChatBot in layout.tsx** | Renders once, persists across all pages |
| **LoRA not full fine-tuning** | 7B model needs only ~80MB adapter, trains on single T4 GPU |
| **4-bit quantization** | Reduces LLaVA-Med VRAM from 28GB → 7GB for inference |
| **Groq over OpenAI** | Free tier, ~1s response, sufficient for medical QA |
| **Source tag on messages** | Visually demonstrates two separate AI modules for faculty |
| **MEDVQA_READY flag** | Single line to enable real MedVQA once adapter is trained |

---

# 10. API Reference

## POST /predict

**Request:** `multipart/form-data`
```
file: image/jpeg or image/png
```

**Response:**
```json
{
  "primary_label":       "Consolidation",
  "primary_display":     "Consolidation",
  "primary_probability": 72.0,
  "all_findings": [
    {
      "label":        "Consolidation",
      "display_name": "Consolidation",
      "probability":  72.0,
      "detected":     true
    }
  ],
  "detected_diseases": [...],
  "gradcam_class":    "Consolidation",
  "gradcam_display":  "Consolidation",
  "original_image":   "<base64 PNG>",
  "gradcam_heatmap":  "<base64 PNG>",
  "overlay_image":    "<base64 PNG>"
}
```

## POST /chat

**Request:**
```json
{
  "message":             "What does this result mean?",
  "history":             [{"role": "user", "content": "..."}],
  "primary_label":       "Consolidation",
  "primary_display":     "Consolidation",
  "primary_probability": 72.0,
  "detected_diseases":   ["Consolidation", "Atelectasis"],
  "has_image":           true
}
```

**Response:**
```json
{
  "reply":  "The primary finding of consolidation...",
  "source": "llm"   // or "medvqa"
}
```

## POST /medvqa

**Request:** `multipart/form-data`
```
file:          image/jpeg or image/png
question:      "Is there pneumothorax?"
question_type: "auto" | "closed" | "open"
```

**Response:**
```json
{
  "question":    "Is there pneumothorax?",
  "answer":      "yes",
  "confidence":  0.94,
  "answer_type": "closed",
  "model_ready": true
}
```

---

# 11. How to Run

## Terminal 1 — Backend

```powershell
cd "c:\Users\Dharm\OneDrive\Desktop\RMS\Lathiyo - Copy\backend"
& "c:\Users\Dharm\OneDrive\Desktop\RMS\Lathiyo - Copy\venv\Scripts\python.exe" -m uvicorn main:app --reload --port 8000
```

**First time only:**
```powershell
& "c:\Users\Dharm\OneDrive\Desktop\RMS\Lathiyo - Copy\venv\Scripts\python.exe" -m pip install -r "c:\Users\Dharm\OneDrive\Desktop\RMS\Lathiyo - Copy\backend\requirements.txt"
```

Expected startup log:
```
[INFO] Multiclass model loaded on cpu
[INFO] Groq API key found — chat endpoint ready
[INFO] MedVQA adapter not found — /medvqa disabled
Uvicorn running on http://127.0.0.1:8000
```

## Terminal 2 — Frontend

```powershell
cd "c:\Users\Dharm\OneDrive\Desktop\RMS\Lathiyo - Copy\frontend"
npm run dev
```

Open: **http://localhost:3000**

## Health Check

Visit http://localhost:8000/health — should return:
```json
{
  "status": "ok",
  "model_loaded": true,
  "classes": ["no_finding", "Atelectasis", "Cardiomegaly", ...]
}
```

---

*Document prepared for RMS-II submission*
*PneumoScan — Nirma University, B.Tech CSE 2025–26*
*Parth Lathiya (23BCE215) | Dharm Patel (23BCE224)*
