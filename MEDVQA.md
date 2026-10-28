  # MedVQA — Medical Visual Question Answering
### Detailed Research & Implementation Guide
**Project: PneumoScan — Explainable and Interactive Medical AI Assistant**
*Parth Lathiya (23BCE215) | Dharm Patel (23BCE224) | Nirma University*

---

## 1. What is MedVQA?

**Medical Visual Question Answering (MedVQA)** is an AI task where a system takes:
- A **medical image** (chest X-ray, MRI, CT scan, pathology slide) as input
- A **natural language question** about that image as input
- And produces a **natural language answer** as output

It is the intersection of two AI fields:
- **Computer Vision** — understanding what is in the image
- **Natural Language Processing (NLP)** — understanding and generating human language

### Simple Example

```
Image   : Chest X-ray of a patient
Question: "What abnormality is visible in the right lung?"
Answer  : "There is consolidation in the right lower lobe, 
           suggesting bacterial pneumonia."
```

### Why is it "Medical" VQA specifically?

General VQA (like asking questions about photos of cats or cars) is well-solved.
Medical VQA is much harder because:
- Medical images look very different from natural photos
- Answers require clinical domain knowledge
- A wrong answer can have serious consequences
- Medical terminology is highly specialized
- Small visual differences have large clinical significance

---

## 2. How MedVQA is Different from a Regular Chatbot

| Feature | Regular Chatbot (what we have now) | MedVQA |
|---|---|---|
| Input | Text only | Image + Text |
| Image understanding | No — just uses prediction labels | Yes — actually sees the X-ray |
| Question type | General medical knowledge | Specific to the uploaded image |
| Example question | "What is pneumonia?" | "What region of this X-ray shows infection?" |
| Answer source | LLM knowledge | Visual features extracted from the image |
| Accuracy on image-specific questions | Low | High |

**Our current PneumoScan chatbot** only knows:
- The predicted label (e.g. "Pneumonia: 94%")
- The list of detected conditions

It **cannot** answer: *"Where exactly is the consolidation in this X-ray?"* because it never actually sees the image pixels.

**MedVQA** would actually look at the image and answer such questions.

---

## 3. Types of MedVQA

### 3.1 Closed-Set VQA (Classification-Based)
The model picks an answer from a **fixed set of possible answers**.

```
Question : "Is there pneumonia present?"
Answers  : [Yes, No, Maybe]  ← model picks one
```

- Easier to train and evaluate
- Limited — cannot give detailed explanations
- Good for yes/no, abnormality detection, organ identification
- Dataset example: VQA-RAD, PathVQA

### 3.2 Open-Ended VQA (Generation-Based)
The model **generates a free-text answer** of any length.

```
Question : "Describe the findings in the left lung."
Answer   : "The left lung shows increased opacity in the 
            lower lobe with air bronchograms, consistent 
            with lobar pneumonia."
```

- Much harder
- More clinically useful
- Requires language generation capability
- Dataset example: MIMIC-CXR, IU X-Ray

### 3.3 Visual Grounding VQA
The model **highlights the region** in the image while answering.

```
Question : "Where is the abnormality?"
Answer   : "Here" + [bounding box drawn on the X-ray]
```

- Combines VQA with object detection
- Very useful for clinical explanation
- Most complex to implement

### 3.4 Report-Based VQA
The model uses the **radiology report** as additional context.

```
Inputs   : X-ray image + radiology report text
Question : "Is the finding in this report visible in the image?"
Answer   : "Yes, the cardiomegaly mentioned in the report is 
            visible as an enlarged cardiac silhouette."
```

- Most clinically accurate
- Bridges imaging and textual clinical records

---

## 4. How MedVQA Works — Architecture

### 4.1 Core Components

Every MedVQA system has 3 main parts:

```
┌─────────────────┐     ┌─────────────────┐
│   IMAGE INPUT   │     │  TEXT QUESTION  │
│  (Chest X-ray)  │     │  (Natural Lang) │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  IMAGE ENCODER  │     │  TEXT ENCODER   │
│  (CNN / ViT)    │     │  (BERT / GPT)   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └──────────┬────────────┘
                    │
                    ▼
         ┌─────────────────┐
         │  FUSION MODULE  │
         │ (Attention/MLP) │
         └────────┬────────┘
                    │
                    ▼
         ┌─────────────────┐
         │  ANSWER DECODER │
         │ (Classifier or  │
         │  Text Generator)│
         └─────────────────┘
                    │
                    ▼
              FINAL ANSWER
```

### 4.2 Image Encoder — What It Does

Converts the X-ray image into a **feature vector** (a list of numbers that represent visual information).

**Options:**

| Encoder | Description | Best For |
|---|---|---|
| DenseNet121 | What we already use | Disease classification features |
| ResNet50/101 | Classic CNN backbone | General image features |
| Vision Transformer (ViT) | Patch-based attention | Fine-grained visual details |
| BioViL | Biomedical vision-language pre-trained | Medical VQA specifically |
| CheXNet | Chest X-ray specific DenseNet | Chest radiology |

In our project, we can **reuse our existing DenseNet121** as the image encoder — this is a major advantage since it is already trained on chest X-rays.

### 4.3 Text Encoder — What It Does

Converts the question text into a **feature vector** (numbers representing meaning).

**Options:**

| Encoder | Description |
|---|---|
| BERT | Bidirectional transformer, understands context |
| BioBERT | BERT fine-tuned on biomedical text |
| ClinicalBERT | BERT fine-tuned on clinical notes |
| PubMedBERT | BERT pre-trained from scratch on PubMed |
| GPT-2 / GPT-3 | Auto-regressive, good for generation |

For medical VQA, **BioBERT or PubMedBERT** is ideal because they understand medical terminology (e.g., "consolidation", "atelectasis", "opacity").

### 4.4 Fusion Module — Combining Image + Text

This is the most critical and innovative part.
It merges the visual features and text features together.

**Methods:**

**a) Simple Concatenation**
```python
combined = torch.cat([image_features, text_features], dim=-1)
```
- Simplest approach
- Works but misses cross-modal relationships

**b) Element-wise Multiplication (Hadamard Product)**
```python
combined = image_features * text_features
```
- Forces alignment between visual and text dimensions
- Used in many early VQA systems

**c) Cross-Attention (Transformer-based)**
```python
# Image features attend to text features and vice versa
attended = cross_attention(query=image_features, 
                           key=text_features, 
                           value=text_features)
```
- State-of-the-art approach
- The model learns WHICH visual regions are relevant to WHICH words in the question
- Example: for question "Is the RIGHT lung normal?" → model focuses on right side of X-ray

**d) Co-Attention**
Both modalities attend to each other simultaneously.
Used in models like MCAN (Multi-head Co-Attention Network).

### 4.5 Answer Decoder

**For Closed-Set VQA:**
```python
output = Linear(combined_features, num_answer_classes)
answer = softmax(output)  # pick highest probability answer
```

**For Open-Ended VQA:**
```python
# Use a language model decoder (like GPT)
answer_tokens = language_model.generate(
    encoder_hidden_states=combined_features,
    max_length=100
)
answer = tokenizer.decode(answer_tokens)
```

---

## 5. Training MedVQA

### 5.1 What Data is Needed

MedVQA models need **triplets** of (image, question, answer):

```
{
  "image_path": "chest_xray_001.jpg",
  "question": "Is there any pleural effusion?",
  "answer": "Yes"
}
```

### 5.2 Key Datasets

| Dataset | Images | QA Pairs | Type | Access |
|---|---|---|---|---|
| **VQA-RAD** | 315 radiology images | 3,515 QA pairs | Closed + Open | Free |
| **PathVQA** | 5,000 pathology images | 32,799 QA pairs | Closed + Open | Free |
| **SLAKE** | 642 radiology images | 14,028 QA pairs | Bilingual (EN+ZH) | Free |
| **MIMIC-CXR** | 227,827 X-rays | With reports | Report-based | PhysioNet (free with registration) |
| **IU X-Ray** | 7,470 X-rays | With reports | Report-based | Free |
| **MedVQA 2019** | ~4,000 images | ~15,000 QA pairs | Closed-set | Free |

**For our project**, VQA-RAD + SLAKE are the best starting points — small enough to train on limited compute, high quality, and chest X-ray focused.

### 5.3 Training Strategy

#### Phase 1 — Pre-training (Optional but powerful)
Train on large general medical image-text pairs to give the model general medical visual-language understanding before fine-tuning on VQA.

Datasets for pre-training: MIMIC-CXR reports, PubMed figure captions.

#### Phase 2 — Fine-tuning on VQA
Train the full model end-to-end on VQA triplets.

```python
# Training loop pseudocode
for image, question, answer in vqa_dataset:
    
    # Encode
    image_features = image_encoder(image)      # [batch, 1024]
    text_features  = text_encoder(question)    # [batch, 768]
    
    # Fuse
    combined = fusion_module(image_features, text_features)
    
    # Predict
    predicted_answer = answer_decoder(combined)
    
    # Loss
    loss = cross_entropy(predicted_answer, ground_truth_answer)
    loss.backward()
    optimizer.step()
```

#### Loss Functions

| Task | Loss Function |
|---|---|
| Closed-set (yes/no) | Binary Cross Entropy |
| Closed-set (multi-class) | Cross Entropy |
| Open-ended generation | Cross Entropy on token sequence |
| Mixed | Weighted combination |

### 5.4 Evaluation Metrics

| Metric | What It Measures |
|---|---|
| **Accuracy** | % of exact answer matches |
| **BLEU Score** | N-gram overlap between predicted and reference answer |
| **ROUGE Score** | Recall-oriented overlap, better for longer answers |
| **BERTScore** | Semantic similarity using BERT embeddings |
| **Open/Closed Split Accuracy** | Separate accuracy for open vs closed questions |

---

## 6. Modern Approach — Vision-Language Models (VLMs)

Instead of building from scratch, modern MedVQA uses **large pre-trained Vision-Language Models** fine-tuned for medical tasks.

### 6.1 Available Models

| Model | Base | Medical Fine-tuned | Use |
|---|---|---|---|
| **LLaVA-Med** | LLaVA (LLaMA + CLIP) | Yes — on PubMed figures | Best open-source medical VLM |
| **BioViL-T** | ResNet + BERT | Yes — on MIMIC-CXR | Temporal chest X-ray analysis |
| **MedFlamingo** | Flamingo | Yes — medical images | Few-shot medical VQA |
| **CheXagent** | Flan-T5 + ViT | Yes — chest X-ray | Chest-specific VQA + report gen |
| **GPT-4V / GPT-4o** | GPT-4 + Vision | Partially | Commercial, very capable |
| **Llama 4 Scout** | Llama 4 | No | General vision, good reasoning |

### 6.2 LLaVA-Med Architecture (Recommended for our project)

```
Chest X-ray Image
       │
       ▼
  CLIP ViT Encoder          ← Converts image to visual tokens
  (vision encoder)
       │
       ▼
  MLP Projection Layer      ← Maps visual tokens to LLM embedding space
       │
       ▼
  LLaMA Language Model  ←── Question (tokenized)
       │
       ▼
  Generated Answer
```

LLaVA-Med was trained on **60,000 biomedical image-text pairs** from PubMed and is specifically designed for medical image understanding.

---

## 7. How MedVQA Fits into Our PneumoScan Project

### 7.1 Current System (What We Have)

```
User uploads X-ray
        │
        ▼
DenseNet121 predicts → [Pneumonia: 94%, Effusion: 30%, ...]
        │
        ▼
Grad-CAM generates heatmap
        │
        ▼
Chatbot answers using prediction labels + LLM knowledge
```

**Limitation:** Chatbot never sees the actual image. It only knows the predicted class names and probabilities.

### 7.2 With MedVQA Integrated

```
User uploads X-ray
        │
        ▼
DenseNet121 predicts → [Pneumonia: 94%, Effusion: 30%, ...]
        │
        ├─────────────────────────────────────┐
        ▼                                     ▼
Grad-CAM heatmap                    MedVQA Module
                                    (sees actual image)
                                             │
User asks question                           │
"Where is the consolidation?" ──────────────▶│
                                             │
                                    Visual features +
                                    Text features fused
                                             │
                                             ▼
                                   "The consolidation is 
                                    visible in the right 
                                    lower lobe, shown by 
                                    increased opacity."
```

### 7.3 Query Routing (From Slide 10 — Step 6)

As described in your presentation, we need **intelligent routing**:

```
User Question
      │
      ▼
Question Classifier
(Is this image-specific or general medical?)
      │
      ├── Image-specific → Route to MedVQA
      │   "What does the heatmap show?"
      │   "Where is the abnormality?"
      │   "Is the right lung clear?"
      │
      └── General medical → Route to LLM Chatbot
          "What is pneumonia?"
          "How is atelectasis treated?"
          "What causes pleural effusion?"
```

**How to classify the question:**
- Simple keyword detection (contains "this", "here", "image", "X-ray", "visible")
- Or fine-tune a small BERT classifier on question types

---

## 8. Implementation Plan for Our Project

### Step 1 — Dataset Preparation

Download VQA-RAD dataset (315 chest X-rays, 3,515 QA pairs):
```
https://osf.io/89kps/
```

Format it as:
```json
[
  {
    "image_id": "synpic100132.jpg",
    "question": "Is there evidence of consolidation?",
    "answer": "yes",
    "answer_type": "closed",
    "category": "abnormality"
  }
]
```

### Step 2 — Choose Architecture

**Option A: Fine-tune LLaVA-Med (Recommended)**
- Download pre-trained LLaVA-Med weights from HuggingFace
- Fine-tune on VQA-RAD
- Integrate into our FastAPI backend as a new `/medvqa` endpoint

**Option B: Build Custom (Educational)**
- Use our DenseNet121 as image encoder
- Use BioBERT as text encoder
- Train fusion + classifier on VQA-RAD
- Simpler, more understandable, less accurate

### Step 3 — Backend Integration

Add a new endpoint to `backend/main.py`:

```python
@app.post("/medvqa")
async def medvqa(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    # Load and preprocess image
    image = preprocess(file)
    
    # Run MedVQA model
    answer = medvqa_model.answer(image, question)
    
    # Optionally generate attention map showing
    # which region the model focused on
    attention_map = medvqa_model.get_attention(image, question)
    
    return {
        "question": question,
        "answer": answer,
        "attention_map": attention_map  # base64 PNG
    }
```

### Step 4 — Frontend Integration

In the chatbot, detect if the question is image-related:

```typescript
// If user has uploaded an image AND asks image-specific question
// → call /medvqa endpoint instead of /chat
const isImageQuestion = (msg: string) => {
  const imageKeywords = [
    "this x-ray", "this image", "this scan", "visible",
    "show", "see", "region", "area", "where", "which part",
    "highlighted", "heatmap", "left lung", "right lung"
  ];
  return imageKeywords.some(kw => msg.toLowerCase().includes(kw));
};
```

---

## 9. Input/Output Summary

### What You Give MedVQA

| Input | Format | Example |
|---|---|---|
| Medical Image | JPEG/PNG (224×224 or 448×448) | Chest X-ray |
| Question | Plain text string | "Is there pleural effusion?" |

### What MedVQA Returns

| Output | Format | Example |
|---|---|---|
| Answer | Text string | "Yes, there is mild pleural effusion in the right costophrenic angle." |
| Confidence | Float 0-1 | 0.87 |
| Attention Map (optional) | base64 image | Region highlighted on X-ray |
| Answer Type | String | "open" / "closed" |

---

## 10. Question Categories in Medical VQA

| Category | Example Questions |
|---|---|
| **Abnormality** | "Is there any abnormality?" "What is the abnormality?" |
| **Presence** | "Is there pneumonia?" "Is consolidation present?" |
| **Location** | "Where is the opacity?" "Which lobe is affected?" |
| **Attribute** | "What size is the effusion?" "Is it bilateral?" |
| **Counting** | "How many lesions are visible?" |
| **Comparison** | "Which lung is more affected?" |
| **Other** | "What is the diagnosis?" "Is this X-ray normal?" |

---

## 11. Challenges in MedVQA

| Challenge | Description | Solution |
|---|---|---|
| **Small datasets** | VQA-RAD has only 315 images | Data augmentation + transfer learning |
| **Answer imbalance** | "Yes" answers dominate | Weighted loss, balanced sampling |
| **Medical complexity** | Requires expert knowledge | Use medically pre-trained models |
| **Evaluation** | Hard to auto-evaluate open answers | Use BERTScore + human evaluation |
| **Hallucination** | Model invents findings not in image | Grounding + attention regularization |
| **Multi-disease** | Image may show multiple conditions | Multi-label answer generation |

---

## 12. Recommended Papers to Read

1. **VQA-RAD** — "A Dataset of Radiology Questions" (Lau et al., 2018)
   - The foundational medical VQA dataset paper

2. **LLaVA-Med** — "LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine" (Microsoft, 2023)
   - Best open-source medical VLM, directly applicable

3. **MAML for MedVQA** — "Medical Visual Question Answering via Conditional Reasoning" (ACM MM 2020)
   - Good architecture reference

4. **BioViL** — "Making the Most of Text Semantics to Improve Biomedical Vision-Language Processing" (ECCV 2022)
   - Chest X-ray specific vision-language model

---

## 13. Summary

```
MedVQA = Vision Encoder + Text Encoder + Fusion + Answer Decoder

For PneumoScan specifically:
  - Image Encoder  : Our existing DenseNet121 (already trained on chest X-rays)
  - Text Encoder   : BioBERT or PubMedBERT
  - Fusion         : Cross-attention transformer
  - Answer Decoder : Classifier (closed) or LLaMA decoder (open-ended)
  - Dataset        : VQA-RAD + SLAKE
  - Integration    : New /medvqa FastAPI endpoint
  - Frontend       : Intelligent routing in ChatBot.tsx
```

The key insight: **MedVQA makes the chatbot actually see the image**, not just know the predicted labels. This is the difference between:
- Current: *"The model predicted Pneumonia with 94% confidence"*  
- With MedVQA: *"The consolidation is visible in the right lower lobe, with air bronchograms extending toward the hilum"*

---

*Document prepared for RMS-II implementation planning*
*PneumoScan Project — Nirma University, B.Tech CSE 2025–26*
