# PneumoScan — Query Routing Documentation
### How every user question is classified, routed, and answered
*Parth Lathiya (23BCE215) | Dharm Patel (23BCE224) | Nirma University*

---

## Overview

PneumoScan has **two AI engines** answering questions:

| Engine | Model | Answers |
|---|---|---|
| **PneumoBot LLM** | Groq `qwen/qwen3.8-27b` | General medical knowledge |
| **MedVQA Vision Model** | LLaVA-Med + LoRA adapter | Image-specific visual questions |

Every question a user types goes through a **routing pipeline** that decides which engine answers it — and renders a different visual theme in the chat so the user can see which AI responded.

---

## Complete Routing Flow

```
User types a question in the chatbot
                │
                ▼
┌───────────────────────────────────────────────┐
│           FRONTEND: ChatContext.tsx            │
│                                               │
│  Step 1: Is scan image uploaded?              │
│    result?.original_image != null             │
│                │                              │
│       YES      │       NO                     │
│        │       │        │                     │
│        ▼       │        ▼                     │
│  Step 2: Is    │   Always → /chat (LLM)       │
│  question      │                              │
│  image-specific│                              │
│  (keyword      │                              │
│  detection)?   │                              │
│        │       │                              │
│   YES  │  NO   │                              │
│    │   │   │   │                              │
│    ▼   │   ▼   │                              │
│ /medvqa│ /chat │                              │
│  route │ route │                              │
└────────┼───────┼───────────────────────────────┘
         │       │
         │       └──────────────────────────┐
         │                                  │
         ▼                                  ▼
┌────────────────────┐          ┌────────────────────────┐
│  POST /medvqa      │          │  POST /chat             │
│  (FastAPI)         │          │  (FastAPI)              │
│                    │          │                         │
│  Receives:         │          │  Receives:              │
│  - X-ray image     │          │  - question text        │
│  - question text   │          │  - chat history         │
│                    │          │  - scan context         │
│  LLaVA-Med         │          │                         │
│  looks at image    │          │  classify_question()    │
│  pixel by pixel    │          │  ↓                      │
│  generates answer  │          │  mode = "medvqa"        │
│                    │          │       or "llm"          │
│  Returns:          │          │                         │
│  {                 │          │  build_system_prompt()  │
│    answer: "yes",  │          │  with correct persona   │
│    confidence: 0.94│          │                         │
│    answer_type:    │          │  Groq API call          │
│      "closed"      │          │                         │
│  }                 │          │  Returns:               │
│                    │          │  {                      │
│                    │          │    reply: "...",        │
│                    │          │    source: "medvqa"     │
│                    │          │           or "llm"      │
│                    │          │  }                      │
└─────────┬──────────┘          └───────────┬─────────────┘
          │                                 │
          └──────────────┬──────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │  FRONTEND: ChatBot.tsx │
            │                        │
            │  source = "medvqa"?    │
            │      ↓ YES             │
            │  Blue bubble           │
            │  "● MedVQA Vision      │
            │     Model" badge       │
            │                        │
            │  source = "llm"?       │
            │      ↓ YES             │
            │  Green bubble          │
            │  "● PneumoBot LLM"     │
            │     badge              │
            └────────────────────────┘
```

---

## Step 1 — Keyword Detection (Frontend)

**File:** `frontend/src/context/ChatContext.tsx`

The frontend first checks if the question contains **image-specific keywords**:

```typescript
const isImageQuestion = (q: string): boolean => {
  const imageKeywords = [
    "this x-ray", "this image", "this scan", "this chest x-ray",
    "in the image", "in the scan", "in the x-ray",
    "left lung", "right lung",
    "highlighted", "heatmap", "overlay",
    "describe this image", "explain this image",
    "what is shown in", "what abnormality is visible",
    "what do you see in", "can you see in",
    "does this x-ray show", "based on the image",
  ];
  return imageKeywords.some((kw) => q.toLowerCase().includes(kw));
};
```

### Routing Decision

```typescript
const shouldUseMedVQA =
  MEDVQA_READY &&                          // adapter is trained & loaded
  ctx !== null &&                           // scan has been uploaded
  currentResult?.original_image != null &&  // image data available
  isImageQuestion(trimmed);                 // question is image-specific
```

| Condition | Result |
|---|---|
| No scan uploaded | Always → `/chat` |
| Scan uploaded + general question | → `/chat` |
| Scan uploaded + image question + adapter ready | → `/medvqa` |

---

## Step 2A — MedVQA Route (`/medvqa`)

**When:** Scan uploaded + image-specific question + `MEDVQA_READY = true`

### What happens in the frontend

```typescript
// 1. Get the stored X-ray image (base64) from global context
const currentResult = resultRefFn();

// 2. Convert base64 → Blob (binary image data)
const byteStr = atob(currentResult.original_image);
const arr     = new Uint8Array(byteStr.length);
for (let i = 0; i < byteStr.length; i++) arr[i] = byteStr.charCodeAt(i);
const blob    = new Blob([arr], { type: "image/png" });

// 3. Build FormData with image + question
const formData = new FormData();
formData.append("file",          blob,    "xray.png");
formData.append("question",      trimmed);
formData.append("question_type", "auto");

// 4. Send to /medvqa endpoint
const res = await fetch(`${API_URL}/medvqa`, {
  method: "POST",
  body:   formData,
});
```

### What happens in the backend (`/medvqa`)

**File:** `backend/main.py` + `utils/medvqa_loader.py`

```
POST /medvqa receives:
  file     = chest X-ray image (binary)
  question = "Is there consolidation in this x-ray?"

Step 1: Decode image
  PIL.Image.open(bytes) → RGB image

Step 2: Load MedVQA model (singleton, loaded once at startup)
  MedVQAModel.is_loaded() → True
  (LLaVA-Med base + LoRA adapter already in memory)

Step 3: Build prompt
  "[INST] <image>
   You are a medical AI assistant specialized in chest X-ray analysis.
   Answer the following question based on the provided chest X-ray image.
   Be concise and clinically accurate.

   Question: Is there consolidation in this x-ray? [/INST] Answer:"

Step 4: Processor encodes image + text
  CLIP ViT-L/14 encodes image → 256 visual tokens
  Tokenizer encodes text → N text tokens
  Combined: [visual tokens] + [text tokens]

Step 5: LLaVA-Med generates answer
  model.generate(
    max_new_tokens = 64,
    do_sample      = False,  # greedy decoding
  )
  → generates "yes" token by token

Step 6: Decode answer
  new_tokens = output[prompt_len:]
  answer = tokenizer.decode(new_tokens) → "yes"

Step 7: Estimate confidence
  first_token_probs = softmax(scores[0])
  confidence = max(first_token_probs) → 0.94

Step 8: Return response
  {
    "question":    "Is there consolidation in this x-ray?",
    "answer":      "yes",
    "confidence":  0.94,
    "answer_type": "closed",
    "model_ready": true
  }
```

### What the frontend does with the response

```typescript
const replyMsg: Message = {
  id:        generateId(),
  role:      "assistant",
  content:   data.answer,           // "yes"
  timestamp: new Date(),
  source:    "medvqa",              // triggers blue bubble
};
setMessages((prev) => [...prev, replyMsg]);
```

### Visual output
```
┌──────────────────────────────────────┐
│ [VQA]  yes                           │  ← blue bubble
│                                      │
│  ● MedVQA Vision Model   03:46 PM   │  ← blue pulsing badge
└──────────────────────────────────────┘
```

---

## Step 2B — LLM Chat Route (`/chat`)

**When:** Any question that doesn't meet MedVQA conditions

### What happens in the frontend

```typescript
// Always goes through sendViaChat()
const history = messagesRef.current
  .filter((m) => m.id !== "welcome" && !m.error)
  .map((m) => ({ role: m.role, content: m.content }));

const res = await fetch(`${API_URL}/chat`, {
  method:  "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    message:             trimmed,
    history,                         // full conversation history
    primary_label:       ctx?.primary_label,
    primary_display:     ctx?.primary_display,
    primary_probability: ctx?.primary_probability,
    detected_diseases:   ctx?.detected_diseases,
    has_image:           ctx != null,
  }),
});
```

### What happens in the backend (`/chat`)

**File:** `backend/main.py`

```
POST /chat receives:
  message             = "What does this result mean?"
  primary_display     = "Consolidation"
  primary_probability = 72.0
  detected_diseases   = ["Consolidation", "Atelectasis"]
  has_image           = true

Step 1: classify_question(message)
  Checks message against IMAGE_KEYWORDS list
  "What does this result mean?" → no image keywords found
  → mode = "llm"

Step 2: build_system_prompt(req, mode="llm")
  Returns PneumoBot persona with scan context injected:

  "You are PneumoBot, an expert medical AI assistant...
   ---CURRENT SCAN CONTEXT---
   Primary finding  : Consolidation (72.0% probability)
   Detected findings: Consolidation, Atelectasis
   ---END CONTEXT---"

Step 3: Build Groq messages
  [
    {"role": "system",    "content": "<system prompt>"},
    {"role": "user",      "content": "previous question"},   ← history
    {"role": "assistant", "content": "previous answer"},     ← history
    {"role": "user",      "content": "What does this result mean?"},
  ]

Step 4: Groq API call
  model       = "qwen/qwen3.8-27b"
  temperature = 0.7
  max_tokens  = 1024

Step 5: Return response
  {
    "reply":  "Consolidation means the air spaces in the lung...",
    "source": "llm"
  }
```

### If the question IS image-specific (but still routed to /chat)

```
"What do you see in this x-ray?"

Step 1: classify_question()
  "this x-ray" detected → mode = "medvqa"

Step 2: build_system_prompt(req, mode="medvqa")
  Returns MedVQA-Bot persona:

  "You are MedVQA-Bot, a Medical Visual Question Answering AI model
   fine-tuned on chest X-ray images using LLaVA-Med architecture...
   You DIRECTLY ANALYSE the visual features of the uploaded chest X-ray
   Respond as if you are ACTUALLY LOOKING AT the X-ray image pixel by pixel
   ---CURRENT SCAN CONTEXT---
   Primary finding  : Consolidation (72.0% probability)
   ---END CONTEXT---"

Step 3: Groq answers with MedVQA tone
  "Examining the X-ray, I observe increased opacity in the right lower lobe..."

Step 4: Return response
  {
    "reply":  "Examining the X-ray, I observe...",
    "source": "medvqa"   ← even though Groq answered, source = medvqa
  }
```

### Visual output
```
LLM question:
┌──────────────────────────────────────────────┐
│ [BOT]  Consolidation means the air spaces... │  ← dark grey bubble
│                                               │
│  ● PneumoBot LLM                  03:46 PM  │  ← green badge
└──────────────────────────────────────────────┘

MedVQA-tone question (LLM simulating MedVQA):
┌──────────────────────────────────────────────┐
│ [VQA]  Examining the X-ray, I observe...     │  ← dark blue bubble
│                                               │
│  ● MedVQA Vision Model  ●  03:46 PM         │  ← blue pulsing badge
└──────────────────────────────────────────────┘
```

---

## Complete Decision Tree

```
User types message
        │
        ▼
Has scan been uploaded?
        │
   NO ──┴── YES
   │         │
   ▼         ▼
 /chat    isImageQuestion()?
 LLM         │
 mode    NO ─┴─ YES
             │       │
             ▼       ▼
           /chat   MEDVQA_READY?
           LLM         │
           mode    NO ─┴─ YES
                   │       │
                   ▼       ▼
                 /chat   /medvqa
                 mode=   LLaVA-Med
                 "medvqa" actual
                 (LLM     vision
                 simulates)model
                   │       │
                   └───┬───┘
                       │
                       ▼
                  source = "llm"
                       or
                  source = "medvqa"
                       │
                       ▼
              Frontend renders:
              "llm"    → green bubble
                          ● PneumoBot LLM
              "medvqa" → blue bubble
                          ● MedVQA Vision Model
```

---

## Routing Summary Table

| Scenario | Route | System Prompt | Source Tag | Bubble |
|---|---|---|---|---|
| No scan + any question | `/chat` | PneumoBot LLM | `llm` | 🟢 Green |
| Scan + general question | `/chat` | PneumoBot LLM | `llm` | 🟢 Green |
| Scan + image question + adapter ready | `/medvqa` | MedVQA-Bot (vision) | `medvqa` | 🔵 Blue |
| Scan + image question + adapter not ready | `/chat` | MedVQA-Bot (LLM simulates) | `medvqa` | 🔵 Blue |

---

## Image Keyword List (Full)

These keywords in a question trigger MedVQA routing:

```
"this x-ray"              "this image"
"this scan"               "this chest x-ray"
"in the image"            "in the scan"
"in the x-ray"            "left lung"
"right lung"              "highlighted"
"heatmap"                 "overlay"
"describe this image"     "explain this image"
"what is shown in"        "what abnormality is visible"
"what do you see in"      "can you see in"
"does this x-ray show"    "based on the image"
```

Any question NOT containing these keywords → PneumoBot LLM route.

---

## Scan Context Injection

When a scan result exists, it is automatically injected into EVERY chat prompt
(both LLM and MedVQA) so the model always knows the current findings:

```
--- CURRENT SCAN CONTEXT ---
Primary finding  : Consolidation (72.0% probability)
Detected findings: Consolidation, Atelectasis
--- END CONTEXT ---
```

This means even LLM answers are scan-aware — if you ask "what should I do next?"
the LLM knows you have Consolidation and Atelectasis detected.

---

## Chat History Persistence

The full conversation is stored in `ChatContext` (React global state in `layout.tsx`).
This means:
- Navigating to Gallery and back → same chat history preserved
- All messages included in the PDF report
- History sent with every new message so LLM has full conversation context
- `clearChat()` wipes everything when user resets the scan

---

## MedVQA Inference Detail (When Adapter is Active)

```
Input  : chest X-ray (224×224 RGB) + question text

Inside LLaVA-Med:

  Image → CLIP ViT-L/14
        → 256 visual tokens [256 × 1024]

  Visual tokens → MLP Projector (1024→4096)
               → 256 tokens in LLM space [256 × 4096]

  Question → Mistral tokenizer
           → N text tokens [N × 4096]

  Combined → [256 + N, 4096] sequence
           → Mistral-7B processes all tokens
           → Auto-regressively generates answer

  answer_type detection:
    question starts with "is/are/does/do/can/has/have" → "closed"
    otherwise → "open"

  Greedy decoding (do_sample=False):
    Pick highest probability token at each step
    Deterministic — same question always gives same answer

Output : answer text + confidence score
```

---

*ROUTING.md — PneumoScan Query Routing Documentation*
*Nirma University B.Tech CSE — RMS Project 2025-26*
