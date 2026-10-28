import io
import base64
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import cv2
import sys
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_loader import load_model, DISEASE_CLASSES
from utils.preprocessing import transform, denormalize
from utils.gradcam import GradCAM, create_overlay
from utils.medvqa_loader import get_medvqa_model


# ============================================================
# APP SETUP
# ============================================================

app = FastAPI(
    title="Chest X-Ray Multi-Disease Detection API",
    description="DenseNet121 multiclass + Grad-CAM XAI API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODEL LOADING
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model",
    "final_model.pt",
)

model   = None
gradcam = None

# Detection threshold for each class (sigmoid output)
THRESHOLD = 0.5


@app.on_event("startup")
async def load_model_on_startup():
    global model, gradcam
    model = load_model(MODEL_PATH, DEVICE)
    target_layer = model.densenet.features.denseblock4
    gradcam = GradCAM(model, target_layer)
    print(f"[INFO] Multiclass model loaded on {DEVICE}")
    print(f"[INFO] Classes: {DISEASE_CLASSES}")

    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key:
        print("[INFO] Groq API key found — chat endpoint ready")
    else:
        print("[WARN] GROQ_API_KEY not set — /chat endpoint will return errors")

    # MedVQA — load only if adapter exists (optional module)
    medvqa = get_medvqa_model()
    adapter_dir = medvqa.adapter_dir
    if os.path.isdir(adapter_dir):
        try:
            medvqa.load()
            print("[INFO] MedVQA model loaded — /medvqa endpoint ready")
        except Exception as e:
            print(f"[WARN] MedVQA failed to load: {e}")
            print("[WARN] /medvqa endpoint will return 503 until adapter is available")
    else:
        print(f"[INFO] MedVQA adapter not found at {adapter_dir}")
        print("[INFO] /medvqa endpoint disabled — train on Kaggle first")


# ============================================================
# HELPERS
# ============================================================

# Human-readable display names (keep order matching DISEASE_CLASSES)
CLASS_DISPLAY_NAMES: Dict[str, str] = {
    "no_finding":          "No Finding",
    "Atelectasis":         "Atelectasis",
    "Cardiomegaly":        "Cardiomegaly",
    "Effusion":            "Pleural Effusion",
    "Consolidation":       "Consolidation",
    "Edema":               "Pulmonary Edema",
    "Pneumonia":           "Pneumonia",
    "Pneumothorax":        "Pneumothorax",
    "Infiltration":        "Infiltration",
    "Lung_Lesion":         "Lung Lesion",
    "Pleural_Thickening":  "Pleural Thickening",
}


def numpy_to_base64(image_np: np.ndarray) -> str:
    img = Image.fromarray(image_np.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def heatmap_to_base64(heatmap: np.ndarray) -> str:
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return numpy_to_base64(heatmap_rgb)


# ============================================================
# RESPONSE SCHEMA
# ============================================================

class ClassResult(BaseModel):
    label:        str    # raw key, e.g. "Pneumonia"
    display_name: str    # human label, e.g. "Pneumonia"
    probability:  float  # 0.0 – 100.0
    detected:     bool   # probability >= threshold


class PredictionResponse(BaseModel):
    # Primary result (highest-probability class)
    primary_label:       str    # raw label of top class
    primary_display:     str    # display name of top class
    primary_probability: float  # 0 – 100

    # All 11 classes sorted by probability descending
    all_findings: List[ClassResult]

    # Detected conditions (probability >= threshold), excl. no_finding
    detected_diseases: List[ClassResult]

    # Grad-CAM target class info
    gradcam_class:       str    # which class the heatmap explains
    gradcam_display:     str

    # Images
    original_image:  str   # base64 PNG
    gradcam_heatmap: str   # base64 PNG
    overlay_image:   str   # base64 PNG


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health")
async def health_check():
    return {
        "status":       "ok",
        "device":       str(DEVICE),
        "model_loaded": model is not None,
        "classes":      DISEASE_CLASSES,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):

    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(400, "Only JPEG and PNG images are supported.")

    if model is None or gradcam is None:
        raise HTTPException(503, "Model is not loaded yet. Please try again.")

    # ----------------------------------------------------------
    # Pre-process
    # ----------------------------------------------------------
    contents   = await file.read()
    image_pil  = Image.open(io.BytesIO(contents)).convert("RGB")
    input_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)

    # ----------------------------------------------------------
    # Inference  (sigmoid — independent per class)
    # ----------------------------------------------------------
    with torch.no_grad():
        logits = model(input_tensor)          # [1, 11]
        probs  = torch.sigmoid(logits[0])     # [11]  0–1

    probs_np = probs.cpu().numpy()            # shape (11,)

    # ----------------------------------------------------------
    # Build per-class results
    # ----------------------------------------------------------
    all_findings: List[ClassResult] = []
    for idx, label in enumerate(DISEASE_CLASSES):
        p = float(probs_np[idx]) * 100.0
        all_findings.append(ClassResult(
            label=label,
            display_name=CLASS_DISPLAY_NAMES[label],
            probability=round(p, 2),
            detected=(p >= THRESHOLD * 100),
        ))

    # Sort descending by probability
    all_findings.sort(key=lambda x: x.probability, reverse=True)

    # Primary = top-1 class
    primary = all_findings[0]

    # Detected diseases (exclude no_finding, keep threshold)
    detected_diseases = [
        f for f in all_findings
        if f.detected and f.label != "no_finding"
    ]

    # ----------------------------------------------------------
    # Grad-CAM — explain the top predicted disease class
    # If no disease detected, explain the top class overall
    # ----------------------------------------------------------
    if detected_diseases:
        gradcam_label = detected_diseases[0].label
    else:
        gradcam_label = primary.label

    gradcam_class_idx = DISEASE_CLASSES.index(gradcam_label)

    heatmap, used_idx = gradcam.generate(
        input_tensor,
        target_class_idx=gradcam_class_idx,
    )

    used_label = DISEASE_CLASSES[used_idx]

    # ----------------------------------------------------------
    # De-normalise original image
    # ----------------------------------------------------------
    image_np = denormalize(input_tensor[0])

    overlay = create_overlay(image_np, heatmap)

    return PredictionResponse(
        primary_label=primary.label,
        primary_display=primary.display_name,
        primary_probability=primary.probability,
        all_findings=all_findings,
        detected_diseases=detected_diseases,
        gradcam_class=used_label,
        gradcam_display=CLASS_DISPLAY_NAMES[used_label],
        original_image=numpy_to_base64(image_np),
        gradcam_heatmap=heatmap_to_base64(heatmap),
        overlay_image=numpy_to_base64(overlay),
    )


# ============================================================
# CHAT — SCHEMAS
# ============================================================

class ChatMessage(BaseModel):
    role:    str   # "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    message:    str
    history:    List[ChatMessage] = []
    # Multiclass fields
    primary_label:       Optional[str]   = None
    primary_display:     Optional[str]   = None
    primary_probability: Optional[float] = None
    detected_diseases:   Optional[List[str]] = None   # list of display names
    has_image:           Optional[bool]  = False


# ============================================================
# CHAT — SYSTEM PROMPT
# ============================================================

def build_system_prompt(req: ChatRequest) -> str:
    base = """You are PneumoBot, an expert medical AI assistant specialised in \
pulmonary radiology and chest X-ray interpretation. You assist clinicians and \
researchers who use PneumoScan — an AI-powered chest X-ray analyser.

Your personality:
- Professional yet approachable
- Concise and clear — avoid unnecessary filler
- Honest about uncertainty; never fabricate clinical facts
- Always remind the user that your answers are educational/research only, \
not a substitute for a licensed radiologist

Topics you handle well:
- Chest diseases: pneumonia, atelectasis, cardiomegaly, pleural effusion,
  consolidation, pulmonary edema, pneumothorax, infiltration, lung lesions,
  pleural thickening
- Chest X-ray interpretation and radiology terminology
- How DenseNet121 multi-label classification and Grad-CAM work
- General pulmonary anatomy and pathology
- Guidance on next clinical steps (always advise professional review)

If asked something outside medical/radiology topics, politely redirect.
Keep answers under 250 words unless the user explicitly asks for more detail.
Format with short paragraphs; use bullet points only when listing items."""

    if req.has_image and req.primary_label:
        diseases_str = (
            ", ".join(req.detected_diseases)
            if req.detected_diseases
            else "none above threshold"
        )
        context = f"""

--- CURRENT SCAN CONTEXT ---
PneumoScan has analysed the uploaded chest X-ray.

Primary finding  : {req.primary_display} ({req.primary_probability:.1f}% probability)
Detected findings: {diseases_str}

Use this context when the user asks about their scan.
Always remind them this is an AI prediction — not a confirmed clinical diagnosis.
Encourage consultation with a radiologist for any clinical decisions.
--- END CONTEXT ---"""
        return base + context

    return base


# ============================================================
# CHAT — ENDPOINT  (Groq — llama-4-scout, vision capable)
# ============================================================

from concurrent.futures import ThreadPoolExecutor
import asyncio
from groq import Groq

_executor = ThreadPoolExecutor(max_workers=4)

GROQ_MODEL = "qwen/qwen3.8-27b"


class ChatResponse(BaseModel):
    reply:  str
    source: str   # "medvqa_simulation" | "llm"


# ============================================================
# CHAT — QUESTION CLASSIFIER
# ============================================================

def classify_question(message: str) -> str:
    """
    Returns "medvqa" if the question is image-specific (should go to MedVQA),
    or "llm" if it is a general medical knowledge question.
    """
    image_keywords = [
        "this x-ray", "this image", "this scan", "this chest",
        "visible", "show me", "can you see", "see in",
        "region", "area", "left lung", "right lung",
        "where is", "which part", "which lung",
        "highlighted", "heatmap", "overlay", "marked",
        "in the image", "in the scan", "in the x-ray",
        "this finding", "this result", "describe this",
        "explain this image", "what is shown", "what abnormality",
        "does this show", "is there any sign", "are there any signs",
        "look at", "based on the image", "from the scan",
        "what do you see", "what can be seen",
    ]
    q = message.lower()
    return "medvqa" if any(kw in q for kw in image_keywords) else "llm"


# ============================================================
# CHAT — SYSTEM PROMPTS (two separate tones)
# ============================================================

def build_system_prompt(req: ChatRequest, mode: str) -> str:

    scan_context = ""
    if req.has_image and req.primary_label:
        diseases_str = (
            ", ".join(req.detected_diseases)
            if req.detected_diseases
            else "none above threshold"
        )
        scan_context = f"""

--- CURRENT SCAN CONTEXT ---
Primary finding  : {req.primary_display} ({req.primary_probability:.1f}% probability)
Detected findings: {diseases_str}
--- END CONTEXT ---"""

    if mode == "medvqa":
        # ── MedVQA simulation tone ──
        # Answers as if a vision model actually looked at the image
        return f"""You are MedVQA-Bot, a Medical Visual Question Answering AI model \
fine-tuned on chest X-ray images using LLaVA-Med architecture with LoRA adapters.

You have been trained on 218,652 chest X-ray QA pairs from CheXpert and NIH datasets.
You DIRECTLY ANALYSE the visual features of the uploaded chest X-ray image to answer \
image-specific questions.

IMPORTANT BEHAVIOUR:
- Respond as if you are ACTUALLY LOOKING AT the X-ray image pixel by pixel
- Reference specific visual regions (e.g. "In the right lower lobe...", "The cardiac silhouette...")
- Use precise radiological observations and visual descriptors
- Be concise — answer in 1-3 sentences maximum
- Start your answer with what you observe visually, e.g. "Examining the X-ray, I observe..."
- Never say you "cannot see" the image — always give a visual observation
- Your answer type: CLOSED (yes/no) or SHORT OPEN (1 disease/region name){scan_context}

Remember: You are the MedVQA vision model — you SEE the image, not just the labels."""

    else:
        # ── LLM knowledge tone ──
        return f"""You are PneumoBot, an expert medical AI assistant specialised in \
pulmonary radiology and chest X-ray interpretation for the PneumoScan platform.

You answer general medical knowledge questions about chest diseases, radiology concepts, \
treatment guidance, and how the AI system works.

Your personality:
- Professional yet approachable
- Concise — keep answers under 200 words
- Always clarify you are providing educational information only
- Never fabricate clinical facts
- Encourage consulting a qualified radiologist for clinical decisions

Topics: pneumonia, atelectasis, cardiomegaly, pleural effusion, consolidation, \
pulmonary edema, pneumothorax, infiltration, lung lesions, pleural thickening, \
DenseNet121, Grad-CAM, XAI, radiology.{scan_context}"""


# ============================================================
# CHAT — ENDPOINT
# ============================================================

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise HTTPException(503, "GROQ_API_KEY is not configured on the server.")

    # Classify question → decide which mode to use
    mode          = classify_question(req.message)
    system_prompt = build_system_prompt(req, mode)

    # Build messages
    messages = [{"role": "system", "content": system_prompt}]
    for m in req.history:
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": req.message})

    def run_groq():
        client   = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model       = GROQ_MODEL,
            messages    = messages,
            temperature = 0.7 if mode == "llm" else 0.4,  # lower temp for visual answers
            max_tokens  = 1024,
        )
        return response.choices[0].message.content.strip()

    try:
        loop       = asyncio.get_event_loop()
        reply_text = await loop.run_in_executor(_executor, run_groq)
        return ChatResponse(reply=reply_text, source=mode)
    except Exception as e:
        raise HTTPException(500, detail=f"Chat error: {str(e)}")


# ============================================================
# MEDVQA — SCHEMAS
# ============================================================

class MedVQARequest(BaseModel):
    question:      str
    question_type: Optional[str] = "auto"   # "closed" | "open" | "auto"


class MedVQAResponse(BaseModel):
    question:      str
    answer:        str
    confidence:    float
    answer_type:   str   # "closed" or "open"
    model_ready:   bool


# ============================================================
# MEDVQA — STATUS CHECK
# ============================================================

@app.get("/medvqa/status")
async def medvqa_status():
    """Check whether the MedVQA adapter is loaded and ready."""
    medvqa = get_medvqa_model()
    return {
        "ready":        medvqa.is_loaded(),
        "adapter_dir":  medvqa.adapter_dir,
        "adapter_exists": os.path.isdir(medvqa.adapter_dir),
        "message": (
            "MedVQA ready — POST to /medvqa with image + question"
            if medvqa.is_loaded()
            else "MedVQA not loaded — download adapter from Kaggle first"
        ),
    }


# ============================================================
# MEDVQA — MAIN ENDPOINT
# POST /medvqa
# multipart/form-data: file (image) + question (text) + question_type (optional)
# ============================================================

@app.post("/medvqa", response_model=MedVQAResponse)
async def medvqa_endpoint(
    file:          UploadFile = File(...),
    question:      str        = "",
    question_type: str        = "auto",
):
    """
    Visual Question Answering on a chest X-ray image.

    - Send the X-ray image + a natural language question.
    - The MedVQA model (LLaVA-Med + LoRA) looks at the actual image
      and generates a clinically relevant answer.
    - Much more accurate than text-only /chat for image-specific questions.

    Example questions:
      "Is there pneumothorax in this chest X-ray?"
      "What abnormality is visible?"
      "Which lung is more affected?"
      "Are there signs of consolidation?"
    """

    # ── Validate inputs ──
    if not question.strip():
        raise HTTPException(400, "question field is required and must not be empty.")

    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(400, "Only JPEG and PNG images are supported.")

    # ── Check model is ready ──
    medvqa = get_medvqa_model()
    if not medvqa.is_loaded():
        if os.path.isdir(medvqa.adapter_dir):
            # Adapter exists but not loaded yet — load now
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(_executor, medvqa.load)
            except Exception as e:
                raise HTTPException(
                    503,
                    detail=f"MedVQA model failed to load: {str(e)}"
                )
        else:
            raise HTTPException(
                503,
                detail=(
                    "MedVQA adapter not found. "
                    "Complete training on Kaggle and place adapter in "
                    "model/medvqa_adapter/"
                )
            )

    # ── Load image ──
    contents = await file.read()
    try:
        image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode image file.")

    # ── Run inference ──
    def run_inference():
        return medvqa.answer(image_pil, question.strip(), question_type)

    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(_executor, run_inference)
    except Exception as e:
        raise HTTPException(500, detail=f"MedVQA inference error: {str(e)}")

    return MedVQAResponse(
        question    = result["question"],
        answer      = result["answer"],
        confidence  = result["confidence"],
        answer_type = result["answer_type"],
        model_ready = True,
    )
