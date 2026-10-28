# PneumoScan — AI Chest X-Ray Analyzer

> DenseNet121 + Grad-CAM XAI • FastAPI Backend • Next.js 14 Frontend

---

## Project Structure

```
Lathiyo - Copy/
├── model/
│   └── final_model.pt          # Trained DenseNet121 weights
├── utils/
│   ├── model_loader.py         # Model loading logic
│   ├── preprocessing.py        # Image transforms + denormalize
│   └── gradcam.py              # Grad-CAM XAI implementation
├── backend/
│   ├── main.py                 # FastAPI app  (/health, /predict)
│   ├── requirements.txt        # Backend Python dependencies
│   └── start.bat               # One-click backend launcher (Windows)
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx      # Root layout + metadata
│   │   │   ├── page.tsx        # Home page (upload + results)
│   │   │   ├── about/page.tsx  # About page
│   │   │   └── gallery/page.tsx
│   │   └── components/
│   │       ├── Navbar.tsx
│   │       ├── Footer.tsx
│   │       ├── UploadSection.tsx   # Drag & drop upload
│   │       ├── LoadingModal.tsx    # Animated analysis overlay
│   │       └── ResultsSection.tsx  # Diagnosis + XAI images
│   ├── .env.local              # NEXT_PUBLIC_API_URL
│   ├── package.json
│   ├── tailwind.config.ts
│   └── tsconfig.json
└── app.py                      # Original Streamlit app (kept for reference)
```

---

## Running the App

### 1. Backend (FastAPI)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Or on Windows, double-click `backend/start.bat`.

API docs available at: http://localhost:8000/docs

---

### 2. Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev
```

Open: http://localhost:3000

---

## API Endpoints

| Method | Endpoint   | Description                        |
|--------|------------|------------------------------------|
| GET    | /health    | Check model load status            |
| POST   | /predict   | Upload image → get prediction + XAI|

### POST /predict

**Request:** `multipart/form-data` with field `file` (JPEG or PNG)

**Response:**
```json
{
  "prediction": "PNEUMONIA",
  "confidence": 94.2,
  "pneumonia_probability": 94.2,
  "original_image": "<base64 PNG>",
  "gradcam_heatmap": "<base64 PNG>",
  "overlay_image": "<base64 PNG>"
}
```

---

## Requirements

- Python 3.9+
- Node.js 18+
- CUDA GPU (optional, CPU inference works fine)

---

## Disclaimer

This application is for **research and educational purposes only**.  
It is not a certified medical diagnostic tool.  
Always consult a qualified radiologist for clinical decisions.
