# PneumoScan — AI Chest X-Ray Analyzer

> DenseNet121 Multi-Disease Detection + Grad-CAM XAI + AI Chat Assistant  
> FastAPI Backend · Next.js 14 Frontend

---

## What This Project Does

PneumoScan is a full-stack AI web application that:

- Accepts a chest X-ray image upload
- Runs a **DenseNet121** deep learning model to detect 11 chest conditions
- Generates **Grad-CAM heatmaps** to visually explain where the model is looking
- Provides an **AI chat assistant (PneumoBot)** powered by Groq LLM for follow-up questions

**Detected conditions:** Pneumonia · Atelectasis · Cardiomegaly · Pleural Effusion · Consolidation · Pulmonary Edema · Pneumothorax · Infiltration · Lung Lesion · Pleural Thickening · No Finding

---

## Project Structure

```
PneumoScan/
├── backend/
│   ├── main.py               # FastAPI app (all API endpoints)
│   ├── requirements.txt      # Python dependencies
│   ├── .env.example          # API key template  ← copy this to .env
│   └── start.bat             # One-click Windows launcher
├── frontend/
│   ├── src/
│   │   ├── app/              # Next.js pages (home, about, gallery)
│   │   └── components/       # UI components (Upload, Results, Chat...)
│   ├── .env.local.example    # Frontend env template  ← copy this to .env.local
│   └── package.json
├── model/
│   ├── final_model.pt        # Trained DenseNet121 weights  ← place here
│   └── medvqa_adapter/       # LoRA adapter for MedVQA (optional)
├── utils/
│   ├── model_loader.py
│   ├── preprocessing.py
│   ├── gradcam.py
│   └── medvqa_loader.py
└── README.md
```

---

## Prerequisites

Install these before you begin:

| Tool | Minimum Version | Download |
|------|----------------|----------|
| Python | 3.9+ | https://www.python.org/downloads/ |
| Node.js | 18+ | https://nodejs.org/ |

---

## Step-by-Step Setup & Run

### Step 1 — Get a Free Groq API Key

The AI chat feature requires a Groq API key (free, no credit card needed).

1. Go to → **https://console.groq.com/keys**
2. Sign up / log in
3. Click **"Create API Key"**
4. Copy the key — you will need it in Step 3

---

### Step 2 — Place the Model File

The trained model weights are too large for GitHub and are shared separately.

- Copy `final_model.pt` into the `model/` folder:

```
model/
└── final_model.pt   ✅ must be here
```

---

### Step 3 — Set Up the Backend

Open a terminal and run these commands one by one:

```bash
# 1. Go into the backend folder
cd backend

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 4. Install all Python dependencies
pip install -r requirements.txt
```

Now create the `.env` file with your API key:

```bash
# Windows:
copy .env.example .env

# macOS / Linux:
cp .env.example .env
```

Open `backend/.env` in any text editor. Change this line:

```
GROQ_API_KEY=your_groq_api_key_here
```

to your actual key:

```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx
```

Save and close the file.

---

### Step 4 — Start the Backend Server

Make sure you are inside `backend/` with the virtual environment still active, then run:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

> **Windows shortcut:** Double-click `backend/start.bat` instead.

You should see this output — the server is ready:

```
INFO:     Uvicorn running on http://0.0.0.0:8000
[INFO] Multiclass model loaded on cpu
[INFO] Groq API key found — chat endpoint ready
```

Confirm it works: open **http://localhost:8000/health** in your browser.  
You should see `"status": "ok"`.

---

### Step 5 — Set Up the Frontend

Open a **new terminal** (keep the backend running in the first one):

```bash
# 1. Go into the frontend folder
cd frontend

# 2. Create the environment file
# Windows:
copy .env.local.example .env.local
# macOS / Linux:
cp .env.local.example .env.local

# 3. Install Node.js dependencies
npm install

# 4. Start the frontend
npm run dev
```

You should see:

```
▲ Next.js 14.2.3
- Local:   http://localhost:3000
```

---

### Step 6 — Open the App

Open your browser and go to:

```
http://localhost:3000
```

The application is now fully running. Upload a chest X-ray to get started.

---

## Quick Reference — Running Every Time

You need **two terminals open** every time you run the project:

| Terminal | Folder | Command |
|----------|--------|---------|
| **1 — Backend** | `backend/` | activate venv → `uvicorn main:app --port 8000 --reload` |
| **2 — Frontend** | `frontend/` | `npm run dev` |

Then open **http://localhost:3000**

---

## Troubleshooting

**`ModuleNotFoundError` when starting backend**  
→ Virtual environment is not activated. Run `venv\Scripts\activate` (Windows) first.

**`GROQ_API_KEY is not configured` error**  
→ `backend/.env` is missing or has the placeholder key. Paste your real Groq key.

**Frontend shows "Failed to fetch" / network error**  
→ Backend is not running. Start it first (Step 4), then refresh the frontend.

**`model/final_model.pt not found` error**  
→ Place the model weights file at `model/final_model.pt` (see Step 2).

**Port already in use**  
→ Run backend on a different port: `uvicorn main:app --port 8001 --reload`  
→ Then update `frontend/.env.local`: `NEXT_PUBLIC_API_URL=http://localhost:8001`

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Check if model is loaded |
| POST | `/predict` | Upload X-ray → predictions + Grad-CAM |
| POST | `/chat` | AI chat assistant (PneumoBot) |
| GET | `/medvqa/status` | MedVQA model status |
| POST | `/medvqa` | Visual question answering on X-ray |

Interactive API docs: **http://localhost:8000/docs**

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | DenseNet121 (PyTorch) — trained on NIH ChestX-ray14 |
| XAI | Grad-CAM |
| Backend | FastAPI + Uvicorn |
| LLM Chat | Groq API |
| Frontend | Next.js 14, React 18, Tailwind CSS, Framer Motion |

---

## Disclaimer

This application is for **research and educational purposes only**.  
It is **not** a certified medical diagnostic tool.  
Always consult a qualified radiologist for any clinical decisions.
