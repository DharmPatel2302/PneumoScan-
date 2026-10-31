# PneumoScan — Chatbot, Chat Storage & Clinical Report
### How the AI assistant works, how chats are stored, and how the PDF report is generated for doctors
*Parth Lathiya (23BCE215) | Dharm Patel (23BCE224) | Nirma University*

---

# PART 1 — The Chatbot

## 1.1 What is PneumoBot?

PneumoBot is the AI assistant embedded in PneumoScan. It runs on every page of the
application and serves two roles depending on what the user asks:

| Role | Name | Engine | When |
|---|---|---|---|
| General medical assistant | PneumoBot LLM | Groq `qwen/qwen3.8-27b` | General questions |
| Visual X-ray analyser | MedVQA Vision Model | LLaVA-Med + LoRA | Image-specific questions |

Both roles share the same chat window — the user types in one place and the system
automatically decides which engine should answer, showing a different coloured bubble
so the user always knows which AI responded.

---

## 1.2 The Two Response Themes

### 🟢 PneumoBot LLM (Green Bubble)

```
┌──────────────────────────────────────────────────────┐
│  [BOT]  Consolidation means the air spaces in the    │
│         lung are filled with fluid or inflammatory   │
│         material instead of air. It appears as an    │
│         opaque region on the X-ray.                  │
│                                                      │
│  ● PneumoBot LLM                        03:46 PM    │
└──────────────────────────────────────────────────────┘
```

- Dark grey background (`#252525`)
- Green `● PneumoBot LLM` badge
- Answers from LLM general medical knowledge
- Has full scan context injected into system prompt

### 🔵 MedVQA Vision Model (Blue Bubble)

```
┌──────────────────────────────────────────────────────┐
│  [VQA]  Examining the X-ray, I observe increased     │
│         opacity in the right lower lobe consistent   │
│         with consolidation. The left lung appears    │
│         relatively clear.                            │
│                                                      │
│  ● MedVQA Vision Model  ●              03:47 PM    │
└──────────────────────────────────────────────────────┘
```

- Dark blue background (`#0f1e35`) with blue border
- Blue pulsing `● MedVQA Vision Model` badge
- Answers as if visually examining the X-ray pixel by pixel
- Uses LLaVA-Med architecture (CLIP ViT + Mistral-7B)

---

# PART 2 — How Chat is Stored

## 2.1 Storage Location

The entire chat history is stored in **React global state** (not a database, not localStorage).
It lives in `ChatContext` which is mounted in `layout.tsx` — the root layout that wraps
every page of the application.

```
layout.tsx
  └── ChatProvider          ← chat state lives here (RAM / React state)
        ├── /home page
        ├── /about page
        └── /gallery page
              └── ChatBot component
```

Because `ChatProvider` is at the root layout level, it is **never unmounted** during
navigation — the chat survives when the user moves between pages.

## 2.2 Message Structure

Every message stored has this shape:

```typescript
interface Message {
  id:        string;                              // unique random ID
  role:      "user" | "assistant";               // who sent it
  content:   string;                             // message text (markdown)
  timestamp: Date;                               // when it was sent
  error?:    boolean;                            // true if API call failed
  source?:   "llm" | "medvqa_simulation" | "medvqa";  // which AI answered
}
```

The `source` field is what determines the bubble colour:
- `"llm"` → green bubble (`● PneumoBot LLM`)
- `"medvqa"` or `"medvqa_simulation"` → blue bubble (`● MedVQA Vision Model`)
- `undefined` (welcome message) → default style

## 2.3 Full State Stored in Context

```typescript
// Everything that persists across page navigation:

messages:           Message[]         // every chat message ever sent this session
open:               boolean           // is chat window open
minimised:          boolean           // is chat minimised
unread:             number            // unread message count (for red badge)
loading:            boolean           // waiting for API response
predictionContext:  PredictionContext // scan result summary (for system prompt)
lastReply:          string            // latest reply (for notification toast)
result:             PredictionResult  // FULL scan result (all 11 classes + images)
apiError:           string            // upload error if any
```

## 2.4 Chat Lifecycle

```
User opens app
        │
        ▼
Welcome message added to messages[]
(role: "assistant", id: "welcome", source: undefined)

User uploads X-ray
        │
        ▼
setPredictionContext() called
Welcome message updated to include scan findings
predictionContext stored in context

User types question
        │
        ▼
User message added to messages[]
  { id, role: "user", content: "question", timestamp }

API call made (/chat or /medvqa)
        │
        ▼
Assistant reply added to messages[]
  { id, role: "assistant", content: "reply",
    timestamp, source: "llm" or "medvqa" }

User navigates to Gallery page
        │
        ▼
ChatProvider stays mounted → messages[] unchanged
ChatBot still visible (rendered in layout.tsx)

User comes back to Home page
        │
        ▼
messages[] still intact → full chat shown
result still in context → ResultsSection shown again

User clicks "Analyze Another X-ray"
        │
        ▼
clearChat() called:
  messages = [new welcome message]
  result = null
  predictionContext = null
  unread = 0
```

## 2.5 Chat History Sent to Backend

With every new message, the full conversation history is sent to the backend
so the LLM has context of the entire conversation:

```json
{
  "message": "Is this serious?",
  "history": [
    { "role": "user",      "content": "What does this result mean?" },
    { "role": "assistant", "content": "Consolidation means..." },
    { "role": "user",      "content": "How is it treated?" },
    { "role": "assistant", "content": "Treatment involves..." }
  ],
  "primary_label":       "Consolidation",
  "primary_display":     "Consolidation",
  "primary_probability": 72.0,
  "detected_diseases":   ["Consolidation", "Atelectasis"],
  "has_image":           true
}
```

This means the chatbot remembers everything said earlier and gives contextual answers.

## 2.6 What is NOT Stored

- Nothing is written to disk or a database
- Closing the browser tab clears everything
- Refreshing the page clears everything
- The chat is session-only (by design — medical data should not persist)

---

# PART 3 — The Clinical PDF Report

## 3.1 Purpose of the Report

The PDF report is designed specifically for **doctors and radiologists**. It collects:
- The original chest X-ray
- The AI prediction with confidence scores
- The Grad-CAM XAI heatmaps
- The full conversation with PneumoBot (both LLM and MedVQA answers)
- Clinical recommendations

The doctor can use this report to:
1. Quickly understand what the AI found in the X-ray
2. See exactly which region the model focused on (from the heatmap)
3. Read the patient's questions and AI explanations
4. Use the clinical recommendations as a starting point for treatment discussion
5. Explain the diagnosis to the patient using the visual heatmaps

## 3.2 How to Generate the Report

On the results page, after the scan is analysed:

```
┌─────────────────────────────────────────────────────┐
│  [Download Clinical Report]  [Analyze Another X-ray] │
└─────────────────────────────────────────────────────┘
```

Click **"Download Clinical Report"** → PDF is generated instantly in the browser
using `jsPDF` (no server needed) → saved as `PneumoScan_Report_RPT-XXXXXX.pdf`

## 3.3 What Triggers the Report

```typescript
// ResultsSection.tsx
const handleDownloadReport = async () => {
  setGenerating(true);
  await generateClinicalReport({
    prediction:            result.primary_display.toUpperCase(),
    confidence:            result.primary_probability,
    pneumonia_probability: result.all_findings.find(f => f.label === "Pneumonia")?.probability ?? 0,
    original_image:        result.original_image,    // base64 PNG
    gradcam_heatmap:       result.gradcam_heatmap,   // base64 PNG
    overlay_image:         result.overlay_image,     // base64 PNG
    chatHistory:           chatHistory,              // full conversation
  });
};
```

Everything needed is already in the global context — no additional API call is made.
The PDF is built entirely client-side.

## 3.4 Complete Report Structure

```
┌─────────────────────────────────────────────────────────┐
│  PneumoScan Clinical AI              CLINICAL RADIOLOGY  │  ← Header bar (dark)
│                                      REPORT             │
│                                      Generated: Oct 2026│
├─────────────────────────────────────────────────────────┤
│  Report ID: RPT-M3X9K2A1   System: DenseNet121+GradCAM  │
├─────────────────────────────────────────────────────────┤
│  ■ 01  DIAGNOSIS RESULT                                  │
│                                                         │
│  [ CONSOLIDATION ]              72.0%                   │  ← badge + big %
│  ████████████████░░░░           CONFIDENCE SCORE        │  ← confidence bar
│                                                         │
│  The AI classifier detected radiological features...    │
├─────────────────────────────────────────────────────────┤
│  ■ 02  IMAGING ANALYSIS — GRAD-CAM XAI                  │
│                                                         │
│  ORIGINAL X-RAY    GRAD-CAM HEATMAP    XAI OVERLAY      │
│  ┌───────────┐     ┌───────────┐     ┌───────────┐      │
│  │           │     │           │     │           │      │
│  │  X-ray    │     │  Heatmap  │     │  Overlay  │      │
│  │  image    │     │  (JET)    │     │  blended  │      │
│  │           │     │           │     │           │      │
│  └───────────┘     └───────────┘     └───────────┘      │
│  Target specimen   Model attention   AI identifying     │
│  for analysis.     heatmap.          visual features.   │
├─────────────────────────────────────────────────────────┤
│  ■ 03  CLINICAL RECOMMENDATIONS                          │
│                                                         │
│  ● Consult a licensed pulmonologist immediately...      │
│  ● Sputum culture or CBC/CRP blood test recommended...  │
│  ● Antibiotic therapy under medical supervision...      │
│  ● Adequate hydration, rest and fever management...     │
│  ● Follow-up chest X-ray in 4–6 weeks...               │
│  ● Avoid smoking and pollutant exposure...              │
│  ● Hospitalization may be required for severe cases...  │
├─────────────────────────────────────────────────────────┤
│  ■ 04  PNEUMOBOT CHAT SUMMARY                           │
│  6 message(s) exchanged with the medical AI assistant.  │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │ Patient / User (blue bg)                        │    │
│  │ What does this result mean?                     │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │ PneumoBot AI (green bg)                         │    │
│  │ Consolidation means the air spaces in the lung  │    │
│  │ are filled with fluid...                        │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │ Patient / User (blue bg)                        │    │
│  │ What do you see in this x-ray?                  │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │ PneumoBot AI (green bg)                         │    │
│  │ Examining the X-ray, I observe increased        │    │
│  │ opacity in the right lower lobe...              │    │
│  └─────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────┤
│  ■ 05  DISCLAIMER                                        │
│  ⚠ This report is generated by an AI-assisted research  │
│    tool for educational purposes only. NOT a certified  │
│    medical diagnosis. Consult a qualified radiologist.  │
├─────────────────────────────────────────────────────────┤
│  PneumoScan Clinical AI • For Research Use Only         │  ← Footer every page
│  Report ID: RPT-M3X9K2A1              Page 1 of 2      │
└─────────────────────────────────────────────────────────┘
```

## 3.5 Section-by-Section Breakdown

### Section 01 — Diagnosis Result

| Element | Content |
|---|---|
| Coloured badge | Primary finding name (red if disease, green if normal) |
| Large percentage | Primary finding probability (e.g. 72.0%) |
| Confidence bar | Visual bar filled to the confidence percentage |
| Summary text | Plain English description of what AI found |

### Section 02 — Imaging Analysis (Grad-CAM XAI)

Three images placed side by side:

| Image | What it shows | Why it helps the doctor |
|---|---|---|
| Original X-Ray | De-normalised input image | Reference — what was actually analysed |
| Grad-CAM Heatmap | JET colourmap attention map | Red = where model focused most |
| XAI Overlay | X-ray + heatmap blended (45% alpha) | Localises abnormality on actual anatomy |

The doctor can point to the red region on the overlay and explain to the patient
exactly which part of the lung is affected.

### Section 03 — Clinical Recommendations

Auto-generated based on the primary finding.

**If disease detected:**
- Consult pulmonologist / physician immediately
- Sputum culture or CBC/CRP blood test
- Antibiotic therapy under medical supervision
- Hydration, rest, fever management
- Follow-up X-ray in 4–6 weeks
- Avoid smoking and pollutants
- Hospitalisation assessment for severe cases

**If no finding (normal):**
- No immediate intervention needed
- Routine preventive care and annual check-ups
- Maintain healthy lifestyle for lung health
- Seek evaluation if symptoms develop
- Pneumococcal and influenza vaccinations

### Section 04 — PneumoBot Chat Summary

Every message exchanged during the session is included — both LLM answers
and MedVQA answers are shown, all attributed correctly.

```
Patient / User   (blue background)  — what the patient or doctor asked
PneumoBot AI     (green background) — what the AI answered
```

All markdown is stripped before rendering in PDF (bold, bullets, headers
converted to clean plain text).

**Why this is valuable for doctors:**
- Doctor can see what questions were asked before the appointment
- Can review AI explanations of the findings
- MedVQA visual observations help confirm or question the AI analysis
- Serves as a conversation starter for explaining diagnosis to patient

### Section 05 — Disclaimer

Legal and ethical notice that the report is for educational/research use only
and is not a certified clinical diagnosis.

## 3.6 Report Generation — Technical Detail

```
User clicks "Download Clinical Report"
        │
        ▼
generateClinicalReport(data) called
(utils/generateReport.ts)
        │
        ▼
new jsPDF({ format: "a4", unit: "mm" })
Page = 210mm × 297mm, Margin = 18mm
        │
        ▼
Page 1 built:
  Header bar (dark rectangle)
  "PneumoScan" logo in white/green
  Report ID generated: RPT-{timestamp.base36.toUpperCase()}
  Timestamp in locale format
        │
        ▼
Section 01 — Diagnosis
  roundedRect() for badge
  addText() for confidence number
  roundedRect() for confidence bar (two layers: bg + filled)
        │
        ▼
Section 02 — XAI Images
  Check if images fit on current page
  If not → addPage() + drawWatermark()
  addImage(base64PNG) × 3
  Labels above, captions below each image
        │
        ▼
Section 03 — Recommendations
  Loop through recommendation strings
  Green filled circle as bullet point
  drawWrappedText() handles line wrapping + page breaks
        │
        ▼
Section 04 — Chat Summary (if messages > 0)
  Filter: exclude system messages + empty content
  stripMarkdown() on each message content
  For each message:
    roundedRect() coloured background
    Bold role label (Patient / User or PneumoBot AI)
    Wrapped message text
    Auto page break if bubble doesn't fit
        │
        ▼
Section 05 — Disclaimer
  Amber outlined rounded rectangle
  Warning text wrapped inside
        │
        ▼
Footer on every page:
  Thin horizontal rule
  "PneumoScan Clinical AI • For Research Use Only"
  "Page X of Y"  (centre)
  "Report ID: RPT-XXXXXX"  (right)
        │
        ▼
doc.save("PneumoScan_Report_RPT-XXXXXX.pdf")
Browser downloads file automatically
```

## 3.7 Watermark

Every page has a faint diagonal watermark:

```
Opacity: 4% (barely visible)
Text: "PneumoScan"
Font: Helvetica Bold, 72pt
Colour: Green #22c55e
Angle: 45 degrees, centred on page
```

This makes the document look professional and prevents misuse as an official
medical certificate.

## 3.8 Report File Naming

```
PneumoScan_Report_RPT-M3X9K2A1.pdf
                   └──────────┘
                   timestamp in base36
                   (unique per report)
```

---

# PART 4 — How the Report Helps Doctors

## 4.1 Doctor's Workflow with PneumoScan

```
1. Patient presents with respiratory symptoms

2. Chest X-ray taken and uploaded to PneumoScan

3. AI analyses in ~5-10 seconds:
   → 11-class prediction with probabilities
   → Grad-CAM heatmap generated
   → XAI overlay created

4. Doctor or patient asks PneumoBot questions:
   "What do you see in this x-ray?"   → MedVQA visual answer
   "How serious is consolidation?"    → LLM medical knowledge
   "What treatment is recommended?"   → LLM clinical guidance

5. Doctor clicks "Download Clinical Report"
   → Full PDF generated with everything

6. Doctor uses report in consultation:
   ┌──────────────────────────────────────────────┐
   │  "Looking at this X-ray, the AI detected     │
   │   consolidation in the right lower lobe.     │
   │   You can see the red region in the heatmap  │
   │   — that is where the infection is located." │
   └──────────────────────────────────────────────┘

7. Report filed with patient records
```

## 4.2 Value of Each Report Section for Doctors

| Section | Doctor uses it to... |
|---|---|
| Diagnosis Result | Quick reference — primary finding + confidence |
| Confidence bar | Assess certainty of AI prediction |
| Original X-ray | Compare with their own visual assessment |
| Grad-CAM Heatmap | See exactly where model detected abnormality |
| XAI Overlay | Point to specific region when talking to patient |
| Clinical Recommendations | Starting point for treatment planning |
| Chat Summary (LLM) | Review AI medical explanations |
| Chat Summary (MedVQA) | Review AI visual observations of the X-ray |

## 4.3 Patient Understanding

The XAI overlay is especially valuable for patient communication:

```
Without PneumoScan:
  Doctor: "You have consolidation in the right lower lobe."
  Patient: "What does that mean? Where is it?"
  Doctor: "It means fluid in this part of the lung..." (points at plain X-ray)

With PneumoScan + Report:
  Doctor shows the XAI overlay image
  "See this red/orange region here? This is where the infection is.
   The AI model highlighted this exact area because it shows the
   characteristic pattern of consolidation."
  Patient: "Oh, I can see it now!"
```

---

# PART 5 — Summary

```
User types question
        │
        ├── Image question? → MedVQA (blue bubble, source: "medvqa")
        └── General question? → LLM (green bubble, source: "llm")
                │
                ▼
        All messages stored in React Context
        (survives page navigation, session-only)
                │
                ▼
        Chat history included in every API call
        (LLM has full conversation context)
                │
                ▼
        "Download Clinical Report" clicked
                │
                ▼
        jsPDF builds PDF entirely in browser:
          01 Diagnosis Result (badge + confidence bar)
          02 Imaging Analysis (Original + Heatmap + Overlay)
          03 Clinical Recommendations (auto-generated)
          04 PneumoBot Chat Summary (all messages, colour-coded)
          05 Disclaimer
          Footer on every page
                │
                ▼
        PDF downloaded → shown to doctor
        Doctor explains diagnosis to patient
        using heatmap images + chat explanations
```

---

*BOT.md — PneumoScan Chatbot, Storage & Report Documentation*
*Nirma University B.Tech CSE — RMS Project 2025-26*
*Parth Lathiya (23BCE215) | Dharm Patel (23BCE224)*
