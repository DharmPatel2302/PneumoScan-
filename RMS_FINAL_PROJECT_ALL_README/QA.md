# PneumoScan — Demo Questions Guide
### Which questions go to which model

---

## 🔵 MedVQA Vision Model Questions
> Blue bubble • `● MedVQA Vision Model` badge
> Model answers as if it is **visually examining the X-ray image**
> *(Upload an X-ray first, then ask these)*

### About the Image Directly
| # | Question |
|---|---|
| 1 | What do you see in this x-ray? |
| 2 | Describe this image findings |
| 3 | What abnormality is visible in this x-ray? |
| 4 | What is shown in this scan? |
| 5 | Explain this image in detail |

### Lung-Specific Visual Questions
| # | Question |
|---|---|
| 6 | What is shown in the left lung? |
| 7 | What is shown in the right lung? |
| 8 | Which lung is more affected in this x-ray? |
| 9 | Is there any opacity visible in the left lung? |
| 10 | Describe what you observe in the right lung |

### Presence Detection
| # | Question |
|---|---|
| 11 | Does this x-ray show any signs of pneumonia? |
| 12 | Can you see any consolidation in this scan? |
| 13 | Is there any fluid visible in this x-ray? |
| 14 | Does this x-ray show cardiomegaly? |
| 15 | Is there any pleural effusion visible in this image? |
| 16 | What do you see in the cardiac region in this x-ray? |

### Heatmap and Overlay Questions
| # | Question |
|---|---|
| 17 | What does the heatmap show? |
| 18 | Which region is highlighted in the overlay? |
| 19 | What area is the model focusing on in this x-ray? |
| 20 | What is shown in the highlighted region? |

### Abnormality Questions
| # | Question |
|---|---|
| 21 | What abnormality is visible in this scan? |
| 22 | What do you see in this x-ray as the main finding? |
| 23 | Based on the image, what is the primary concern? |
| 24 | What can be seen in the lower region of this x-ray? |
| 25 | What is shown in the upper left region of this scan? |

---

## 🟢 PneumoBot LLM Questions
> Green bubble • `● PneumoBot LLM` badge
> Model answers from **general medical knowledge**
> *(Can be asked anytime — no image needed)*

### About the Diagnosis Result
| # | Question |
|---|---|
| 1 | What does this result mean? |
| 2 | How serious is this diagnosis? |
| 3 | What should I do next after this diagnosis? |
| 4 | Is this result accurate? |
| 5 | What are the next steps? |

### Disease Knowledge
| # | Question |
|---|---|
| 6 | What is pneumonia? |
| 7 | What causes pneumonia? |
| 8 | What are the symptoms of pneumonia? |
| 9 | How is pneumonia treated? |
| 10 | What is atelectasis? |
| 11 | What causes pleural effusion? |
| 12 | How serious is pneumothorax? |
| 13 | What is pulmonary edema? |
| 14 | What is cardiomegaly? |
| 15 | What is consolidation in the lungs? |
| 16 | What is pleural thickening? |
| 17 | What is infiltration in chest X-ray? |

### Treatment and Clinical
| # | Question |
|---|---|
| 18 | How is atelectasis treated? |
| 19 | What antibiotics are used for pneumonia? |
| 20 | When should I see a doctor for chest pain? |
| 21 | What is the recovery time for pneumonia? |
| 22 | Can pneumonia be prevented? |
| 23 | What is the difference between viral and bacterial pneumonia? |
| 24 | Is pneumothorax an emergency? |
| 25 | What lifestyle changes help lung health? |

### About the AI System
| # | Question |
|---|---|
| 26 | How does this AI model work? |
| 27 | How accurate is this model? |
| 28 | What is Grad-CAM? |
| 29 | What is DenseNet121? |
| 30 | What is XAI in medical imaging? |
| 31 | How was this model trained? |
| 32 | What datasets were used to train this model? |
| 33 | What is the confidence score? |
| 34 | What is MedVQA? |
| 35 | How does the chatbot work? |

### Radiology Concepts
| # | Question |
|---|---|
| 36 | How is a chest X-ray read? |
| 37 | What is a normal chest X-ray supposed to look like? |
| 38 | What does opacity mean in a chest X-ray? |
| 39 | What is the difference between X-ray and CT scan? |
| 40 | What does air bronchogram mean in radiology? |

---

## 📋 Quick Demo Script for Faculty

**Step 1** — Upload a chest X-ray image and wait for the prediction result.

**Step 2** — Ask these questions in order to demonstrate both models:

```
1. "What do you see in this x-ray?"          → 🔵 MedVQA blue bubble
2. "What is pneumonia?"                       → 🟢 LLM green bubble
3. "Describe this image findings"             → 🔵 MedVQA blue bubble
4. "How accurate is this model?"              → 🟢 LLM green bubble
5. "What is shown in the left lung?"          → 🔵 MedVQA blue bubble
6. "What should I do next?"                   → 🟢 LLM green bubble
7. "Does this x-ray show any consolidation?"  → 🔵 MedVQA blue bubble
8. "What is Grad-CAM?"                        → 🟢 LLM green bubble
```

**What to show:** The alternating blue and green bubbles visually prove that two separate AI modules are working — the MedVQA Vision Model for image-specific analysis and the PneumoBot LLM for general medical knowledge.

---

## 🔑 Keyword Trigger Reference

These words/phrases in a question **trigger MedVQA mode** (blue bubble):

| Trigger Phrase | Example |
|---|---|
| `this x-ray` | "What do you see in **this x-ray**?" |
| `this image` | "Describe **this image** findings" |
| `this scan` | "What is shown in **this scan**?" |
| `in the image` | "What abnormality is **in the image**?" |
| `left lung` | "What is shown in the **left lung**?" |
| `right lung` | "Is **right lung** normal?" |
| `heatmap` | "What does the **heatmap** show?" |
| `overlay` | "What region is in the **overlay**?" |
| `highlighted` | "What is **highlighted** here?" |
| `based on the image` | "**Based on the image**, what is wrong?" |
| `does this x-ray show` | "**Does this x-ray show** effusion?" |

Any other question → **PneumoBot LLM** (green bubble).
