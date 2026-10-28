// ============================================================
// GALLERY CONFIGURATION
// ============================================================
// 1. Drop your X-ray image files into:  frontend/public/gallery/
// 2. Add an entry below — just set filename, label, type, confidence.
//    Description is auto-generated from the type if you leave it out.
// ============================================================

export interface GalleryItem {
  filename: string;
  label: string;
  type: "PNEUMONIA" | "NORMAL";
  confidence: number;
  description?: string; // optional — auto-filled if omitted
}

// ── Auto descriptions by type ────────────────────────────────
const PNEUMONIA_DESCRIPTIONS = [
  "Bilateral infiltrates visible in lower lobes consistent with pneumonia.",
  "Right lower lobe opacity suggesting lobar consolidation.",
  "Diffuse airspace opacification indicating active pulmonary infection.",
  "Perihilar haziness with increased lung markings in basal regions.",
  "Patchy consolidation in left lower lobe with air bronchograms.",
  "Dense homogeneous opacity in right upper lobe, lobar pneumonia pattern.",
];

const NORMAL_DESCRIPTIONS = [
  "Clear lung fields with no consolidation or infiltrates.",
  "Well-aerated lungs with normal cardiac silhouette.",
  "No focal opacities, pleural effusion, or pneumothorax detected.",
  "Normal bronchovascular markings, no acute cardiopulmonary findings.",
  "Bilateral lung fields clear, costophrenic angles sharp and well-defined.",
  "No evidence of airspace disease or pleural abnormality.",
];

// Returns a stable description based on the filename (not random each render)
function getAutoDescription(type: "PNEUMONIA" | "NORMAL", filename: string): string {
  const pool = type === "PNEUMONIA" ? PNEUMONIA_DESCRIPTIONS : NORMAL_DESCRIPTIONS;
  // Use character code sum of filename as a stable index
  const index =
    filename.split("").reduce((acc, c) => acc + c.charCodeAt(0), 0) % pool.length;
  return pool[index];
}

// ── Resolve final description ────────────────────────────────
export function resolveDescription(item: GalleryItem): string {
  return item.description ?? getAutoDescription(item.type, item.filename);
}

// ============================================================
// ADD YOUR IMAGES HERE
// ============================================================
const galleryItems: GalleryItem[] = [
  // Just fill in filename, label, type, confidence.
  // Description is auto-generated — or you can add your own.
  //
  {
    filename: "p1.jpg",
    label: "Pneumonia Case",
    type: "PNEUMONIA",
    confidence: 94.2,
  },
  {
    filename: "n1.jpg",
    label: "Normal Case",
    type: "NORMAL",
    confidence: 98.7,
  },
  {
    filename: "p2.jpg",
    label: "Pneumonia Case",
    type: "PNEUMONIA",
    confidence: 86.2,
  },
  {
    filename: "n2.jpg",
    label: "Normal Case",
    type: "NORMAL",
    confidence: 84.7,
  },
];

export default galleryItems;
