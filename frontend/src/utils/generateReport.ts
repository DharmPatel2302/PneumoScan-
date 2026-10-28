import { jsPDF } from "jspdf";

// ============================================================
// TYPES
// ============================================================

export interface ReportData {
  prediction: string;
  confidence: number;
  pneumonia_probability: number;
  original_image: string;   // base64 PNG
  gradcam_heatmap: string;  // base64 PNG
  overlay_image: string;    // base64 PNG
  chatHistory: { role: string; content: string }[];
  reportDate?: string;
}

// ============================================================
// HELPERS
// ============================================================

const GREEN  = [34, 197, 94]   as [number, number, number];
const RED    = [220, 38, 38]   as [number, number, number];
const DARK   = [28, 28, 28]    as [number, number, number];
const GRAY   = [107, 114, 128] as [number, number, number];
const LGRAY  = [243, 244, 246] as [number, number, number];
const WHITE  = [255, 255, 255] as [number, number, number];
const AMBER  = [217, 119, 6]   as [number, number, number];

function hex(doc: jsPDF, r: number, g: number, b: number) {
  doc.setTextColor(r, g, b);
}

function stripMarkdown(text: string): string {
  return text
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/\*([^*]+)\*/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/^#{1,3}\s+/gm, "")
    .replace(/^[-*•]\s/gm, "• ")
    .replace(/^\d+\.\s/gm, "")
    .replace(/\|[^\n]+\|/g, "")
    .replace(/^[-|]+$/gm, "")
    .trim();
}

// Wraps text and returns final Y after drawing
function drawWrappedText(
  doc: jsPDF,
  text: string,
  x: number,
  y: number,
  maxWidth: number,
  lineHeight: number,
  pageH: number,
  margin: number
): number {
  const lines = doc.splitTextToSize(text, maxWidth);
  for (const line of lines) {
    if (y + lineHeight > pageH - margin) {
      doc.addPage();
      y = margin;
      drawWatermark(doc);
    }
    doc.text(line, x, y);
    y += lineHeight;
  }
  return y;
}

function drawWatermark(doc: jsPDF) {
  const pw = doc.internal.pageSize.getWidth();
  const ph = doc.internal.pageSize.getHeight();
  doc.saveGraphicsState();
  doc.setGState(new (doc as any).GState({ opacity: 0.04 }));
  doc.setFontSize(72);
  doc.setTextColor(34, 197, 94);
  doc.setFont("helvetica", "bold");
  // Rotate 45° watermark in center
  const cx = pw / 2;
  const cy = ph / 2;
  doc.text("PneumoScan", cx, cy, {
    align: "center",
    angle: 45,
  });
  doc.restoreGraphicsState();
  // Reset text color
  doc.setTextColor(0, 0, 0);
}

function drawHRule(doc: jsPDF, y: number, margin: number, color: [number,number,number] = LGRAY) {
  const pw = doc.internal.pageSize.getWidth();
  doc.setDrawColor(...color);
  doc.setLineWidth(0.3);
  doc.line(margin, y, pw - margin, y);
}

function sectionHeader(
  doc: jsPDF,
  title: string,
  y: number,
  margin: number
): number {
  doc.setFillColor(...DARK);
  const pw = doc.internal.pageSize.getWidth();
  doc.roundedRect(margin, y, pw - margin * 2, 9, 1.5, 1.5, "F");
  doc.setFontSize(9);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(...WHITE);
  doc.text(title.toUpperCase(), margin + 4, y + 6);
  doc.setTextColor(0, 0, 0);
  return y + 14;
}

// ============================================================
// MAIN EXPORT
// ============================================================

export async function generateClinicalReport(data: ReportData): Promise<void> {
  const doc = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });

  const pw    = doc.internal.pageSize.getWidth();   // 210
  const ph    = doc.internal.pageSize.getHeight();  // 297
  const M     = 18;   // margin
  const cw    = pw - M * 2; // content width
  const isPneumonia = data.prediction === "PNEUMONIA";
  const now   = data.reportDate ?? new Date().toLocaleString("en-IN", {
    day: "2-digit", month: "long", year: "numeric",
    hour: "2-digit", minute: "2-digit",
  });

  // ── PAGE 1 ───────────────────────────────────────────────
  drawWatermark(doc);

  // ── Header bar ──
  doc.setFillColor(...DARK);
  doc.rect(0, 0, pw, 22, "F");

  // Logo text
  doc.setFontSize(16);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(...WHITE);
  doc.text("Pneumo", M, 14);
  doc.setTextColor(...GREEN);
  doc.text("Scan", M + 31, 14);

  // Right side — report title
  doc.setFontSize(8);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(180, 180, 180);
  doc.text("CLINICAL RADIOLOGY REPORT", pw - M, 10, { align: "right" });
  doc.text(`Generated: ${now}`, pw - M, 16, { align: "right" });

  let y = 32;

  // ── Report ID & info row ──
  doc.setFontSize(8);
  doc.setFont("helvetica", "normal");
  hex(doc, ...GRAY);
  const reportId = `RPT-${Date.now().toString(36).toUpperCase()}`;
  doc.text(`Report ID: ${reportId}`, M, y);
  doc.text("System: PneumoScan v1.0 — DenseNet121 + Grad-CAM", pw - M, y, { align: "right" });
  y += 5;
  drawHRule(doc, y, M);
  y += 8;

  // ── DIAGNOSIS RESULT ─────────────────────────────────────
  y = sectionHeader(doc, "01  Diagnosis Result", y, M);

  // Result badge
  const badgeColor = isPneumonia ? RED : GREEN;
  doc.setFillColor(...badgeColor);
  doc.roundedRect(M, y, 60, 14, 2, 2, "F");
  doc.setFontSize(11);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(...WHITE);
  doc.text(data.prediction, M + 30, y + 9, { align: "center" });

  // Confidence on the right
  doc.setFontSize(22);
  doc.setFont("helvetica", "bold");
  hex(doc, ...DARK);
  doc.text(`${data.confidence.toFixed(1)}%`, pw - M, y + 11, { align: "right" });
  doc.setFontSize(7);
  hex(doc, ...GRAY);
  doc.text("CONFIDENCE SCORE", pw - M, y + 16, { align: "right" });

  y += 20;

  // Confidence bar
  const barW = cw - 70;
  doc.setFillColor(...LGRAY);
  doc.roundedRect(M, y, barW, 5, 2, 2, "F");
  doc.setFillColor(...badgeColor);
  doc.roundedRect(M, y, barW * (data.confidence / 100), 5, 2, 2, "F");
  y += 12;

  // Summary line
  doc.setFontSize(9);
  doc.setFont("helvetica", "normal");
  hex(doc, ...DARK);
  const summary = isPneumonia
    ? `The AI classifier detected radiological features consistent with pneumonia (probability: ${data.pneumonia_probability.toFixed(1)}%). The DenseNet121 model identified patterns in the chest X-ray that suggest active pulmonary infection.`
    : `The AI classifier found no significant radiological features associated with pneumonia (pneumonia probability: ${data.pneumonia_probability.toFixed(1)}%). The chest X-ray appears within normal limits.`;
  y = drawWrappedText(doc, summary, M, y, cw, 5.5, ph, M);
  y += 6;

  // ── IMAGING ANALYSIS ─────────────────────────────────────
  y = sectionHeader(doc, "02  Imaging Analysis — Grad-CAM XAI", y, M);

  doc.setFontSize(8);
  doc.setFont("helvetica", "normal");
  hex(doc, ...GRAY);
  doc.text(
    "Gradient-weighted Class Activation Mapping (Grad-CAM) highlights regions that most influenced the model prediction.",
    M, y, { maxWidth: cw }
  );
  y += 10;

  // Three images side by side
  const imgW = (cw - 8) / 3;
  const imgH = imgW * 0.9;
  const labels = ["Original X-Ray", "Grad-CAM Heatmap", "XAI Overlay"];
  const captions = [
    "Target specimen provided for analysis.",
    "Model attention heatmap (JET colormap).",
    "Overlay identifying contributing features.",
  ];
  const imgs = [data.original_image, data.gradcam_heatmap, data.overlay_image];

  // Check page space
  if (y + imgH + 20 > ph - M) {
    doc.addPage();
    drawWatermark(doc);
    y = M;
  }

  imgs.forEach((b64, idx) => {
    const ix = M + idx * (imgW + 4);
    try {
      doc.addImage(`data:image/png;base64,${b64}`, "PNG", ix, y, imgW, imgH);
    } catch {
      doc.setFillColor(...LGRAY);
      doc.rect(ix, y, imgW, imgH, "F");
    }
    // Label above
    doc.setFontSize(7);
    doc.setFont("helvetica", "bold");
    hex(doc, ...GRAY);
    doc.text(labels[idx].toUpperCase(), ix + imgW / 2, y - 2, { align: "center" });
    // Caption below
    doc.setFont("helvetica", "italic");
    doc.setFontSize(6.5);
    hex(doc, ...GRAY);
    const capLines = doc.splitTextToSize(captions[idx], imgW);
    capLines.forEach((cl: string, ci: number) => {
      doc.text(cl, ix + imgW / 2, y + imgH + 4 + ci * 4, { align: "center" });
    });
  });

  y += imgH + 14;

  // ── CLINICAL RECOMMENDATIONS ─────────────────────────────
  if (y + 60 > ph - M) {
    doc.addPage();
    drawWatermark(doc);
    y = M;
  }

  y = sectionHeader(doc, "03  Clinical Recommendations", y, M);

  const recs = isPneumonia ? [
    "Consult a licensed pulmonologist or general physician immediately for clinical evaluation.",
    "A confirmatory sputum culture or blood test (CBC, CRP) is recommended to identify causative organism.",
    "Antibiotic therapy should be initiated only under medical supervision based on culture results.",
    "Adequate hydration, rest, and fever management (antipyretics as prescribed) are advised.",
    "Follow-up chest X-ray in 4–6 weeks to confirm radiological clearance after treatment.",
    "Avoid smoking and exposure to pollutants during recovery period.",
    "Hospitalization may be required for severe cases — assess SpO₂ and respiratory rate.",
  ] : [
    "No immediate intervention required based on the AI analysis.",
    "Continue routine preventive health care and annual check-ups.",
    "Maintain adequate ventilation, exercise, and a balanced diet to support lung health.",
    "If respiratory symptoms develop (cough, fever, dyspnea), seek immediate clinical evaluation.",
    "Consider pneumococcal and influenza vaccinations as preventive measures.",
  ];

  doc.setFontSize(8.5);
  for (const rec of recs) {
    if (y + 8 > ph - M) {
      doc.addPage();
      drawWatermark(doc);
      y = M;
    }
    doc.setFont("helvetica", "normal");
    hex(doc, ...DARK);
    // bullet dot
    doc.setFillColor(...GREEN);
    doc.circle(M + 2, y - 1.5, 1.2, "F");
    y = drawWrappedText(doc, rec, M + 6, y, cw - 6, 5.2, ph, M);
    y += 1.5;
  }
  y += 4;

  // ── CHAT SUMMARY ─────────────────────────────────────────
  const chatMsgs = data.chatHistory.filter(
    (m) => m.role !== "system" && m.content.trim() !== ""
  );

  if (chatMsgs.length > 0) {
    if (y + 30 > ph - M) {
      doc.addPage();
      drawWatermark(doc);
      y = M;
    }

    y = sectionHeader(doc, "04  PneumoBot Chat Summary", y, M);

    doc.setFontSize(8);
    doc.setFont("helvetica", "normal");
    hex(doc, ...GRAY);
    doc.text(
      `${chatMsgs.length} message(s) exchanged with the medical AI assistant.`,
      M, y
    );
    y += 8;

    for (const msg of chatMsgs) {
      if (y + 14 > ph - M) {
        doc.addPage();
        drawWatermark(doc);
        y = M;
      }

      const isUser = msg.role === "user";
      const roleLabel = isUser ? "Patient / User" : "PneumoBot AI";
      const bgColor: [number, number, number] = isUser ? [239, 246, 255] : [240, 253, 244];
      const labelColor: [number, number, number] = isUser ? [37, 99, 235] : [22, 163, 74];

      // Background bubble
      const cleanText = stripMarkdown(msg.content);
      const lines = doc.splitTextToSize(cleanText, cw - 10);
      const bubbleH = lines.length * 4.8 + 10;

      if (y + bubbleH > ph - M) {
        doc.addPage();
        drawWatermark(doc);
        y = M;
      }

      doc.setFillColor(...bgColor);
      doc.roundedRect(M, y, cw, bubbleH, 2, 2, "F");

      // Role label
      doc.setFontSize(7);
      doc.setFont("helvetica", "bold");
      doc.setTextColor(...labelColor);
      doc.text(roleLabel, M + 4, y + 5);

      // Message text
      doc.setFontSize(8);
      doc.setFont("helvetica", "normal");
      hex(doc, ...DARK);
      lines.forEach((line: string, li: number) => {
        doc.text(line, M + 4, y + 10 + li * 4.8);
      });

      y += bubbleH + 3;
    }
    y += 4;
  }

  // ── DISCLAIMER ───────────────────────────────────────────
  if (y + 28 > ph - M) {
    doc.addPage();
    drawWatermark(doc);
    y = M;
  }

  y = sectionHeader(doc, isPneumonia ? "05  Disclaimer" : "04  Disclaimer", y, M);

  doc.setFillColor(255, 251, 235);
  doc.setDrawColor(...AMBER);
  doc.setLineWidth(0.4);
  doc.roundedRect(M, y, cw, 22, 2, 2, "FD");
  doc.setFontSize(8);
  doc.setFont("helvetica", "bold");
  hex(doc, ...AMBER);
  doc.text("⚠  Important Notice", M + 4, y + 6);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(7.5);
  hex(doc, ...DARK);
  const disclaimer =
    "This report is generated by an AI-assisted research tool (PneumoScan) for educational and research purposes only. " +
    "It is NOT a certified medical diagnosis and must NOT replace professional clinical evaluation by a qualified radiologist or physician. " +
    "Always seek appropriate medical advice before making any health decisions.";
  const dLines = doc.splitTextToSize(disclaimer, cw - 8);
  dLines.forEach((dl: string, di: number) => {
    doc.text(dl, M + 4, y + 11 + di * 4);
  });
  y += 28;

  // ── FOOTER on every page ──────────────────────────────────
  const totalPages = (doc.internal as any).getNumberOfPages();
  for (let p = 1; p <= totalPages; p++) {
    doc.setPage(p);
    const fy = ph - 10;
    drawHRule(doc, fy - 4, M, [200, 200, 200]);
    doc.setFontSize(7);
    doc.setFont("helvetica", "normal");
    hex(doc, ...GRAY);
    doc.text("PneumoScan Clinical AI  •  For Research Use Only", M, fy);
    doc.text(`Page ${p} of ${totalPages}`, pw - M, fy, { align: "right" });
    doc.text(`Report ID: ${reportId}`, pw / 2, fy, { align: "center" });
  }

  // ── SAVE ─────────────────────────────────────────────────
  const filename = `PneumoScan_Report_${reportId}.pdf`;
  doc.save(filename);
}
