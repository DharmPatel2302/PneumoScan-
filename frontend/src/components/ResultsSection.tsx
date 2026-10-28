"use client";

import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import {
  Download, ZoomIn, AlertTriangle, CheckCircle2,
  Loader2, X, Activity,
} from "lucide-react";
import { generateClinicalReport } from "@/utils/generateReport";
import type { ChatMessage } from "@/app/page";
import type { ClassResult, PredictionResult } from "@/context/ChatContext";

// ============================================================
// TYPES
// ============================================================

interface ResultsSectionProps {
  result: PredictionResult;
  onReset: () => void;
  chatHistory: ChatMessage[];
}

// ============================================================
// HELPERS
// ============================================================

// Colour theme per disease severity group
function classColor(label: string): {
  badge: string; bar: string; text: string; bg: string;
} {
  switch (label) {
    case "Pneumonia":
    case "Consolidation":
    case "Pneumothorax":
      return { badge: "bg-red-100 text-red-700 border-red-200",    bar: "bg-red-500",    text: "text-red-700",    bg: "bg-red-50"    };
    case "Edema":
    case "Cardiomegaly":
    case "Lung_Lesion":
      return { badge: "bg-orange-100 text-orange-700 border-orange-200", bar: "bg-orange-500", text: "text-orange-700", bg: "bg-orange-50" };
    case "Atelectasis":
    case "Effusion":
    case "Pleural_Thickening":
    case "Infiltration":
      return { badge: "bg-yellow-100 text-yellow-700 border-yellow-200", bar: "bg-yellow-500", text: "text-yellow-700", bg: "bg-yellow-50" };
    case "no_finding":
      return { badge: "bg-green-100 text-green-700 border-green-200",  bar: "bg-green-500",  text: "text-green-700",  bg: "bg-green-50"  };
    default:
      return { badge: "bg-gray-100 text-gray-700 border-gray-200",    bar: "bg-gray-400",   text: "text-gray-700",   bg: "bg-gray-50"   };
  }
}

// ============================================================
// ANIMATED BAR
// ============================================================

function ProbabilityBar({ value, color }: { value: number; color: string }) {
  const [animated, setAnimated] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setAnimated(value), 80);
    return () => clearTimeout(t);
  }, [value]);
  return (
    <div className="w-full h-1.5 bg-gray-100 rounded-full overflow-hidden">
      <div
        className={`h-full ${color} rounded-full transition-all duration-700 ease-out`}
        style={{ width: `${animated}%` }}
      />
    </div>
  );
}

// ============================================================
// FINDINGS LIST — all 11 classes
// ============================================================

function FindingsPanel({ findings }: { findings: ClassResult[] }) {
  const [showAll, setShowAll] = useState(false);
  const visible = showAll ? findings : findings.slice(0, 6);

  return (
    <div className="space-y-2.5">
      {visible.map((f) => {
        const c = classColor(f.label);
        return (
          <div key={f.label} className="flex items-center gap-3">
            {/* Badge */}
            <div className={`flex-shrink-0 w-5 h-5 rounded-full border flex items-center justify-center ${c.badge}`}>
              {f.detected
                ? <span className="w-2 h-2 rounded-full bg-current" />
                : <span className="w-1 h-1 rounded-full bg-current opacity-30" />
              }
            </div>

            {/* Name */}
            <span className={`text-xs font-medium w-36 flex-shrink-0 ${f.detected ? c.text : "text-gray-400"}`}>
              {f.display_name}
            </span>

            {/* Bar */}
            <div className="flex-1">
              <ProbabilityBar value={f.probability} color={f.detected ? c.bar : "bg-gray-200"} />
            </div>

            {/* Percent */}
            <span className={`text-xs font-semibold w-10 text-right flex-shrink-0 ${f.detected ? c.text : "text-gray-400"}`}>
              {f.probability.toFixed(1)}%
            </span>

            {/* Detected pill */}
            {f.detected && (
              <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded-full border flex-shrink-0 ${c.badge}`}>
                DETECTED
              </span>
            )}
          </div>
        );
      })}

      {findings.length > 6 && (
        <button
          onClick={() => setShowAll((v) => !v)}
          className="text-xs text-green-600 hover:text-green-700 font-semibold mt-1"
        >
          {showAll ? "Show less ▲" : `Show all ${findings.length} findings ▼`}
        </button>
      )}
    </div>
  );
}

// ============================================================
// XAI IMAGE CARD (unchanged UI, updated label/caption/footer)
// ============================================================

interface XAIImageCardProps {
  src: string;
  label: string;
  caption: string;
  badgeLabel: string;   // e.g. "Consolidation" or "No Finding"
  badgeColor: string;   // tailwind classes
  probability: number;
}

function XAIImageCard({ src, label, caption, badgeLabel, badgeColor, probability }: XAIImageCardProps) {
  const [zoomed, setZoomed] = useState(false);
  const filename = `${label.toLowerCase().replace(/\s+/g, "_")}.png`;

  const handleDownload = (e: React.MouseEvent) => {
    e.stopPropagation();
    const a = document.createElement("a");
    a.href = `data:image/png;base64,${src}`;
    a.download = filename;
    a.click();
  };

  useEffect(() => {
    if (!zoomed) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setZoomed(false); };
    window.addEventListener("keydown", onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [zoomed]);

  return (
    <>
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <span className="text-[10px] font-semibold tracking-widest text-gray-500 uppercase">
            {label}
          </span>
          <button onClick={() => setZoomed(true)} className="text-gray-400 hover:text-gray-700 transition-colors">
            <ZoomIn size={15} />
          </button>
        </div>

        <div
          className="relative w-full aspect-square rounded-xl overflow-hidden bg-gray-100 cursor-pointer group"
          onClick={() => setZoomed(true)}
        >
          <img
            src={`data:image/png;base64,${src}`}
            alt={label}
            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
          />
          <div className="absolute inset-0 bg-black/0 group-hover:bg-black/25 transition-colors duration-200 flex items-center justify-center">
            <span className="opacity-0 group-hover:opacity-100 transition-opacity duration-200 bg-white/90 text-gray-900 text-xs font-semibold px-3 py-1.5 rounded-full flex items-center gap-1.5">
              <ZoomIn size={12} /> View Full Image
            </span>
          </div>
        </div>
        <p className="text-[11px] text-gray-400 italic text-center">{caption}</p>
      </div>

      {zoomed && createPortal(
        <div
          className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/40"
          onClick={() => setZoomed(false)}
        >
          <div
            className="bg-[#1e1e1e] rounded-2xl overflow-hidden shadow-2xl w-full max-w-3xl max-h-[92vh] flex flex-col"
            style={{ animation: "fadeScaleIn 0.18s ease-out" }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-start justify-between px-5 py-4 border-b border-white/10 flex-shrink-0">
              <div className="min-w-0 pr-4">
                <h3 className="text-white font-bold text-base leading-tight truncate">{label}</h3>
                <p className="text-gray-400 text-xs mt-0.5 truncate">{filename}</p>
              </div>
              <div className="flex items-center gap-2 flex-shrink-0">
                <button onClick={handleDownload} title="Download"
                  className="w-8 h-8 rounded-full bg-white/10 hover:bg-white/20 flex items-center justify-center text-gray-300 hover:text-white transition-colors">
                  <Download size={15} />
                </button>
                <button onClick={() => setZoomed(false)} title="Close"
                  className="w-8 h-8 rounded-full bg-white/10 hover:bg-white/20 flex items-center justify-center text-gray-300 hover:text-white transition-colors">
                  <X size={15} />
                </button>
              </div>
            </div>

            {/* Image */}
            <div className="bg-black flex items-center justify-center p-4 flex-1 min-h-0 overflow-hidden">
              <img
                src={`data:image/png;base64,${src}`}
                alt={label}
                className="max-w-full max-h-[70vh] rounded-lg object-contain"
              />
            </div>

            {/* Footer */}
            <div className="px-5 py-3 flex items-center justify-between border-t border-white/10 flex-shrink-0">
              <span className={`text-[10px] font-bold px-2.5 py-1 rounded-full uppercase tracking-wide border ${badgeColor}`}>
                {badgeLabel}
              </span>
              <span className="text-gray-400 text-xs">
                Probability: <span className="text-white font-semibold">{probability.toFixed(1)}%</span>
              </span>
            </div>
          </div>
        </div>,
        document.body
      )}
    </>
  );
}

// ============================================================
// MAIN COMPONENT
// ============================================================

export default function ResultsSection({ result, onReset, chatHistory }: ResultsSectionProps) {
  const [generating, setGenerating] = useState(false);

  const isNormal    = result.primary_label === "no_finding";
  const hasDisease  = result.detected_diseases.length > 0;
  const primaryColor = classColor(result.primary_label);

  // XAI badge: use gradcam class info
  const gradcamColor = classColor(result.gradcam_class);

  const handleDownloadReport = async () => {
    setGenerating(true);
    try {
      await generateClinicalReport({
        // Map to legacy ReportData shape so the existing PDF generator still works
        prediction:            result.primary_display.toUpperCase(),
        confidence:            result.primary_probability,
        pneumonia_probability: result.all_findings.find((f) => f.label === "Pneumonia")?.probability ?? 0,
        original_image:        result.original_image,
        gradcam_heatmap:       result.gradcam_heatmap,
        overlay_image:         result.overlay_image,
        chatHistory:           chatHistory.map((m) => ({ role: m.role, content: m.content })),
      });
    } finally {
      setGenerating(false);
    }
  };

  return (
    <section className="w-full max-w-4xl mx-auto animate-fade-up">

      {/* ── PRIMARY DIAGNOSIS BANNER ── */}
      <div className={`w-full rounded-2xl p-6 mb-6 flex items-start gap-4 ${
        isNormal ? "bg-green-50 border border-green-200" : "bg-red-50 border border-red-200"
      }`}>
        <div className="mt-0.5">
          {isNormal
            ? <CheckCircle2 size={24} className="text-green-600" />
            : <AlertTriangle size={24} className="text-red-500" />
          }
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-[11px] font-semibold tracking-widest text-gray-500 uppercase mb-1">
            Primary Finding
          </p>
          <div className="flex flex-wrap items-center gap-3 mb-1">
            <h2 className="text-2xl font-black tracking-tight text-gray-900">
              {result.primary_display}
            </h2>
            <span className={`text-xs font-bold px-2.5 py-1 rounded-full border ${primaryColor.badge}`}>
              {result.primary_probability.toFixed(1)}%
            </span>
          </div>

          {/* Detected diseases chips */}
          {hasDisease && (
            <div className="flex flex-wrap gap-1.5 mt-2">
              {result.detected_diseases.map((d) => {
                const dc = classColor(d.label);
                return (
                  <span key={d.label}
                    className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border ${dc.badge}`}>
                    {d.display_name} {d.probability.toFixed(0)}%
                  </span>
                );
              })}
            </div>
          )}

          {isNormal && !hasDisease && (
            <p className="text-sm text-gray-500 mt-1">
              No significant pathological findings detected above threshold.
            </p>
          )}
        </div>
      </div>

      {/* ── ALL FINDINGS PANEL ── */}
      <div className="bg-white rounded-2xl border border-gray-200 p-6 mb-6 shadow-sm">
        <div className="flex items-center gap-2 mb-4">
          <Activity size={16} className="text-green-500" />
          <p className="text-[10px] font-semibold tracking-widest text-gray-500 uppercase">
            All Classification Findings
          </p>
        </div>
        <FindingsPanel findings={result.all_findings} />
      </div>

      {/* ── XAI SECTION ── */}
      <div className="bg-white rounded-2xl border border-gray-200 p-6 mb-6 shadow-sm">
        <div className="mb-5">
          <p className="text-[10px] font-semibold tracking-widest text-gray-500 uppercase mb-1">
            Interpretability Analysis (XAI)
          </p>
          <p className="text-sm text-gray-600">
            Grad-CAM highlights regions the model focused on when predicting{" "}
            <span className="font-semibold text-gray-800">{result.gradcam_display}</span>.
          </p>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-5">
          <XAIImageCard
            src={result.original_image}
            label="Original Image"
            caption="Target specimen provided for analysis."
            badgeLabel={result.primary_display}
            badgeColor={primaryColor.badge}
            probability={result.primary_probability}
          />
          <XAIImageCard
            src={result.gradcam_heatmap}
            label="Grad-CAM Heatmap"
            caption="Heat map of model attention regions."
            badgeLabel={result.gradcam_display}
            badgeColor={gradcamColor.badge}
            probability={result.all_findings.find((f) => f.label === result.gradcam_class)?.probability ?? 0}
          />
          <XAIImageCard
            src={result.overlay_image}
            label="XAI Overlay"
            caption="AI identifying contributing visual features."
            badgeLabel={result.gradcam_display}
            badgeColor={gradcamColor.badge}
            probability={result.all_findings.find((f) => f.label === result.gradcam_class)?.probability ?? 0}
          />
        </div>
      </div>

      {/* ── Chat included notice ── */}
      {chatHistory.length > 0 && (
        <div className="bg-blue-50 border border-blue-200 rounded-xl px-4 py-3 mb-6 flex items-center gap-3">
          <span className="text-blue-500 text-lg">💬</span>
          <p className="text-xs text-blue-700">
            <span className="font-semibold">{chatHistory.length} chat message{chatHistory.length !== 1 ? "s" : ""}</span>
            {" "}will be included in the PDF report.
          </p>
        </div>
      )}

      {/* ── Disclaimer ── */}
      <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 mb-6">
        <p className="text-xs text-amber-700 leading-relaxed">
          ⚠️ This application is intended for research and educational purposes only.
          It is <strong>not</strong> a certified medical diagnostic tool. Always consult
          a qualified radiologist for clinical decisions.
        </p>
      </div>

      {/* ── Actions ── */}
      <div className="flex flex-wrap gap-3 justify-center">
        <button
          onClick={handleDownloadReport}
          disabled={generating}
          className="flex items-center gap-2 px-5 py-2.5 bg-[#1c1c1c] hover:bg-gray-800 disabled:bg-gray-400 text-white text-sm font-semibold rounded-xl transition-colors"
        >
          {generating
            ? <><Loader2 size={15} className="animate-spin" /> Generating Report...</>
            : <><Download size={15} /> Download Clinical Report</>
          }
        </button>
        <button
          onClick={onReset}
          className="flex items-center gap-2 px-5 py-2.5 bg-green-500 text-white text-sm font-semibold rounded-xl hover:bg-green-600 transition-colors"
        >
          Analyze Another X-ray
        </button>
      </div>
    </section>
  );
}
