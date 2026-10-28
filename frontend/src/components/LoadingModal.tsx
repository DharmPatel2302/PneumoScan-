"use client";

import { useEffect, useState } from "react";
import { Microscope } from "lucide-react";

const STEPS = [
  "Validating image format...",
  "Preprocessing X-ray...",
  "Running DenseNet121...",
  "Computing Grad-CAM...",
  "Generating XAI overlay...",
  "Finalizing diagnosis...",
];

interface LoadingModalProps {
  visible: boolean;
}

export default function LoadingModal({ visible }: LoadingModalProps) {
  const [progress, setProgress] = useState(0);
  const [stepIndex, setStepIndex] = useState(0);

  useEffect(() => {
    if (!visible) {
      setProgress(0);
      setStepIndex(0);
      return;
    }

    // Advance progress smoothly, slowing near 90% to wait for real response
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 90) {
          clearInterval(interval);
          return prev;
        }
        return prev + Math.random() * 4 + 1;
      });
    }, 350);

    // Cycle through steps
    const stepInterval = setInterval(() => {
      setStepIndex((prev) => (prev + 1) % STEPS.length);
    }, 1400);

    return () => {
      clearInterval(interval);
      clearInterval(stepInterval);
    };
  }, [visible]);

  if (!visible) return null;

  const clampedProgress = Math.min(Math.round(progress), 90);

  return (
    /* ── Backdrop ── */
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Blurred background */}
      <div className="absolute inset-0 bg-[#f5f5f0]/60 backdrop-blur-md" />

      {/* ── Modal Card ── */}
      <div className="relative z-10 bg-[#f0f0eb] border border-gray-200 rounded-2xl shadow-2xl p-10 w-[90vw] max-w-sm flex flex-col items-center gap-6 animate-fade-in">

        {/* ── Spinning Arc Icon ── */}
        <div className="relative w-20 h-20 flex items-center justify-center">
          {/* Outer spinning arc */}
          <svg
            className="absolute inset-0 arc-spin"
            viewBox="0 0 80 80"
            fill="none"
          >
            <circle
              cx="40"
              cy="40"
              r="34"
              stroke="#e5e7eb"
              strokeWidth="4"
            />
            <path
              d="M 40 6 A 34 34 0 0 1 74 40"
              stroke="#22c55e"
              strokeWidth="4"
              strokeLinecap="round"
            />
          </svg>
          {/* Center icon */}
          <div className="w-11 h-11 bg-white rounded-full flex items-center justify-center shadow-sm">
            <Microscope size={22} className="text-green-500" />
          </div>
        </div>

        {/* ── Headline ── */}
        <div className="text-center">
          <h2 className="text-xl font-bold text-gray-900 leading-tight">
            Analyzing Specimen...
          </h2>
          <p className="text-green-600 text-sm font-medium mt-1 min-h-[20px] transition-all duration-500">
            {STEPS[stepIndex]}
          </p>
        </div>

        {/* ── Progress Bar ── */}
        <div className="w-full">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[10px] font-semibold tracking-widest text-gray-500 uppercase">
              Diagnostic Pipeline
            </span>
            <span className="text-[11px] font-bold text-gray-700">
              {clampedProgress}%
            </span>
          </div>

          {/* Track */}
          <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
            {/* Fill */}
            <div
              className="h-full bg-gradient-to-r from-green-500 to-green-400 rounded-full transition-all duration-300 ease-out"
              style={{ width: `${clampedProgress}%` }}
            />
          </div>
        </div>

        {/* ── Footer Note ── */}
        <p className="text-gray-500 text-xs text-center">
          Hang tight, we&apos;re examining your X-ray 🩺
        </p>
      </div>
    </div>
  );
}
