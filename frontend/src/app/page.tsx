"use client";

import { useState } from "react";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import UploadSection from "@/components/UploadSection";
import LoadingModal from "@/components/LoadingModal";
import ResultsSection from "@/components/ResultsSection";
import { useChatContext } from "@/context/ChatContext";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// Kept as a named export so ResultsSection can still import the type
export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export default function HomePage() {
  // pageLoading is local — it only matters while we're on this page
  const [pageLoading, setPageLoading] = useState(false);

  // result, apiError, and chat state all live in global context
  // so they survive navigation away and back
  const {
    result, setResult,
    apiError, setApiError,
    messages,
    setPredictionContext,
    clearChat,
  } = useChatContext();

  // Derive chatHistory for the PDF report from global messages
  // (excludes welcome message and error messages)
  const chatHistory: ChatMessage[] = messages
    .filter((m) => m.id !== "welcome" && !m.error)
    .map((m) => ({ role: m.role, content: m.content }));

  const handleAnalyze = async (file: File) => {
    setPageLoading(true);
    setResult(null);
    setApiError(null);
    // do NOT clear chat — pre-scan conversation should appear in the report

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData?.detail ?? `Server error: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
      // Update chatbot context with multiclass findings
      setPredictionContext({
        primary_label:       data.primary_label,
        primary_display:     data.primary_display,
        primary_probability: data.primary_probability,
        detected_diseases:   data.detected_diseases.map((d: { display_name: string }) => d.display_name),
      });
    } catch (err: unknown) {
      setApiError(err instanceof Error ? err.message : "An unexpected error occurred.");
    } finally {
      setPageLoading(false);
    }
  };

  const handleReset = () => {
    // clearChat resets result, apiError, predictionContext, and chat messages
    clearChat();
  };

  return (
    <div className="flex flex-col min-h-screen bg-[#f5f5f0]">
      <Navbar active="Home" />

      <main className="flex-1 w-full max-w-5xl mx-auto px-8 pt-10 pb-12">

        {/* ── Hero Text ── */}
        <div className="mb-8">
          <h1 className="text-6xl sm:text-7xl font-black tracking-tight text-gray-900 leading-[1.0]">
            Chest X-Ray
            <br />
            <span className="text-green-500">AI Analysis.</span>
          </h1>
          <p className="text-gray-500 text-sm mt-4 max-w-md leading-relaxed">
            Multiclass detection of 10 chest conditions using DenseNet121 with
            Grad-CAM explainability — trained on 311,000+ NIH &amp; CheXpert images.
          </p>
        </div>

        {/* ── Upload or Results ── */}
        {!result ? (
          <>
            <UploadSection onAnalyze={handleAnalyze} loading={pageLoading} />
            {apiError && (
              <div className="mt-6 bg-red-50 border border-red-200 rounded-xl p-4">
                <p className="text-sm text-red-700 font-medium">❌ {apiError}</p>
                <p className="text-xs text-red-500 mt-1">
                  Make sure the backend is running at{" "}
                  <code className="bg-red-100 px-1 rounded">{API_URL}</code>
                </p>
              </div>
            )}
          </>
        ) : (
          <ResultsSection
            result={result}
            onReset={handleReset}
            chatHistory={chatHistory}
          />
        )}
      </main>

      <LoadingModal visible={pageLoading} />

      <Footer />
    </div>
  );
}
