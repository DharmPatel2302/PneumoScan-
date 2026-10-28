"use client";

import {
  createContext,
  useContext,
  useState,
  useCallback,
  useRef,
  useEffect,
  type ReactNode,
} from "react";

// ============================================================
// TYPES
// ============================================================

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  error?: boolean;
  source?: "llm" | "medvqa_simulation" | "medvqa";  // which engine answered
}

// Matches the backend PredictionResponse schema exactly
export interface ClassResult {
  label:        string;
  display_name: string;
  probability:  number;   // 0 – 100
  detected:     boolean;
}

export interface PredictionResult {
  primary_label:       string;
  primary_display:     string;
  primary_probability: number;
  all_findings:        ClassResult[];
  detected_diseases:   ClassResult[];
  gradcam_class:       string;
  gradcam_display:     string;
  original_image:      string;
  gradcam_heatmap:     string;
  overlay_image:       string;
}

export interface PredictionContext {
  primary_label:       string;
  primary_display:     string;
  primary_probability: number;
  detected_diseases:   string[];   // list of display_names
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

interface ChatContextValue {
  messages: Message[];
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  open: boolean;
  setOpen: (v: boolean) => void;
  minimised: boolean;
  setMinimised: (v: boolean) => void;
  unread: number;
  setUnread: React.Dispatch<React.SetStateAction<number>>;
  loading: boolean;
  setLoading: (v: boolean) => void;
  predictionContext: PredictionContext | null;
  setPredictionContext: (ctx: PredictionContext | null) => void;
  sendMessage: (text: string) => Promise<void>;
  clearChat: () => void;
  lastReply: string | null;
  setLastReply: (s: string | null) => void;
  // ── persisted scan result ──
  result: PredictionResult | null;
  setResult: (r: PredictionResult | null) => void;
  apiError: string | null;
  setApiError: (e: string | null) => void;
}

// ============================================================
// HELPERS
// ============================================================

function generateId() {
  return Math.random().toString(36).slice(2, 9);
}

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

function buildWelcome(ctx: PredictionContext | null): Message {
  if (ctx) {
    const diseases = ctx.detected_diseases.length > 0
      ? ctx.detected_diseases.join(", ")
      : "no specific disease above threshold";
    return {
      id: "welcome",
      role: "assistant",
      timestamp: new Date(),
      content:
        `## Scan Analysis Ready\n\nHi! I'm **PneumoBot** 🩺 — your medical AI assistant.\n\n` +
        `**Primary finding:** ${ctx.primary_display} (${ctx.primary_probability.toFixed(1)}%)\n` +
        `**Detected conditions:** ${diseases}\n\n` +
        `Ask me anything about these findings, what they mean, or recommended next steps.`,
    };
  }
  return {
    id: "welcome",
    role: "assistant",
    timestamp: new Date(),
    content:
      `## Welcome to PneumoBot 🩺\n\nI'm your **medical AI assistant** specialised in pulmonary radiology.\n\n` +
      `I can help you understand:\n- Chest X-ray findings and disease classification\n` +
      `- Conditions like pneumonia, atelectasis, effusion, edema and more\n` +
      `- How this AI system works\n\n` +
      `*For medical emergencies, contact a professional immediately.*`,
  };
}

// ============================================================
// CONTEXT
// ============================================================

const ChatCtx = createContext<ChatContextValue | null>(null);

export function useChatContext() {
  const ctx = useContext(ChatCtx);
  if (!ctx) throw new Error("useChatContext must be used inside ChatProvider");
  return ctx;
}

// ============================================================
// PROVIDER
// ============================================================

export function ChatProvider({ children }: { children: ReactNode }) {
  const [messages, setMessages]               = useState<Message[]>([buildWelcome(null)]);
  const [open, setOpenState]                  = useState(false);
  const [minimised, setMinimised]             = useState(false);
  const [unread, setUnread]                   = useState(0);
  const [loading, setLoading]                 = useState(false);
  const [predictionContext, setPredCtx]       = useState<PredictionContext | null>(null);
  const [lastReply, setLastReply]             = useState<string | null>(null);
  // ── persisted across page navigation ──
  const [result, setResult]                   = useState<PredictionResult | null>(null);
  const [apiError, setApiError]               = useState<string | null>(null);

  // keep a stable ref so sendMessage closure always sees latest values
  const messagesRef        = useRef(messages);
  const predictionCtxRef   = useRef(predictionContext);
  const openRef            = useRef(open);
  const resultRef          = useRef(result);

  useEffect(() => { messagesRef.current = messages; }, [messages]);
  useEffect(() => { predictionCtxRef.current = predictionContext; }, [predictionContext]);
  useEffect(() => { openRef.current = open; }, [open]);
  useEffect(() => { resultRef.current = result; }, [result]);

  // stable function so sendMessage closure can read latest result
  const resultRefFn = useCallback(() => resultRef.current, []);

  const setOpen = useCallback((v: boolean) => {
    setOpenState(v);
    if (v) setUnread(0);
  }, []);

  // When predictionContext changes (new scan uploaded), update the welcome msg
  const setPredictionContext = useCallback((ctx: PredictionContext | null) => {
    setPredCtx(ctx);
    // Only update welcome if it's still the only message
    setMessages((prev) => {
      if (prev.length === 1 && prev[0].id === "welcome") {
        return [buildWelcome(ctx)];
      }
      return prev;
    });
  }, []);

  const clearChat = useCallback(() => {
    setLoading(false);
    setMessages([buildWelcome(predictionCtxRef.current)]);
    setLastReply(null);
    setUnread(0);
    setResult(null);
    setApiError(null);
    setPredCtx(null);
  }, []);

  const sendMessage = useCallback(async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || loading) return;

    const userMsg: Message = {
      id: generateId(),
      role: "user",
      content: trimmed,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    const ctx = predictionCtxRef.current;

    // ── Routing logic ──
    // ALWAYS use /chat endpoint for now (MedVQA adapter not yet trained).
    // The backend classifies the question and uses the correct system prompt:
    //   - Image-specific question → MedVQA-Bot persona (blue theme)
    //   - General question        → PneumoBot LLM persona (green theme)
    // Once MedVQA adapter is ready, flip MEDVQA_READY to true.
    const MEDVQA_READY = false;

    const isImageQuestion = (q: string): boolean => {
      const imageKeywords = [
        "this x-ray", "this image", "this scan", "this chest x-ray",
        "in the image", "in the scan", "in the x-ray",
        "left lung", "right lung",
        "highlighted", "heatmap", "overlay",
        "describe this image", "explain this image",
        "what is shown in", "what abnormality is visible",
        "what do you see in", "can you see in",
        "does this x-ray show", "based on the image",
      ];
      const q_lower = q.toLowerCase();
      return imageKeywords.some((kw) => q_lower.includes(kw));
    };

    const currentResult  = resultRefFn();
    const shouldUseMedVQA =
      MEDVQA_READY &&
      ctx !== null &&
      currentResult?.original_image != null &&
      isImageQuestion(trimmed);

    try {
      if (shouldUseMedVQA) {
        // ── Real MedVQA route (only when adapter is trained & ready) ──
        const byteStr = atob(currentResult!.original_image);
        const arr     = new Uint8Array(byteStr.length);
        for (let i = 0; i < byteStr.length; i++) arr[i] = byteStr.charCodeAt(i);
        const blob    = new Blob([arr], { type: "image/png" });

        const formData = new FormData();
        formData.append("file",          blob,    "xray.png");
        formData.append("question",      trimmed);
        formData.append("question_type", "auto");

        const res = await fetch(`${API_URL}/medvqa`, { method: "POST", body: formData });

        if (!res.ok) {
          // Any failure → fall back to /chat silently
          console.warn("[MedVQA] Endpoint failed, falling back to /chat");
          return await sendViaChat(trimmed, ctx);
        }

        const data = await res.json();
        const replyMsg: Message = {
          id:        generateId(),
          role:      "assistant",
          content:   data.answer,
          timestamp: new Date(),
          source:    "medvqa",
        };
        setMessages((prev) => [...prev, replyMsg]);
        if (!openRef.current) { setUnread((u) => u + 1); setLastReply(data.answer); }

      } else {
        // ── Always /chat route (MedVQA simulation via LLM) ──
        await sendViaChat(trimmed, ctx);
      }

    } catch (err: unknown) {
      const detail = err instanceof Error ? err.message : "Something went wrong.";
      const errMsg: Message = {
        id:        generateId(),
        role:      "assistant",
        content:   `⚠️ ${detail}`,
        timestamp: new Date(),
        error:     true,
      };
      setMessages((prev) => [...prev, errMsg]);
      if (!openRef.current) {
        setUnread((u) => u + 1);
        setLastReply(`⚠️ ${detail}`);
      }
    } finally {
      setLoading(false);
    }
  }, [loading]); // eslint-disable-line

  // Internal helper — send via Groq /chat
  const sendViaChat = useCallback(async (
    trimmed: string,
    ctx: PredictionContext | null,
  ) => {
    const history = messagesRef.current
      .filter((m) => m.id !== "welcome" && !m.error)
      .map((m) => ({ role: m.role, content: m.content }));

    const res = await fetch(`${API_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message:             trimmed,
        history,
        primary_label:       ctx?.primary_label       ?? null,
        primary_display:     ctx?.primary_display     ?? null,
        primary_probability: ctx?.primary_probability ?? null,
        detected_diseases:   ctx?.detected_diseases   ?? null,
        has_image: ctx != null,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err?.detail ?? `Server error ${res.status}`);
    }

    const data = await res.json();
    const replyMsg: Message = {
      id:        generateId(),
      role:      "assistant",
      content:   data.reply,
      timestamp: new Date(),
      source:    (data.source === "medvqa" ? "medvqa_simulation" : "llm") as "llm" | "medvqa_simulation",
    };
    setMessages((prev) => [...prev, replyMsg]);
    if (!openRef.current) {
      setUnread((u) => u + 1);
      setLastReply(data.reply);
    }
  }, []);

  const value: ChatContextValue = {
    messages, setMessages,
    open, setOpen,
    minimised, setMinimised,
    unread, setUnread,
    loading, setLoading,
    predictionContext, setPredictionContext,
    sendMessage, clearChat,
    lastReply, setLastReply,
    result, setResult,
    apiError, setApiError,
  };

  return <ChatCtx.Provider value={value}>{children}</ChatCtx.Provider>;
}
