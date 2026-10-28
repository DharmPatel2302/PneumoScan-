"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
  MessageCircle, X, Send, Minus, Bot, User,
  AlertCircle, Stethoscope, RotateCcw, Sparkles, Loader2, ChevronUp,
} from "lucide-react";
import { useChatContext, type Message } from "@/context/ChatContext";

// ============================================================
// CONSTANTS
// ============================================================

const SUGGESTED_WITH_RESULT = [
  "What does this result mean?",
  "What should I do next?",
  "Explain the Grad-CAM heatmap",
  "How accurate is the model?",
];

const SUGGESTED_GENERAL = [
  "What is pneumonia?",
  "What causes pneumonia?",
  "How is chest X-ray read?",
  "What are pneumonia symptoms?",
];

// ============================================================
// HELPERS
// ============================================================

function formatTime(d: Date) {
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

// Truncate a reply for the notification toast
function truncate(text: string, max = 80) {
  const clean = text.replace(/#+\s*/g, "").replace(/\*+/g, "").trim();
  return clean.length > max ? clean.slice(0, max) + "…" : clean;
}

// ============================================================
// RICH MARKDOWN RENDERER
// ============================================================

function RichMarkdown({ text }: { text: string }) {
  const lines = text.split("\n");
  const nodes: React.ReactNode[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (line.trim() === "") {
      nodes.push(<div key={i} className="h-1.5" />);
      i++; continue;
    }
    if (line.startsWith("## ")) {
      nodes.push(
        <p key={i} className="text-green-400 font-bold text-xs uppercase tracking-widest mt-3 mb-1">
          {line.slice(3)}
        </p>
      );
      i++; continue;
    }
    if (line.startsWith("# ")) {
      nodes.push(
        <p key={i} className="text-white font-bold text-sm mt-2 mb-1">{line.slice(2)}</p>
      );
      i++; continue;
    }
    if (line.startsWith(":::")) {
      const tag = line.slice(3).trim().toUpperCase();
      const styles: Record<string, string> = {
        HIGH:   "bg-red-500/20 text-red-300 border-red-500/30",
        MEDIUM: "bg-yellow-500/20 text-yellow-300 border-yellow-500/30",
        LOW:    "bg-green-500/20 text-green-300 border-green-500/30",
        INFO:   "bg-blue-500/20 text-blue-300 border-blue-500/30",
      };
      const cls = styles[tag] ?? "bg-white/10 text-gray-300 border-white/20";
      nodes.push(
        <span key={i} className={`inline-flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded-full border ${cls} my-1`}>
          {tag}
        </span>
      );
      i++; continue;
    }
    if (line.startsWith("|")) {
      const tableLines: string[] = [];
      while (i < lines.length && lines[i].startsWith("|")) { tableLines.push(lines[i]); i++; }
      const rows = tableLines.filter((r) => !r.match(/^\|[\s\-|]+\|$/));
      if (rows.length > 0) {
        const header = rows[0].split("|").filter((c) => c.trim() !== "");
        const body   = rows.slice(1);
        nodes.push(
          <div key={`table-${i}`} className="overflow-x-auto my-2 rounded-lg border border-white/10">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-green-900/30">
                  {header.map((h, hi) => (
                    <th key={hi} className="px-3 py-2 text-left text-green-300 font-semibold whitespace-nowrap">{h.trim()}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {body.map((row, ri) => {
                  const cells = row.split("|").filter((c) => c.trim() !== "");
                  return (
                    <tr key={ri} className={ri % 2 === 0 ? "bg-white/5" : ""}>
                      {cells.map((cell, ci) => (
                        <td key={ci} className="px-3 py-1.5 text-gray-300 whitespace-nowrap">
                          <InlineMarkdown text={cell.trim()} />
                        </td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        );
      }
      continue;
    }
    if (line.match(/^[-*•]\s/)) {
      const bulletLines: string[] = [];
      while (i < lines.length && lines[i].match(/^[-*•]\s/)) {
        bulletLines.push(lines[i].replace(/^[-*•]\s/, "")); i++;
      }
      nodes.push(
        <ul key={`ul-${i}`} className="my-2 space-y-1.5">
          {bulletLines.map((b, bi) => (
            <li key={bi} className="flex gap-2 text-gray-300 text-[13px] leading-relaxed">
              <span className="mt-[6px] w-1.5 h-1.5 rounded-full bg-green-500 flex-shrink-0" />
              <span className="flex-1"><InlineMarkdown text={b} /></span>
            </li>
          ))}
        </ul>
      );
      continue;
    }
    if (line.match(/^\d+\.\s/)) {
      const numLines: string[] = [];
      while (i < lines.length && lines[i].match(/^\d+\.\s/)) {
        numLines.push(lines[i].replace(/^\d+\.\s/, "")); i++;
      }
      nodes.push(
        <ol key={`ol-${i}`} className="my-2 space-y-1.5">
          {numLines.map((b, bi) => (
            <li key={bi} className="flex gap-2 text-gray-300 text-[13px] leading-relaxed">
              <span className="text-[11px] font-bold text-green-400 flex-shrink-0 w-5 text-right mt-0.5">{bi + 1}.</span>
              <span className="flex-1"><InlineMarkdown text={b} /></span>
            </li>
          ))}
        </ol>
      );
      continue;
    }
    if (line.trim() === "---") {
      nodes.push(<hr key={i} className="border-white/10 my-2" />);
      i++; continue;
    }
    nodes.push(
      <p key={i} className="text-gray-200 text-[13px] leading-relaxed">
        <InlineMarkdown text={line} />
      </p>
    );
    i++;
  }

  return <div className="space-y-0.5">{nodes}</div>;
}

function InlineMarkdown({ text }: { text: string }) {
  const parts = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g);
  return (
    <>
      {parts.map((part, i) => {
        if (part.startsWith("**") && part.endsWith("**"))
          return <strong key={i} className="text-white font-semibold">{part.slice(2, -2)}</strong>;
        if (part.startsWith("*") && part.endsWith("*"))
          return <em key={i} className="text-gray-300 italic">{part.slice(1, -1)}</em>;
        if (part.startsWith("`") && part.endsWith("`"))
          return <code key={i} className="bg-white/15 px-1.5 py-0.5 rounded text-[11px] font-mono text-green-300">{part.slice(1, -1)}</code>;
        return <span key={i}>{part}</span>;
      })}
    </>
  );
}

// ============================================================
// TYPING INDICATOR
// ============================================================

function TypingIndicator() {
  return (
    <div className="flex items-end gap-2">
      <div className="w-7 h-7 rounded-full bg-green-700 flex items-center justify-center flex-shrink-0">
        <Bot size={13} className="text-white" />
      </div>
      <div className="bg-[#2a2a2a] border border-white/10 rounded-2xl rounded-bl-sm px-4 py-3">
        <div className="flex items-center gap-1.5">
          {[0, 1, 2].map((i) => (
            <span key={i} className="w-1.5 h-1.5 bg-green-400 rounded-full animate-bounce"
              style={{ animationDelay: `${i * 150}ms` }} />
          ))}
        </div>
      </div>
    </div>
  );
}

// ============================================================
// MESSAGE BUBBLE
// ============================================================

function SourceBadge({ source }: { source?: string }) {
  if (!source || source === "llm") {
    return (
      <span className="inline-flex items-center gap-1 text-[9px] font-bold px-1.5 py-0.5 rounded-full bg-green-500/15 text-green-400 border border-green-500/20 mt-1.5">
        <span className="w-1 h-1 rounded-full bg-green-400 inline-block" />
        PneumoBot LLM
      </span>
    );
  }
  if (source === "medvqa_simulation" || source === "medvqa") {
    return (
      <span className="inline-flex items-center gap-1 text-[9px] font-bold px-1.5 py-0.5 rounded-full bg-blue-500/15 text-blue-400 border border-blue-500/20 mt-1.5">
        <span className="w-1 h-1 rounded-full bg-blue-400 inline-block animate-pulse" />
        MedVQA Vision Model
      </span>
    );
  }
  return null;
}

function MessageBubble({ msg }: { msg: Message }) {
  const isUser    = msg.role === "user";
  const isMedVQA  = msg.source === "medvqa_simulation" || msg.source === "medvqa";

  // Bubble colour scheme
  const assistantBg = isMedVQA
    ? "bg-[#0f1e35] border border-blue-500/30"   // dark blue for MedVQA
    : "bg-[#252525] border border-white/10";       // dark grey for LLM

  return (
    <div className={`flex items-end gap-2 ${isUser ? "flex-row-reverse" : "flex-row"}`}>
      {/* Avatar */}
      <div className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 self-start mt-1 ${
        isUser
          ? "bg-green-500"
          : isMedVQA
            ? "bg-blue-600"
            : "bg-green-700"
      }`}>
        {isUser
          ? <User size={13} className="text-white" />
          : isMedVQA
            ? <span className="text-[9px] font-black text-white">VQA</span>
            : <Bot size={13} className="text-white" />
        }
      </div>

      {/* Bubble */}
      <div className={`max-w-[82%] rounded-2xl px-4 py-3 text-sm ${
        isUser
          ? "bg-green-500 text-white rounded-br-sm"
          : msg.error
          ? "bg-red-900/40 border border-red-500/30 rounded-bl-sm"
          : `${assistantBg} rounded-bl-sm`
      }`}>
        {isUser
          ? <p className="text-white text-[13px] leading-relaxed">{msg.content}</p>
          : <RichMarkdown text={msg.content} />
        }

        {/* Source badge + timestamp row */}
        <div className={`flex items-center justify-between gap-2 mt-1.5 ${isUser ? "flex-row-reverse" : "flex-row"}`}>
          {!isUser && <SourceBadge source={msg.source} />}
          <p className={`text-[10px] flex-shrink-0 ${isUser ? "text-green-100/60" : "text-gray-600"}`}>
            {formatTime(msg.timestamp)}
          </p>
        </div>
      </div>
    </div>
  );
}

// ============================================================
// NOTIFICATION TOAST
// ============================================================

function NotificationToast({ text, onClick }: { text: string; onClick: () => void }) {
  return (
    <div
      onClick={onClick}
      className="absolute bottom-16 right-0 w-64 bg-[#1e1e1e] border border-white/15 rounded-2xl shadow-2xl px-4 py-3 cursor-pointer hover:bg-[#252525] transition-colors"
      style={{ animation: "fadeScaleIn 0.2s ease-out" }}
    >
      <div className="flex items-center gap-2 mb-1">
        <div className="w-5 h-5 rounded-full bg-green-700 flex items-center justify-center flex-shrink-0">
          <Bot size={11} className="text-white" />
        </div>
        <p className="text-green-400 text-[11px] font-semibold">PneumoBot replied</p>
      </div>
      <p className="text-gray-300 text-[12px] leading-snug line-clamp-2">{text}</p>
    </div>
  );
}

// ============================================================
// MAIN CHATBOT COMPONENT
// ============================================================

export default function ChatBot() {
  const {
    messages, open, setOpen,
    minimised, setMinimised,
    unread, loading,
    predictionContext,
    sendMessage, clearChat,
    lastReply, setLastReply,
  } = useChatContext();

  const [input, setInput] = useState("");
  const messagesEndRef             = useRef<HTMLDivElement>(null);
  const inputRef                   = useRef<HTMLTextAreaElement>(null);
  const chatWindowRef              = useRef<HTMLDivElement>(null);

  // Auto-scroll to latest message
  useEffect(() => {
    if (open && !minimised) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, open, minimised]);

  // Auto-dismiss notification after 5 s
  useEffect(() => {
    if (!lastReply) return;
    const t = setTimeout(() => setLastReply(null), 5000);
    return () => clearTimeout(t);
  }, [lastReply, setLastReply]);

  // Click-outside → close
  useEffect(() => {
    if (!open || minimised) return;
    const handler = (e: MouseEvent) => {
      if (chatWindowRef.current && !chatWindowRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    const t = setTimeout(() => document.addEventListener("mousedown", handler), 100);
    return () => { clearTimeout(t); document.removeEventListener("mousedown", handler); };
  }, [open, minimised, setOpen]);

  // Escape key → close
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, setOpen]);

  const handleOpen = () => {
    setOpen(true);
    setMinimised(false);
    setLastReply(null);
    setTimeout(() => inputRef.current?.focus(), 120);
  };

  const handleSend = useCallback(() => {
    if (!input.trim()) return;
    sendMessage(input);
    setInput("");
  }, [input, sendMessage]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  const suggested = predictionContext ? SUGGESTED_WITH_RESULT : SUGGESTED_GENERAL;
  const showSuggestions = messages.length === 1 && !loading;

  return (
    <>
      {/* ── Floating Bubble ── */}
      {!open && (
        <div className="fixed bottom-6 right-6 z-50 flex flex-col items-end gap-2">
          {/* Notification toast */}
          {lastReply && (
            <NotificationToast text={truncate(lastReply)} onClick={handleOpen} />
          )}

          <button
            onClick={handleOpen}
            className="relative w-14 h-14 bg-green-500 hover:bg-green-600 text-white rounded-full shadow-2xl flex items-center justify-center transition-all duration-200 hover:scale-110 active:scale-95"
            aria-label="Open PneumoBot"
          >
            <MessageCircle size={24} />
            {unread > 0 && (
              <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 text-white text-[10px] font-bold rounded-full flex items-center justify-center">
                {unread > 9 ? "9+" : unread}
              </span>
            )}
          </button>
        </div>
      )}

      {/* ── Chat Window ── */}
      {open && (
        <div
          ref={chatWindowRef}
          className={`fixed bottom-6 right-6 z-50 w-[390px] bg-[#1a1a1a] rounded-2xl shadow-2xl border border-white/10 flex flex-col overflow-hidden transition-all duration-300 ${
            minimised ? "h-14" : "h-[600px]"
          }`}
          style={{ maxHeight: "calc(100vh - 48px)" }}
        >
          {/* ── Header ── */}
          <div className="flex items-center gap-3 px-4 h-14 border-b border-white/10 flex-shrink-0 bg-[#1a1a1a]">
            <div className="relative">
              <div className="w-9 h-9 rounded-full bg-gradient-to-br from-green-500 to-green-700 flex items-center justify-center flex-shrink-0">
                <Stethoscope size={16} className="text-white" />
              </div>
              <span className="absolute bottom-0 right-0 w-2.5 h-2.5 bg-green-400 border-2 border-[#1a1a1a] rounded-full" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-1.5">
                <p className="text-white font-semibold text-sm leading-tight">PneumoBot</p>
                <Sparkles size={11} className="text-green-400" />
              </div>
              <p className={`text-[10px] leading-tight transition-colors ${loading ? "text-yellow-400" : "text-green-400"}`}>
                {loading ? "Generating response..." : "Medical AI Assistant • Online"}
              </p>
            </div>
            <div className="flex items-center gap-1">
              <button onClick={clearChat} title="Clear chat"
                className="w-7 h-7 rounded-lg hover:bg-white/10 flex items-center justify-center text-gray-400 hover:text-white transition-colors">
                <RotateCcw size={13} />
              </button>
              <button onClick={() => setMinimised(!minimised)} title={minimised ? "Expand" : "Minimise"}
                className="w-7 h-7 rounded-lg hover:bg-white/10 flex items-center justify-center text-gray-400 hover:text-white transition-colors">
                {minimised ? <ChevronUp size={13} /> : <Minus size={13} />}
              </button>
              <button onClick={() => setOpen(false)} title="Close"
                className="w-7 h-7 rounded-lg hover:bg-white/10 flex items-center justify-center text-gray-400 hover:text-white transition-colors">
                <X size={13} />
              </button>
            </div>
          </div>

          {/* ── Context Pill ── */}
          {!minimised && predictionContext && (
            <div className="flex-shrink-0 px-4 py-2 bg-green-950/50 border-b border-green-800/30">
              <div className="flex items-center gap-2">
                <AlertCircle size={11} className="text-green-400 flex-shrink-0" />
                <p className="text-[11px] text-green-300 leading-tight">
                  Active context:{" "}
                  <span className="font-bold text-green-400">
                    {predictionContext.primary_display}
                  </span>
                  {" "}— {predictionContext.primary_probability.toFixed(1)}%
                </p>
              </div>
            </div>
          )}

          {/* ── Messages ── */}
          {!minimised && (
            <>
              <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4 [&::-webkit-scrollbar]:w-[3px] [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-green-600 [&::-webkit-scrollbar-thumb]:rounded-full">
                {messages.map((msg) => (
                  <MessageBubble key={msg.id} msg={msg} />
                ))}
                {loading && <TypingIndicator />}
                {showSuggestions && (
                  <div className="flex flex-wrap gap-2 pt-1">
                    {suggested.map((q) => (
                      <button key={q} onClick={() => sendMessage(q)}
                        className="text-[11px] px-3 py-1.5 bg-white/5 hover:bg-green-600/20 border border-white/10 hover:border-green-500/40 text-gray-400 hover:text-green-300 rounded-full transition-all duration-150">
                        {q}
                      </button>
                    ))}
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              <div className="flex-shrink-0 px-4 py-1.5 bg-[#141414] border-t border-white/5">
                <p className="text-[9px] text-gray-600 text-center">
                  ⚠️ Educational purposes only — not a substitute for clinical advice
                </p>
              </div>

              {/* ── Input ── */}
              <div className="flex-shrink-0 px-3 pb-3 pt-2 bg-[#1a1a1a] border-t border-white/10">
                <div className="flex items-end gap-2 bg-[#252525] border border-white/10 rounded-xl px-3 py-2.5 focus-within:border-green-500/60 transition-colors">
                  <textarea
                    ref={inputRef}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Ask about your results..."
                    rows={1}
                    disabled={loading}
                    className="flex-1 bg-transparent text-gray-200 text-sm placeholder-gray-600 resize-none outline-none leading-relaxed max-h-28 overflow-y-auto disabled:opacity-40"
                    style={{ minHeight: "22px" }}
                    onInput={(e) => {
                      const el = e.currentTarget;
                      el.style.height = "22px";
                      el.style.height = `${Math.min(el.scrollHeight, 112)}px`;
                    }}
                  />
                  <button
                    onClick={handleSend}
                    disabled={loading || !input.trim()}
                    className="w-8 h-8 bg-green-500 hover:bg-green-600 disabled:bg-green-500/25 disabled:cursor-not-allowed text-white rounded-lg flex items-center justify-center transition-colors flex-shrink-0"
                  >
                    {loading
                      ? <Loader2 size={14} className="animate-spin" />
                      : <Send size={14} />
                    }
                  </button>
                </div>
                <p className="text-[10px] text-gray-700 mt-1.5 text-center">
                  Enter to send • Shift+Enter for new line
                </p>
              </div>
            </>
          )}
        </div>
      )}
    </>
  );
}

