"use client";

import { useState } from "react";
import { Download, X, ExternalLink, AlertTriangle } from "lucide-react";
import type { GalleryItem } from "@/data/gallery.config";
import { resolveDescription } from "@/data/gallery.config";

interface GalleryGridProps {
  items: GalleryItem[];
}

interface LightboxState {
  item: GalleryItem;
  src: string;
}

export default function GalleryGrid({ items }: GalleryGridProps) {
  const [lightbox, setLightbox] = useState<LightboxState | null>(null);

  const openLightbox = (item: GalleryItem) => {
    setLightbox({ item, src: `/gallery/${item.filename}` });
  };

  const closeLightbox = () => setLightbox(null);

  const handleDownload = (src: string, filename: string) => {
    const a = document.createElement("a");
    a.href = src;
    a.download = filename;
    a.click();
  };

  return (
    <>
      {/* ── Card Grid ── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-2 gap-6">
        {items.map((item, i) => (
          <GalleryCard
            key={i}
            item={item}
            onClick={() => openLightbox(item)}
          />
        ))}
      </div>

      {/* ── Lightbox ── */}
      {lightbox && (
        <Lightbox
          item={lightbox.item}
          src={lightbox.src}
          onClose={closeLightbox}
          onDownload={() =>
            handleDownload(lightbox.src, lightbox.item.filename)
          }
        />
      )}
    </>
  );
}

/* ── Individual Gallery Card ── */
function GalleryCard({
  item,
  onClick,
}: {
  item: GalleryItem;
  onClick: () => void;
}) {
  const isPneumonia = item.type === "PNEUMONIA";
  const src = `/gallery/${item.filename}`;

  return (
    <div className="bg-white border border-gray-200 rounded-2xl overflow-hidden shadow-sm hover:shadow-md transition-shadow group">
      {/* Image area */}
      <div
        className="relative w-full h-56 bg-gray-100 overflow-hidden cursor-pointer"
        onClick={onClick}
      >
        {/* Badge top-left */}
        <div
          className={`absolute top-3 left-3 z-10 flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-wide shadow-sm ${
            isPneumonia
              ? "bg-red-500 text-white"
              : "bg-green-500 text-white"
          }`}
        >
          {isPneumonia && <AlertTriangle size={10} />}
          {item.type}
        </div>

        {/* Actual image */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={src}
          alt={item.label}
          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
        />

        {/* Hover overlay */}
        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/25 transition-colors duration-200 flex items-center justify-center">
          <span className="opacity-0 group-hover:opacity-100 transition-opacity duration-200 bg-white/90 text-gray-900 text-xs font-semibold px-3 py-1.5 rounded-full flex items-center gap-1.5">
            <ExternalLink size={12} />
            View Full Image
          </span>
        </div>
      </div>

      {/* Card body */}
      <div className="p-5">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-bold text-gray-900">{item.label}</span>
          <span
            className={`text-[10px] font-bold px-2.5 py-1 rounded-full uppercase tracking-wide ${
              isPneumonia
                ? "bg-red-100 text-red-600"
                : "bg-green-100 text-green-600"
            }`}
          >
            {item.type}
          </span>
        </div>
        <p className="text-xs text-gray-500 mb-4 leading-relaxed">
          {resolveDescription(item)}
        </p>

        {/* Bottom row */}
        <div className="flex items-center justify-between border-t border-gray-100 pt-3">
          <span className="text-[10px] uppercase tracking-widest text-gray-400">
            Confidence
          </span>
          <span className="text-sm font-bold text-gray-800">
            {item.confidence.toFixed(1)}%
          </span>
        </div>
      </div>
    </div>
  );
}

/* ── Lightbox Modal ── */
function Lightbox({
  item,
  src,
  onClose,
  onDownload,
}: {
  item: GalleryItem;
  src: string;
  onClose: () => void;
  onDownload: () => void;
}) {
  return (
    /* Backdrop */
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm"
      onClick={onClose}
    >
      {/* Modal panel — stop propagation so clicking inside doesn't close */}
      <div
        className="relative bg-[#1e1e1e] rounded-2xl shadow-2xl w-full max-w-xl overflow-hidden animate-fade-in"
        onClick={(e) => e.stopPropagation()}
      >
        {/* ── Header ── */}
        <div className="flex items-start justify-between px-5 py-4 border-b border-white/10">
          <div className="min-w-0 pr-4">
            <h3 className="text-white font-bold text-base leading-tight truncate">
              {item.label}
            </h3>
            <p className="text-gray-400 text-xs mt-0.5 truncate">
              {item.filename}
            </p>
          </div>

          <div className="flex items-center gap-2 flex-shrink-0">
            {/* Download button */}
            <button
              onClick={onDownload}
              title="Download image"
              className="w-8 h-8 rounded-full bg-white/10 hover:bg-white/20 flex items-center justify-center text-gray-300 hover:text-white transition-colors"
            >
              <Download size={15} />
            </button>
            {/* Close button */}
            <button
              onClick={onClose}
              title="Close"
              className="w-8 h-8 rounded-full bg-white/10 hover:bg-white/20 flex items-center justify-center text-gray-300 hover:text-white transition-colors"
            >
              <X size={15} />
            </button>
          </div>
        </div>

        {/* ── Image ── */}
        <div className="bg-black flex items-center justify-center p-4 min-h-64">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={src}
            alt={item.label}
            className="max-w-full max-h-[65vh] rounded-lg object-contain"
          />
        </div>

        {/* ── Footer ── */}
        <div className="px-5 py-3 flex items-center justify-between border-t border-white/10">
          <span
            className={`text-[10px] font-bold px-2.5 py-1 rounded-full uppercase tracking-wide ${
              item.type === "PNEUMONIA"
                ? "bg-red-500/20 text-red-400"
                : "bg-green-500/20 text-green-400"
            }`}
          >
            {item.type}
          </span>
          <span className="text-gray-400 text-xs">
            Confidence:{" "}
            <span className="text-white font-semibold">
              {item.confidence.toFixed(1)}%
            </span>
          </span>
        </div>
      </div>
    </div>
  );
}
