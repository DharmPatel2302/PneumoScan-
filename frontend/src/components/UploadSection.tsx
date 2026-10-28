"use client";

import { useCallback, useRef, useState } from "react";
import { FileText, X, Microscope } from "lucide-react";

interface SelectedFile {
  file: File;
  preview: string;
}

interface UploadSectionProps {
  onAnalyze: (file: File) => void;
  loading: boolean;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export default function UploadSection({ onAnalyze, loading }: UploadSectionProps) {
  const [selected, setSelected] = useState<SelectedFile | null>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const validate = (file: File): string | null => {
    const allowed = ["image/jpeg", "image/jpg", "image/png"];
    if (!allowed.includes(file.type)) return "Only JPEG and PNG files are supported.";
    if (file.size > 10 * 1024 * 1024) return "File size must be under 10 MB.";
    return null;
  };

  const handleFile = useCallback((file: File) => {
    const err = validate(file);
    if (err) { setError(err); return; }
    setError(null);
    const preview = URL.createObjectURL(file);
    setSelected({ file, preview });
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const clearFile = () => {
    if (selected) URL.revokeObjectURL(selected.preview);
    setSelected(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  const fileExt = selected?.file.name.split(".").pop()?.toUpperCase() ?? "";

  return (
    <div className="w-full">

      {/* ── Drop Zone ── */}
      {!selected && (
        <div
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
          className={`
            w-full rounded-2xl cursor-pointer select-none
            border-2 border-dashed border-green-400
            bg-[#f0faf0]
            flex flex-col items-center justify-center gap-3
            py-20 px-8
            transition-all duration-200
            ${dragging ? "border-green-500 bg-green-80" : "hover:border-green-600 hover:bg-green-90/70"}
          `}
        >
          {/* File icon — matches reference */}
          <div className="w-12 h-12 bg-white rounded-full border border-gray-200 shadow-sm flex items-center justify-center">
            <FileText size={22} className="text-gray-400" />
          </div>

          <div className="text-center">
            <p className="font-semibold text-gray-800 text-base">
              Drag &amp; drop or click to upload a chest X-ray
            </p>
            <p className="text-gray-400 text-sm mt-1">
              High-resolution JPEG or PNG supported (max 10MB)
            </p>
          </div>

          <input
            ref={inputRef}
            type="file"
            accept="image/jpeg,image/png"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleFile(file);
            }}
          />
        </div>
      )}

      {/* ── Error ── */}
      {error && (
        <p className="mt-3 text-sm text-red-600 text-center font-medium">{error}</p>
      )}

      {/* ── File Preview Card ── */}
      {selected && (
        <div className="bg-white border border-gray-200 rounded-2xl p-4 shadow-sm flex items-center gap-4 animate-fade-in">
          {/* Thumbnail */}
          <div className="relative w-16 h-16 rounded-xl overflow-hidden flex-shrink-0 bg-gray-100 border border-gray-200">
            <img
              src={selected.preview}
              alt="preview"
              className="w-full h-full object-cover"
            />
            <button
              onClick={clearFile}
              disabled={loading}
              className="absolute top-0.5 right-0.5 w-5 h-5 bg-red-500 rounded-full flex items-center justify-center text-white hover:bg-red-600 transition-colors disabled:opacity-50"
            >
              <X size={11} strokeWidth={3} />
            </button>
          </div>

          {/* File info */}
          <div className="flex-1 min-w-0">
            <p className="text-[10px] font-semibold tracking-widest text-green-600 uppercase mb-0.5">
              Status: Image Ready
            </p>
            <p className="text-sm font-semibold text-gray-900 truncate">
              {selected.file.name}
            </p>
            <p className="text-xs text-gray-400 mt-0.5">
              {formatBytes(selected.file.size)} • {fileExt}
            </p>
          </div>

          {/* Analyze button */}
          <button
            onClick={() => onAnalyze(selected.file)}
            disabled={loading}
            className="flex items-center gap-2 px-5 py-2.5 bg-green-500 hover:bg-green-600 disabled:bg-green-300 text-white text-sm font-bold rounded-xl transition-colors flex-shrink-0 shadow-sm"
          >
            <Microscope size={15} />
            Analyze X-ray
          </button>
        </div>
      )}
    </div>
  );
}
