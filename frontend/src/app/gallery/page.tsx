import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import GalleryGrid from "@/components/GalleryGrid";
import { ImageIcon } from "lucide-react";
import galleryItems from "@/data/gallery.config";

export default function GalleryPage() {
  const hasImages = galleryItems.length > 0;

  return (
    <div className="flex flex-col min-h-screen bg-[#f5f5f0]">
      <Navbar active="Gallery" />

      <main className="flex-1 w-full max-w-6xl mx-auto px-6 py-16">
        {/* ── Header ── */}
        <div className="mb-10">
          <p className="text-[11px] font-semibold tracking-widest text-gray-500 uppercase mb-3">
            Example Diagnoses
          </p>
          <h1 className="text-4xl font-black text-gray-900 tracking-tight">
            Case <span className="text-green-500">Gallery</span>
          </h1>
          <p className="text-gray-500 text-sm mt-3 max-w-lg leading-relaxed">
            Browse example chest X-ray analyses. Click any image to view full
            size and download. Upload your own on the{" "}
            <a href="/" className="text-green-600 hover:underline">
              Home page
            </a>{" "}
            for a live prediction.
          </p>
        </div>

        {/* ── Add-images guide banner ── */}
        {!hasImages && (
          <div className="mb-8 bg-amber-50 border border-amber-200 rounded-2xl p-5 flex items-start gap-4">
            <div className="w-9 h-9 bg-amber-100 rounded-xl flex items-center justify-center flex-shrink-0 mt-0.5">
              <ImageIcon size={18} className="text-amber-600" />
            </div>
            <div>
              <p className="text-sm font-semibold text-amber-800 mb-1">
                No images added yet
              </p>
              <p className="text-xs text-amber-700 leading-relaxed">
                Drop your X-ray images into{" "}
                <code className="bg-amber-100 px-1.5 py-0.5 rounded font-mono text-[11px]">
                  frontend/public/gallery/
                </code>{" "}
                then register them in{" "}
                <code className="bg-amber-100 px-1.5 py-0.5 rounded font-mono text-[11px]">
                  src/data/gallery.config.ts
                </code>
                . They will appear here automatically.
              </p>
            </div>
          </div>
        )}

        {/* ── Grid or empty placeholder ── */}
        {hasImages ? (
          <GalleryGrid items={galleryItems} />
        ) : (
          <PlaceholderGrid />
        )}
      </main>

      <Footer />
    </div>
  );
}

/* ── Shows ghost cards when no real images are added yet ── */
function PlaceholderGrid() {
  const placeholders = [
    { label: "Pneumonia Case",  type: "PNEUMONIA", confidence: "94.2%", description: "Bilateral infiltrates visible in lower lobes." },
    { label: "Normal Case",     type: "NORMAL",    confidence: "91.7%", description: "Clear lung fields with no consolidation." },
    { label: "Pneumonia Case",  type: "PNEUMONIA", confidence: "87.3%", description: "Right lower lobe opacity suggesting lobar pneumonia." },
    { label: "Normal Case",     type: "NORMAL",    confidence: "96.1%", description: "Well-aerated lungs, normal cardiac silhouette." },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-2 gap-6">
      {placeholders.map((c, i) => (
        <div
          key={i}
          className="bg-white border border-gray-200 rounded-2xl overflow-hidden shadow-sm"
        >
          {/* Placeholder image area */}
          <div className="w-full h-52 bg-gray-100 flex items-center justify-center border-b border-gray-100">
            <div className="text-center">
              <ImageIcon size={36} className="text-gray-300 mx-auto mb-2" />
              <p className="text-xs text-gray-400">Sample X-ray</p>
            </div>
          </div>
          {/* Info */}
          <div className="p-5">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-bold text-gray-900">{c.label}</span>
              <span
                className={`text-[10px] font-bold px-2.5 py-1 rounded-full uppercase tracking-wide ${
                  c.type === "PNEUMONIA"
                    ? "bg-red-100 text-red-600"
                    : "bg-green-100 text-green-600"
                }`}
              >
                {c.type}
              </span>
            </div>
            <p className="text-xs text-gray-500 mb-4 leading-relaxed">
              {c.description}
            </p>
            <div className="flex items-center justify-between border-t border-gray-100 pt-3">
              <span className="text-[10px] uppercase tracking-widest text-gray-400">
                Confidence
              </span>
              <span className="text-sm font-bold text-gray-700">
                {c.confidence}
              </span>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
