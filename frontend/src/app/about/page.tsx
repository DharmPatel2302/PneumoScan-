import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import { Brain, Layers, ShieldCheck, Zap } from "lucide-react";

const features = [
  {
    icon: Brain,
    title: "DenseNet121 Architecture",
    description:
      "Our model uses DenseNet121, a deep convolutional neural network pre-trained on ImageNet, fine-tuned on chest X-ray datasets for binary pneumonia classification.",
  },
  {
    icon: Zap,
    title: "Grad-CAM Explainability",
    description:
      "Gradient-weighted Class Activation Mapping (Grad-CAM) generates visual explanations showing which regions of the X-ray the model focused on for its decision.",
  },
  {
    icon: Layers,
    title: "Full-Stack Architecture",
    description:
      "FastAPI backend handles model inference and XAI generation. Next.js 14 frontend delivers a responsive, accessible UI with real-time feedback.",
  },
  {
    icon: ShieldCheck,
    title: "Research Purpose",
    description:
      "This system is developed for academic and research exploration. Results should not replace clinical radiology review by a licensed professional.",
  },
];

export default function AboutPage() {
  return (
    <div className="flex flex-col min-h-screen bg-[#f5f5f0]">
      <Navbar active="About" />

      <main className="flex-1 w-full max-w-4xl mx-auto px-6 py-16">
        {/* ── Header ── */}
        <div className="mb-12">
          <p className="text-[11px] font-semibold tracking-widest text-gray-500 uppercase mb-3">
            About the Project
          </p>
          <h1 className="text-4xl sm:text-5xl font-black text-gray-900 tracking-tight leading-tight">
            AI-Powered<br />
            <span className="text-green-500">Pneumonia Detection</span>
          </h1>
          <p className="text-gray-500 mt-4 text-base max-w-xl leading-relaxed">
            PneumoScan combines state-of-the-art deep learning with explainable AI
            to provide transparent, interpretable chest X-ray analysis.
          </p>
        </div>

        {/* ── Feature Cards ── */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-5 mb-14">
          {features.map(({ icon: Icon, title, description }) => (
            <div
              key={title}
              className="bg-white border border-gray-200 rounded-2xl p-6 shadow-sm hover:shadow-md transition-shadow"
            >
              <div className="w-10 h-10 bg-green-50 rounded-xl flex items-center justify-center mb-4">
                <Icon size={20} className="text-green-600" />
              </div>
              <h3 className="font-bold text-gray-900 mb-2 text-base">{title}</h3>
              <p className="text-sm text-gray-500 leading-relaxed">{description}</p>
            </div>
          ))}
        </div>

        {/* ── Technical Details ── */}
        <div className="bg-[#1c1c1c] rounded-2xl p-8 text-white">
          <h2 className="text-xl font-bold mb-6 text-green-400">Technical Stack</h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
            {[
              ["Model",       "DenseNet121"],
              ["Framework",   "PyTorch"],
              ["XAI",         "Grad-CAM"],
              ["Backend",     "FastAPI"],
              ["Frontend",    "Next.js 14"],
              ["Styling",     "Tailwind CSS"],
            ].map(([label, value]) => (
              <div key={label}>
                <p className="text-[10px] uppercase tracking-widest text-gray-500 mb-1">{label}</p>
                <p className="text-sm font-semibold text-white">{value}</p>
              </div>
            ))}
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}
