import Link from "next/link";

export default function Footer() {
  return (
    <footer className="w-full bg-[#1c1c1c] mt-auto">
      <div className="max-w-7xl mx-auto px-6 py-5 flex flex-col md:flex-row items-center justify-between gap-3">
        {/* ── Brand ── */}
        <div className="flex items-center gap-1">
          <span className="text-white font-bold text-sm">Pneumo</span>
          <span className="text-green-400 font-bold text-sm">Scan</span>
          <span className="text-gray-500 text-xs ml-2 hidden sm:inline">
            Clinical AI
          </span>
          <span className="text-gray-600 text-xs ml-1 hidden sm:inline">
            © {new Date().getFullYear()} PNEUMOSCAN CLINICAL AI. ALL RIGHTS RESERVED.
          </span>
        </div>

        {/* ── Links ── */}
        <div className="flex items-center gap-6">
          <Link
            href="/privacy"
            className="text-gray-500 text-xs hover:text-gray-300 transition-colors uppercase tracking-wide"
          >
            Privacy Policy
          </Link>
          <Link
            href="/terms"
            className="text-gray-500 text-xs hover:text-gray-300 transition-colors uppercase tracking-wide"
          >
            Terms of Service
          </Link>
          <a
            href="mailto:contact@pneumoscan.ai"
            className="text-green-500 text-xs hover:text-green-300 transition-colors uppercase tracking-wide"
          >
            Contact Science Team
          </a>
        </div>
      </div>
    </footer>
  );
}
