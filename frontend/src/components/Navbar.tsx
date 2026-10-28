"use client";

import { useState } from "react";
import Link from "next/link";
import { Menu, X } from "lucide-react";

const navLinks = [
  { label: "Home",    href: "/" },
  { label: "About",   href: "/about" },
  { label: "Gallery", href: "/gallery" },
];

export default function Navbar({ active = "Home" }: { active?: string }) {
  const [open, setOpen] = useState(false);

  return (
    <nav className="w-full bg-[#1c1c1c] sticky top-0 z-50 shadow-md">
      <div className="max-w-7xl mx-auto px-6 h-14 grid grid-cols-3 items-center">
        {/* ── Logo — left column ── */}
        <Link href="/" className="flex items-center gap-0.5 select-none justify-self-start">
          <span className="text-white font-bold text-lg tracking-tight">Pneumo</span>
          <span className="text-green-400 font-bold text-lg tracking-tight">Scan</span>
        </Link>

        {/* ── Desktop Links — center column ── */}
        <ul className="hidden md:flex items-center justify-center gap-8">
          {navLinks.map((link) => {
            const isActive = link.label === active;
            return (
              <li key={link.label}>
                <Link
                  href={link.href}
                  className={`relative text-sm font-medium pb-1 transition-colors ${
                    isActive ? "text-white" : "text-gray-400 hover:text-green-400"
                  }`}
                >
                  {link.label}
                  {isActive && (
                    <span className="absolute bottom-0 left-0 w-full h-[2px] bg-green-400 rounded-full" />
                  )}
                </Link>
              </li>
            );
          })}
        </ul>

        {/* ── Right column — hamburger on mobile, empty on desktop ── */}
        <div className="flex justify-end">
          <button
            className="md:hidden text-gray-300 hover:text-white transition-colors"
            onClick={() => setOpen((o) => !o)}
            aria-label="Toggle menu"
          >
            {open ? <X size={22} /> : <Menu size={22} />}
          </button>
        </div>
      </div>

      {/* ── Mobile Menu ── */}
      {open && (
        <div className="md:hidden bg-[#252525] border-t border-[#333] px-6 py-4 flex flex-col gap-4">
          {navLinks.map((link) => {
            const isActive = link.label === active;
            return (
              <Link
                key={link.label}
                href={link.href}
                onClick={() => setOpen(false)}
                className={`text-sm font-medium transition-colors ${
                  isActive ? "text-green-400" : "text-gray-400 hover:text-green-400"
                }`}
              >
                {link.label}
              </Link>
            );
          })}
        </div>
      )}
    </nav>
  );
}
