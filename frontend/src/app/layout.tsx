import type { Metadata } from "next";
import "./globals.css";
import { ChatProvider } from "@/context/ChatContext";
import ChatBot from "@/components/ChatBot";

export const metadata: Metadata = {
  title: "PneumoScan — AI Chest X-Ray Analysis",
  description:
    "High-precision computer vision to diagnose pneumonia with clinical accuracy using DenseNet121 and Grad-CAM explainability.",
  icons: { icon: "/favicon.ico" },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-[#f5f5f0]">
        <ChatProvider>
          {children}
          {/* ChatBot lives here so it persists across all pages */}
          <ChatBot />
        </ChatProvider>
      </body>
    </html>
  );
}
