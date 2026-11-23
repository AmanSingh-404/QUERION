import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";

import Footer from "@/components/Footer";
import Nav from "@/components/Nav";

import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter", weight: ["300", "400", "500"] });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono", weight: ["400"] });

export const metadata: Metadata = {
  title: "querion",
  description: "Ask your data in plain English. A 101M-parameter text-to-SQL model trained from scratch.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${mono.variable}`}>
      <body>
        <Nav />
        {children}
        <Footer />
      </body>
    </html>
  );
}