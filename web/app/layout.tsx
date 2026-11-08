import type { Metadata, Viewport } from "next";
import Link from "next/link";
import Nav from "@/components/Nav";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "cognitive-core-lab · LoRA for tool use in a 0.5B model",
    template: "%s · cognitive-core-lab",
  },
  description:
    "Does LoRA fine-tuning improve tool selection and multi-step agent orchestration in Qwen2.5-0.5B-Instruct? Benchmarks, replayable traces and live local inference.",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#ffffff" },
    { media: "(prefers-color-scheme: dark)", color: "#1a1a19" },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <a className="skip-link" href="#main">
          Skip to content
        </a>
        <header className="site-header">
          <div className="site-header-inner">
            <Link href="/" className="brand">
              <span className="brand-mark" aria-hidden="true" />
              cognitive-core-lab
            </Link>
            <Nav />
          </div>
        </header>
        <main id="main">{children}</main>
        <footer className="site-footer">
          <div className="site-footer-inner">
            <span>
              All numbers and traces are read from <code>public/data/</code>, exported from real evaluation runs by{" "}
              <code>scripts/export_web.py</code>.
            </span>
            <span>Qwen2.5-0.5B-Instruct · MLX LoRA · deterministic tool simulators</span>
          </div>
        </footer>
      </body>
    </html>
  );
}
