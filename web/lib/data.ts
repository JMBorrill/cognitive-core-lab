// Server-side loaders for public/data/*.json (written by scripts/export_web.py).
// Files are read from disk on every request, so re-running the export is picked up
// by `npm start` without a rebuild. Missing files resolve to null — never to fake data.

import { readFile } from "node:fs/promises";
import path from "node:path";
import { connection } from "next/server";
import type {
  Datasets,
  Failures,
  Leakage,
  Manifest,
  Results,
  Showcase,
  Training,
} from "./types";

const DATA_DIR = path.join(process.cwd(), "public", "data");

/**
 * Python's json.dumps emits bare NaN / Infinity for undefined statistics (e.g. a bootstrap
 * over zero tasks). Those are not valid JSON; map them to null outside of string literals.
 */
export function sanitizeJsonText(text: string): string {
  let out = "";
  let inStr = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inStr) {
      out += ch;
      if (ch === "\\") {
        out += text[i + 1] ?? "";
        i++;
      } else if (ch === '"') {
        inStr = false;
      }
      continue;
    }
    if (ch === '"') {
      inStr = true;
      out += ch;
      continue;
    }
    if (text.startsWith("NaN", i)) {
      out += "null";
      i += 2;
      continue;
    }
    if (text.startsWith("-Infinity", i)) {
      out += "null";
      i += 8;
      continue;
    }
    if (text.startsWith("Infinity", i)) {
      out += "null";
      i += 7;
      continue;
    }
    out += ch;
  }
  return out;
}

async function readText(name: string): Promise<string | null> {
  try {
    return await readFile(path.join(DATA_DIR, name), "utf8");
  } catch {
    return null;
  }
}

async function readJson<T>(name: string): Promise<T | null> {
  const text = await readText(name);
  if (text === null) return null;
  try {
    return JSON.parse(text) as T;
  } catch {
    try {
      return JSON.parse(sanitizeJsonText(text)) as T;
    } catch {
      console.error(`[data] could not parse public/data/${name}`);
      return null;
    }
  }
}

export type SiteData = {
  manifest: Manifest | null;
  results: Results | null;
  showcase: Showcase | null;
  failures: Failures | null;
  training: Training | null;
  teacher: Record<string, unknown> | null;
  datasets: Datasets | null;
  leakage: Leakage | null;
  report: string | null;
};

type Key = keyof SiteData;

const FILES: Record<Key, string> = {
  manifest: "manifest.json",
  results: "results.json",
  showcase: "showcase.json",
  failures: "failures.json",
  training: "training.json",
  teacher: "teacher.json",
  datasets: "datasets.json",
  leakage: "leakage.json",
  report: "report.md",
};

/** Load the requested data files (request-time; opts the page out of prerendering). */
export async function loadData<K extends Key>(keys: readonly K[]): Promise<Pick<SiteData, K>> {
  await connection();
  const entries = await Promise.all(
    keys.map(async (k) => [k, k === "report" ? await readText(FILES[k]) : await readJson(FILES[k])] as const),
  );
  return Object.fromEntries(entries) as Pick<SiteData, K>;
}
