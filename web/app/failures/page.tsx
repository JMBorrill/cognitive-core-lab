import type { Metadata } from "next";
import { loadData } from "@/lib/data";
import { configList } from "@/lib/configs";
import FailureGallery from "@/components/FailureGallery";
import MissingData from "@/components/MissingData";

export const metadata: Metadata = { title: "Failure gallery" };

export default async function FailuresPage() {
  const { manifest, failures } = await loadData(["manifest", "failures"] as const);
  const configs = configList(manifest).map((c) => ({
    key: c.key,
    label: c.label,
    short: c.short,
    color: c.color,
    replayRun: manifest?.configs[c.key]?.replay_run ?? c.rkey,
  }));
  return (
    <>
      <header className="page-head">
        <div className="eyebrow">Failure gallery</div>
        <h1>How each configuration fails</h1>
        <p className="lede">
          Aggregate success hides <em>why</em> episodes fail. Every failed episode is classified into one failure mode
          from its validator checks and runtime counters, and up to three recorded traces per mode are kept as
          examples.
        </p>
      </header>
      {failures ? <FailureGallery configs={configs} failures={failures} /> : <MissingData file="failures.json" what="The failure gallery" />}
    </>
  );
}
