import type { Metadata } from "next";
import { loadData } from "@/lib/data";
import { configList } from "@/lib/configs";
import TraceExplorer from "@/components/TraceExplorer";
import MissingData from "@/components/MissingData";

export const metadata: Metadata = { title: "Trace explorer" };

type SP = Promise<Record<string, string | string[] | undefined>>;

export default async function TracesPage({ searchParams }: { searchParams: SP }) {
  const sp = await searchParams;
  const one = (k: string) => (typeof sp[k] === "string" ? (sp[k] as string) : undefined);
  const { manifest, showcase } = await loadData(["manifest", "showcase"] as const);
  const configs = configList(manifest);

  return (
    <>
      <header className="page-head">
        <div className="eyebrow">Trace explorer</div>
        <h1>Watch each configuration work through the same task</h1>
        <p className="lede">
          Every configuration is evaluated on identical held-out tasks. Pick a task, then read the turn-by-turn trace:
          the model&apos;s raw output, the calls the runtime parsed from it, what each call returned, and how the
          validator scored the episode. <strong>Replay</strong> shows recorded evaluation traces;{" "}
          <strong>Live</strong> runs a configuration on this machine through the local inference server.
        </p>
      </header>
      {showcase ? (
        <TraceExplorer
          tasks={showcase.tasks}
          traces={showcase.traces}
          configs={configs}
          initial={{ task: one("task"), config: one("config"), compare: one("compare"), mode: one("mode") }}
        />
      ) : (
        <MissingData file="showcase.json" />
      )}
    </>
  );
}
