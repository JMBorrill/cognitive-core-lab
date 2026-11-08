import type { Manifest, Results } from "@/lib/types";
import type { ConfigInfo } from "@/lib/configs";
import { SPLIT_ORDER, SPLIT_SHORT, getCI } from "@/lib/configs";
import DotPlot, { type DotGroup } from "./DotPlot";

type Props = {
  manifest: Manifest | null;
  results: Results | null;
  configs: ConfigInfo[];
  metric?: string;
  includePooled?: boolean;
  format?: "pct" | "num";
  digits?: number;
};

/** Grouped dot plot: one group per split, one row per configuration (absent → "no results yet"). */
export default function SuccessBySplit({
  manifest,
  results,
  configs,
  metric = "success",
  includePooled = false,
  format = "pct",
  digits = 1,
}: Props) {
  const splits = [...SPLIT_ORDER, ...(includePooled ? ["all_main"] : [])];
  const groups: DotGroup[] = splits.map((s) => ({
    id: s,
    title: SPLIT_SHORT[s] ?? s,
    subtitle: s === "all_main" ? "(pooled)" : `${s}${manifest?.splits?.[s] ? ` — ${manifest.splits[s]}` : ""}`,
    rows: configs.map((c) => ({
      id: c.key,
      label: c.label,
      color: c.color,
      ci: getCI(results, c.rkey, s, metric),
      empty: results?.configs?.[c.rkey] ? "not measured" : "no results yet",
    })),
  }));
  return (
    <DotPlot
      groups={groups}
      format={format}
      digits={digits}
      caption={`${metric} by split with 95% confidence intervals`}
    />
  );
}
