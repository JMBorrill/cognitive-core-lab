export function ReplayBadge({ run }: { run: string }) {
  return (
    <div className="mode-badge mode-replay" role="note">
      <span className="dot" aria-hidden="true" />
      REPLAY
      <span className="detail">
        — recorded trace from evaluation run <code>{run}</code>
      </span>
    </div>
  );
}

export function LiveBadge({ detail }: { detail?: string }) {
  return (
    <div className="mode-badge mode-live" role="note">
      <span className="dot" aria-hidden="true" />
      LIVE
      <span className="detail">— local inference on this machine{detail ? ` · ${detail}` : ""}</span>
    </div>
  );
}
