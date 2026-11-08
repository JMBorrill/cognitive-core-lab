export default function MissingData({ file, what }: { file: string; what?: string }) {
  return (
    <div className="empty">
      <strong>No data yet.</strong> {what ?? "This section"} reads <code>public/data/{file}</code>, which has not been
      exported. Run <code>python scripts/export_web.py</code> from the repository root after the relevant runs finish.
    </div>
  );
}
