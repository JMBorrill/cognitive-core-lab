import Link from "next/link";

export default function NotFound() {
  return (
    <div className="page-head">
      <div className="eyebrow">404</div>
      <h1>Page not found</h1>
      <p className="lede">
        <Link href="/">Back to the overview</Link>
      </p>
    </div>
  );
}
