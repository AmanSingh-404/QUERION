import Link from "next/link";

const links = [
  ["/playground", "Playground"],
  ["/benchmarks", "Benchmarks"],
  ["/how-it-works", "How it works"],
  ["/model-card", "Model"],
  ["/status", "Status"],
] as const;

export default function Nav() {
  return (
    <header className="wrap hd u">
      <Link href="/" className="logo">
        querion
      </Link>
      <nav className="nav">
        {links.map(([href, label]) => (
          <Link key={href} href={href}>
            {label}
          </Link>
        ))}
      </nav>
    </header>
  );
}