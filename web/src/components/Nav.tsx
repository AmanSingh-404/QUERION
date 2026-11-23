import Link from "next/link";

const links = [
  ["/", "Playground"],
  ["/benchmarks", "Benchmarks"],
  ["/how-it-works", "How it works"],
  ["/model-card", "Model card"],
  ["/status", "Status"],
] as const;

export default function Nav() {
  return (
    <nav className="mx-auto flex max-w-4xl flex-wrap gap-4 p-6 pb-0 text-sm">
      {links.map(([href, label]) => (
        <Link key={href} href={href} className="underline-offset-4 hover:underline">
          {label}
        </Link>
      ))}
    </nav>
  );
}