"use client";

import { useEffect, useState } from "react";

import { API_URL } from "@/lib/config";
import { TUNNEL_HEADERS } from "@/lib/api";

type State = { kind: "checking" } | { kind: "up"; body: Record<string, unknown> } | { kind: "down" };

export default function Status() {
  const [state, setState] = useState<State>({ kind: "checking" });

  useEffect(() => {
    let cancelled = false;
    fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(5000), headers: TUNNEL_HEADERS })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error("bad status"))))
      .then((body) => !cancelled && setState({ kind: "up", body }))
      .catch(() => !cancelled && setState({ kind: "down" }));
    return () => {
      cancelled = true;
    };
  }, []);

  const label = state.kind === "checking" ? "Checking" : state.kind === "up" ? "Up" : "Offline";
  return (
    <main className="wrap" style={{ paddingBottom: 96 }}>
      <section style={{ padding: "64px 0 48px" }}>
        <h1 className="h1 u">Status</h1>
      </section>
      <div className="k u" style={{ animationDelay: ".2s", alignItems: "center" }}>
        <span style={{ display: "flex", alignItems: "center", gap: 12, color: "var(--ink)" }}>
          <i
            style={{
              width: 10,
              height: 10,
              borderRadius: 5,
              background: state.kind === "down" ? "var(--muted)" : "var(--orange)",
              animation: state.kind === "down" ? undefined : "beat 2s ease-in-out infinite",
            }}
          />
          Model server
        </span>
        <span>{label}</span>
      </div>
      <div className="k u" style={{ animationDelay: ".3s" }}>
        <span>When offline</span>
        <span>The playground shows recorded examples.</span>
      </div>
      {state.kind === "up" && (
        <pre className="mo" style={{ marginTop: 24 }}>
          {JSON.stringify(state.body, null, 2)}
        </pre>
      )}
    </main>
  );
}