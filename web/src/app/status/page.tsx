"use client";

import { useEffect, useState } from "react";

import { API_URL } from "@/lib/config";

type State = { kind: "checking" } | { kind: "up"; body: Record<string, unknown> } | { kind: "down" };

export default function Status() {
  const [state, setState] = useState<State>({ kind: "checking" });

  useEffect(() => {
    let cancelled = false;
    fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(5000) })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error("bad status"))))
      .then((body) => !cancelled && setState({ kind: "up", body }))
      .catch(() => !cancelled && setState({ kind: "down" }));
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <main className="mx-auto max-w-4xl space-y-4 p-6">
      <h1 className="text-3xl font-semibold">Status</h1>
      {state.kind === "checking" && <p className="opacity-70">Checking the model server...</p>}
      {state.kind === "down" && (
        <p>
          The live model server is offline. The playground still works with the recorded
          examples.
        </p>
      )}
      {state.kind === "up" && (
        <div className="space-y-2">
          <p>The live model server is up.</p>
          <pre className="overflow-x-auto rounded border p-3 text-sm">
            {JSON.stringify(state.body, null, 2)}
          </pre>
        </div>
      )}
    </main>
  );
}