export type GenerateEvent =
  | { type: "start"; samples: number }
  | { type: "candidate"; index: number; sql: string }
  | { type: "error"; index: number; message: string }
  | { type: "done"; candidates: (string | null)[]; seconds: number };

export type GenerateBody = { tables: string[]; question: string; samples?: number };
export const TUNNEL_HEADERS = { "ngrok-skip-browser-warning": "1" };

/** One server-sent event block ("event: name" and "data: json") as a typed event. */
export function parseEvent(raw: string): GenerateEvent | null {
  let name = "";
  let data = "";
  for (const line of raw.split("\n")) {
    if (line.startsWith("event: ")) name = line.slice(7);
    else if (line.startsWith("data: ")) data += line.slice(6);
  }
  if (!name || !data) return null;
  return { type: name, ...JSON.parse(data) } as GenerateEvent;
}

/** Streams the candidate queries of the API as they are produced. */
export async function* generate(
  baseUrl: string,
  body: GenerateBody,
  signal?: AbortSignal,
): AsyncGenerator<GenerateEvent> {
  const response = await fetch(`${baseUrl}/generate`, {
    method: "POST",
    headers: { "content-type": "application/json", ...TUNNEL_HEADERS },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok || !response.body) {
    let detail: unknown = response.statusText;
    try {
      detail = (await response.json()).detail ?? detail;
    } catch {
      /* the body was not JSON */
    }
    throw new Error(`${response.status}: ${typeof detail === "string" ? detail : JSON.stringify(detail)}`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut = buffer.indexOf("\n\n");
    while (cut >= 0) {
      const event = parseEvent(buffer.slice(0, cut));
      buffer = buffer.slice(cut + 2);
      if (event) yield event;
      cut = buffer.indexOf("\n\n");
    }
  }
}