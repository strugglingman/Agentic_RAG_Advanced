/**
 * SSE (Server-Sent Events) stream parsing utilities.
 *
 * Parses standard SSE protocol with typed events:
 *  - "text"    → answer text chunks → forwarded to onTextChunk
 *  - "hitl"    → HITL interrupt JSON → accumulated in acc.hitlRaw
 *  - "context" → context array JSON → accumulated in acc.contextRaw
 */

export type SSEAccumulator = { hitlRaw: string; contextRaw: string };

/**
 * Consume an SSE stream, dispatching events by type.
 *
 * Handles chunked delivery: raw bytes from `reader.read()` may split
 * SSE events mid-way, so we buffer and split on double-newline boundaries.
 *
 * `acc` is mutated in-place so partial data survives if the stream throws.
 */
export async function consumeSSEStream(
  stream: AsyncIterable<string>,
  onTextChunk: (text: string) => void,
  acc: SSEAccumulator,
): Promise<void> {
  let buffer = '';

  for await (const chunk of stream) {
    buffer += chunk;

    // Process complete SSE events (terminated by \n\n)
    let boundary: number;
    while ((boundary = buffer.indexOf('\n\n')) !== -1) {
      const rawEvent = buffer.substring(0, boundary);
      buffer = buffer.substring(boundary + 2);

      if (!rawEvent.trim()) continue;

      // Parse SSE fields: "event: <type>" and "data: <payload>"
      let eventType = 'message';
      const dataLines: string[] = [];

      for (const line of rawEvent.split('\n')) {
        if (line.startsWith('event: ')) {
          eventType = line.substring(7);
        } else if (line.startsWith('data: ')) {
          dataLines.push(line.substring(6));
        }
      }

      const data = dataLines.join('\n');

      switch (eventType) {
        case 'text':
          onTextChunk(data);
          break;
        case 'hitl':
          acc.hitlRaw = data;
          break;
        case 'context':
          acc.contextRaw = data;
          break;
      }
    }
  }
}

/** Parse HITL payload JSON, or null on failure. */
export function parseHitlPayload(raw: string): Record<string, unknown> | null {
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch (e) {
    console.error('Failed to parse HITL JSON:', e);
    return null;
  }
}

/** Parse context payload JSON into an array, or [] on failure. */
export function parseContextPayload(raw: string): unknown[] {
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch (e) {
    console.error('Failed to parse context JSON:', e);
    return [];
  }
}
