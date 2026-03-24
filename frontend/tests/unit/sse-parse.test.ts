import { consumeSSEStream, parseContextPayload, parseHitlPayload } from "@/lib/sse-parse";

async function* chunks(parts: string[]) {
  for (const part of parts) {
    yield part;
  }
}

describe("consumeSSEStream", () => {
  it("parses text/hitl/context events across chunk boundaries", async () => {
    const text: string[] = [];
    const acc = { hitlRaw: "", contextRaw: "" };

    await consumeSSEStream(
      chunks([
        "event: text\ndata: Hel",
        "lo\n\nevent: hitl\ndata: {\"status\":\"awaiting_confirmation\"}\n\n",
        "event: context\ndata: [{\"chunk\":\"doc\",\"source\":\"f\",\"page\":1}]\n\n",
      ]),
      (t) => text.push(t),
      acc,
    );

    expect(text).toEqual(["Hello"]);
    expect(acc.hitlRaw).toContain("awaiting_confirmation");
    expect(acc.contextRaw).toContain("\"chunk\":\"doc\"");
  });
});

describe("payload parsers", () => {
  it("returns parsed HITL payload or null", () => {
    expect(parseHitlPayload("{\"ok\":true}")).toEqual({ ok: true });
    expect(parseHitlPayload("not-json")).toBeNull();
  });

  it("returns parsed context payload array or empty array", () => {
    expect(parseContextPayload("[{\"a\":1}]")).toEqual([{ a: 1 }]);
    expect(parseContextPayload("{\"a\":1}")).toEqual([]);
    expect(parseContextPayload("not-json")).toEqual([]);
  });
});
