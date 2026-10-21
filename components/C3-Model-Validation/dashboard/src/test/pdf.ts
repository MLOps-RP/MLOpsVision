import { inflateSync } from "node:zlib";

/** The bytes of a PDF as a binary string (one character per byte). */
export const binary = (bytes: Uint8Array) => Buffer.from(bytes).toString("latin1");

const unescapePdf = (s: string) => s.replace(/\\([()\\])/g, "$1").replace(/\\(\d{3})/g, (_, o: string) => String.fromCharCode(parseInt(o, 8)));

/** Reads a PDF the way a viewer would: inflate each page stream, then collect the strings drawn with Tj / TJ. */
export function pdfInfo(bytes: Uint8Array) {
  const raw = binary(bytes);
  const texts: string[] = [];
  for (const match of raw.matchAll(/stream\r?\n([\s\S]*?)\r?\nendstream/g)) {
    let content = Buffer.from(match[1], "latin1");
    try { content = inflateSync(content); } catch { /* not compressed */ }
    const stream = content.toString("latin1");
    for (const t of stream.matchAll(/\(((?:\\.|[^\\)])*)\)\s*Tj/g)) texts.push(unescapePdf(t[1]));
    for (const arr of stream.matchAll(/\[((?:\\.|[^\]\\])*)\]\s*TJ/g)) {
      texts.push([...arr[1].matchAll(/\(((?:\\.|[^\\)])*)\)/g)].map((p) => unescapePdf(p[1])).join(""));
    }
  }
  return {
    startsLikePdf: raw.startsWith("%PDF-"),
    endsLikePdf: raw.trimEnd().endsWith("%%EOF"),
    pages: (raw.match(/\/Type\s*\/Page\b(?!s)/g) ?? []).length,
    texts,
    text: texts.join("\n"),
    raw,
  };
}
