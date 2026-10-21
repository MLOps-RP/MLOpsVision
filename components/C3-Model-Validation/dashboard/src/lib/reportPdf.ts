import type { PhrasingContent, Root, RootContent, Table } from "mdast";
import remarkGfm from "remark-gfm";
import remarkParse from "remark-parse";
import { unified } from "unified";

import { downloadBlob } from "./format";

/**
 * Builds a PDF from the gate's own Markdown report. Nothing is added to the report's content: every heading, table,
 * list and paragraph in the PDF is one of the report's, in the report's order. The only additions are the cover block
 * (the title and the facts the API returned with the run) and the page footer.
 */
export interface PdfMeta {
  /** The friendly name shown for the model, e.g. "Model 1". */
  name: string;
  runId: string;
  /** When the run was validated (from the API); null if unknown. */
  createdAt: string | null;
}

export interface BuildOptions {
  /** Compress page streams (default). Tests turn it off or inflate the streams themselves. */
  compress?: boolean;
  /** The time shown as "PDF created"; defaults to now. */
  now?: Date;
}

const MARGIN = 48;
const PAGE_W = 595.28;
const PAGE_H = 841.89;
const BOTTOM = 56;
const BLUE: [number, number, number] = [30, 64, 175];
const INK: [number, number, number] = [30, 41, 59];
const MUTED: [number, number, number] = [100, 116, 139];

/** `Model 1` -> `Model-1-Validation-Report.pdf`. */
export function reportPdfFilename(name: string): string {
  const stem = name.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^[-.]+|[-.]+$/g, "");
  return `${stem || "Validation"}-Validation-Report.pdf`;
}

/**
 * The PDF's built-in fonts only cover Western European text, so the few symbols the report uses beyond that are
 * written in ASCII, and anything else becomes "?" rather than a broken glyph.
 */
const ASCII: Record<string, string> = {
  "→": "->", "←": "<-", "≥": ">=", "≤": "<=", "≈": "~", "≠": "!=", "−": "-", "✓": "ok", "✗": "x", " ": " ", "‑": "-", " ": " ",
};
export function pdfText(text: string): string {
  return text.replace(/[^\x09\x0a\x20-\x7e\xa1-\xff–—‘’“”…•]/g, (c) => ASCII[c] ?? "?")
    .replace(/ /g, " ");
}

type Style = "normal" | "bold" | "italic" | "bolditalic" | "code";
interface Run { text: string; style: Style }

const plain = (nodes: PhrasingContent[]): string => nodes.map((n) => {
  if ("children" in n) return plain(n.children as PhrasingContent[]);
  if (n.type === "break") return " ";
  if (n.type === "image") return n.alt ?? "";
  return "value" in n && n.type !== "html" ? n.value : "";
}).join("");

function runsOf(nodes: PhrasingContent[], bold = false, italic = false): Run[] {
  const style = (): Style => (bold && italic ? "bolditalic" : bold ? "bold" : italic ? "italic" : "normal");
  return nodes.flatMap((n): Run[] => {
    switch (n.type) {
      case "text": return [{ text: n.value, style: style() }];
      case "strong": return runsOf(n.children, true, italic);
      case "emphasis": return runsOf(n.children, bold, true);
      case "delete": case "link": case "linkReference": return runsOf(n.children, bold, italic);
      case "inlineCode": return [{ text: n.value, style: "code" }];
      case "break": return [{ text: "\n", style: "normal" }];
      case "image": return [{ text: n.alt ?? "", style: style() }];
      default: return [];
    }
  });
}

export async function buildReportPdf(markdown: string, meta: PdfMeta, options: BuildOptions = {}) {
  const [{ jsPDF }, autoTableModule] = await Promise.all([import("jspdf"), import("jspdf-autotable")]);
  const autoTable = autoTableModule.default;
  const tree = unified().use(remarkParse).use(remarkGfm).parse(markdown) as Root;
  const doc = new jsPDF({ unit: "pt", format: "a4", compress: options.compress ?? true });
  const width = PAGE_W - 2 * MARGIN;
  let y = MARGIN;

  const fontFor = (style: Style, size: number) => {
    if (style === "code") doc.setFont("courier", "normal");
    else doc.setFont("helvetica", style);
    doc.setFontSize(style === "code" ? size * 0.92 : size);
  };
  const room = (height: number) => {
    if (y + height > PAGE_H - BOTTOM) { doc.addPage(); y = MARGIN; }
  };
  const widthOf = (text: string, style: Style, size: number) => { fontFor(style, size); return doc.getTextWidth(text); };

  /** Lays out styled runs as wrapped lines inside [x, x + maxWidth]. */
  function paragraph(runs: Run[], x: number, maxWidth: number, size = 9.5, color = INK) {
    const lineHeight = size * 1.45;
    type Token = { text: string; style: Style; newline?: boolean };
    const tokens: Token[] = [];
    for (const run of runs) {
      const text = pdfText(run.text);
      for (const piece of text.split(/(\n|\s+)/)) {
        if (piece === "") continue;
        if (piece === "\n") tokens.push({ text: "", style: run.style, newline: true });
        else tokens.push({ text: /^\s+$/.test(piece) ? " " : piece, style: run.style });
      }
    }
    let line: Token[] = [];
    let used = 0;
    const flush = () => {
      while (line.length && line[line.length - 1].text === " ") line.pop();
      if (!line.length) { line = []; used = 0; return; }
      room(lineHeight);
      let cursor = x;
      let i = 0;
      while (i < line.length) {                       // one text call per run of the same style
        let j = i; let chunk = "";
        while (j < line.length && line[j].style === line[i].style) { chunk += line[j].text; j++; }
        fontFor(line[i].style, size);
        const [r, g, b] = line[i].style === "code" ? ([30, 58, 138] as const) : color;
        doc.setTextColor(r, g, b);
        doc.text(chunk, cursor, y + size);
        cursor += doc.getTextWidth(chunk);
        i = j;
      }
      y += lineHeight;
      line = []; used = 0;
    };
    for (const token of tokens) {
      if (token.newline) { flush(); continue; }
      if (token.text === " " && used === 0) continue;
      let pieces = [token.text];
      if (token.text !== " " && widthOf(token.text, token.style, size) > maxWidth) {   // a word wider than the line (a long hash or path)
        pieces = []; let current = "";
        for (const ch of token.text) {
          if (current && widthOf(current + ch, token.style, size) > maxWidth) { pieces.push(current); current = ""; }
          current += ch;
        }
        if (current) pieces.push(current);
      }
      pieces.forEach((piece, index) => {
        const w = widthOf(piece, token.style, size);
        if (index > 0 || (used + w > maxWidth && used > 0 && piece !== " ")) flush();
        line.push({ text: piece, style: token.style });
        used += w;
      });
    }
    flush();
  }

  function table(node: Table, x: number) {
    const rows = node.children.map((row) => row.children.map((cell) => pdfText(plain(cell.children as PhrasingContent[]))));
    if (!rows.length) return;
    const columns = rows[0].length;
    room(40);
    autoTable(doc, {
      head: [rows[0]], body: rows.slice(1), startY: y, theme: "grid",
      margin: { left: x, right: MARGIN, bottom: BOTTOM },
      styles: { font: "helvetica", fontSize: columns > 6 ? 7 : 8, cellPadding: 3, overflow: "linebreak", lineColor: [203, 213, 225], lineWidth: 0.4, textColor: INK },
      headStyles: { fillColor: [219, 234, 254], textColor: [30, 58, 138], fontStyle: "bold" },
      alternateRowStyles: { fillColor: [248, 250, 252] },
    });
    y = (doc as unknown as { lastAutoTable: { finalY: number } }).lastAutoTable.finalY + 10;
  }

  function block(node: RootContent, x: number, maxWidth: number) {
    switch (node.type) {
      case "heading": {
        if (node.depth === 1) return;                                   // the title is drawn once, above
        const size = node.depth === 2 ? 14 : node.depth === 3 ? 11.5 : 10;
        y += node.depth === 2 ? 12 : 7;
        room(size * 2.4);
        paragraph(runsOf(node.children).map((r) => ({ ...r, style: "bold" as Style })), x, maxWidth, size, node.depth === 2 ? BLUE : INK);
        if (node.depth === 2) { doc.setDrawColor(191, 219, 254); doc.setLineWidth(0.8); doc.line(x, y, x + maxWidth, y); y += 5; }
        else y += 2;
        return;
      }
      case "paragraph": paragraph(runsOf(node.children), x, maxWidth); y += 5; return;
      case "blockquote": {
        const page = doc.getNumberOfPages(); const top = y;
        node.children.forEach((child) => block(child, x + 12, maxWidth - 12));
        if (doc.getNumberOfPages() === page) { doc.setFillColor(96, 165, 250); doc.rect(x + 2, top, 2.5, Math.max(y - top - 4, 4), "F"); }
        return;
      }
      case "list": {
        node.children.forEach((item, index) => {
          const marker = node.ordered ? `${(node.start ?? 1) + index}.` : "-";
          room(14);
          const markerY = y;
          fontFor("normal", 9.5); doc.setTextColor(...INK);
          doc.text(marker, x + 2, markerY + 9.5);
          item.children.forEach((child) => block(child, x + 16, maxWidth - 16));
        });
        y += 2; return;
      }
      case "table": table(node, x); return;
      case "code": {
        const lines = pdfText(node.value).split("\n");
        for (const text of lines) paragraph([{ text: text || " ", style: "code" }], x + 6, maxWidth - 6, 8.5);
        y += 5; return;
      }
      case "thematicBreak": room(10); doc.setDrawColor(226, 232, 240); doc.line(x, y + 4, x + maxWidth, y + 4); y += 10; return;
      default: return;                                                  // raw HTML and definitions: not part of the report text
    }
  }

  // ---------------------------------------------------------------- cover block
  const first = tree.children[0];
  const title = pdfText(first?.type === "heading" && first.depth === 1 ? plain(first.children as PhrasingContent[]) : `Model Quality Validation Report: ${meta.name}`);
  doc.setFillColor(...BLUE); doc.rect(0, 0, PAGE_W, 8, "F");
  y = 40;
  paragraph([{ text: title, style: "bold" }], MARGIN, width, 19, BLUE);
  y += 4;
  const when = (iso: string | null) => (iso && !Number.isNaN(new Date(iso).getTime()) ? `${new Date(iso).toISOString().slice(0, 16).replace("T", " ")} UTC` : "not recorded");
  const facts: [string, string][] = [
    ["Candidate", meta.name], ["Validated", when(meta.createdAt)], ["Run id", meta.runId], ["PDF created", when((options.now ?? new Date()).toISOString())],
  ];
  facts.forEach(([label, value]) => {
    room(14); fontFor("bold", 9.5); doc.setTextColor(...MUTED); doc.text(`${label}:`, MARGIN, y + 9.5);
    fontFor(label === "Run id" ? "code" : "normal", 9.5); doc.setTextColor(...INK); doc.text(pdfText(value), MARGIN + 78, y + 9.5);
    y += 14;
  });
  doc.setDrawColor(191, 219, 254); doc.setLineWidth(1); doc.line(MARGIN, y + 4, MARGIN + width, y + 4); y += 12;

  tree.children.forEach((node) => block(node, MARGIN, width));

  // ---------------------------------------------------------------- footer with page numbers
  const pages = doc.getNumberOfPages();
  for (let page = 1; page <= pages; page++) {
    doc.setPage(page);
    doc.setDrawColor(226, 232, 240); doc.setLineWidth(0.5); doc.line(MARGIN, PAGE_H - 40, PAGE_W - MARGIN, PAGE_H - 40);
    fontFor("normal", 8); doc.setTextColor(...MUTED);
    doc.text(pdfText(`${meta.name} - Model Quality Validation Report`), MARGIN, PAGE_H - 27);
    doc.text(`Page ${page} of ${pages}`, PAGE_W - MARGIN, PAGE_H - 27, { align: "right" });
  }
  doc.setProperties({ title, subject: `Validation report for ${pdfText(meta.name)}`, author: "Validation Gate", creator: "Validation Gate dashboard" });
  return doc;
}

/** Builds the PDF and hands it to the browser as `Model-1-Validation-Report.pdf`. */
export async function downloadReportPdf(markdown: string, meta: PdfMeta): Promise<void> {
  const doc = await buildReportPdf(markdown, meta);
  downloadBlob(reportPdfFilename(meta.name), doc.output("blob"));
}
