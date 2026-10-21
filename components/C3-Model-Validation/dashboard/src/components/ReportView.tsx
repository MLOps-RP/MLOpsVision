import { Download } from "lucide-react";
import { useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { downloadText } from "../lib/format";
import type { PdfMeta } from "../lib/reportPdf";
import { Button, Card, CardHeader } from "./ui";

/**
 * Markdown exactly as the gate wrote it (react-markdown does not render raw HTML).
 * With `pdf`, the download is a real PDF built from the same text; without it, the Markdown file.
 */
export function ReportView({ markdown, filename, pdf, title = "Unified report", subtitle = "The same Markdown the gate writes to disk." }: {
  markdown: string; filename: string; pdf?: PdfMeta; title?: string; subtitle?: string;
}) {
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);
  async function downloadPdf() {
    if (!pdf) return;
    setBusy(true); setFailure(null);
    try {
      const { downloadReportPdf } = await import("../lib/reportPdf");          // loaded only when a PDF is asked for
      await downloadReportPdf(markdown, pdf);
    } catch (error) {
      setFailure(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }
  return (
    <Card>
      <CardHeader title={title} subtitle={subtitle}
        actions={pdf
          ? <Button small disabled={busy} onClick={() => void downloadPdf()}><Download size={15} aria-hidden /> {busy ? "Preparing PDF…" : "Download PDF"}</Button>
          : <Button small onClick={() => downloadText(filename, markdown, "text/markdown")}><Download size={15} aria-hidden /> Download .md</Button>} />
      {failure && <p role="alert" className="px-5 pb-2 text-[13px] text-red-700">The PDF could not be created: {failure}</p>}
      <article aria-label="Report" className="max-w-4xl px-7 py-5 leading-relaxed [&_blockquote]:my-3 [&_blockquote]:border-l-[3px] [&_blockquote]:border-blue-400 [&_blockquote]:bg-blue-50 [&_blockquote]:px-4 [&_blockquote]:py-1 [&_code]:rounded [&_code]:bg-blue-50 [&_code]:px-1 [&_code]:text-blue-900 [&_h1]:mb-3 [&_h1]:border-b [&_h1]:border-slate-200 [&_h1]:pb-2.5 [&_h1]:text-[22px] [&_h1]:font-bold [&_h2]:mb-2.5 [&_h2]:mt-6 [&_h2]:text-[17px] [&_h2]:font-semibold [&_h2]:text-blue-800 [&_h3]:mb-2 [&_h3]:mt-4 [&_h3]:font-semibold [&_li]:my-0.5 [&_ol]:my-2 [&_ol]:list-decimal [&_ol]:pl-6 [&_p]:my-2 [&_table]:my-3 [&_table]:block [&_table]:w-full [&_table]:overflow-x-auto [&_table]:border-collapse [&_table]:text-[13px] [&_td]:border [&_td]:border-slate-200 [&_td]:px-3 [&_td]:py-1.5 [&_td]:align-top [&_th]:border [&_th]:border-blue-100 [&_th]:bg-blue-50 [&_th]:px-3 [&_th]:py-2 [&_th]:text-left [&_th]:text-blue-900 [&_ul]:my-2 [&_ul]:list-disc [&_ul]:pl-6">
        <Markdown remarkPlugins={[remarkGfm]}>{markdown}</Markdown>
      </article>
    </Card>
  );
}
