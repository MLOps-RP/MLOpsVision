import {
  AlertTriangle, CheckCircle2, Info, OctagonX, PlugZap, SearchX, type LucideIcon,
} from "lucide-react";
import { useRef, type ButtonHTMLAttributes, type ReactNode } from "react";

import { ApiError } from "../api/client";
import { useConnection } from "../state/connection";

export const cx = (...parts: (string | false | null | undefined)[]) => parts.filter(Boolean).join(" ");

/* ------------------------------------------------------------------ surfaces */
export function Card({ className, children, ...rest }: { className?: string; children: ReactNode } & React.HTMLAttributes<HTMLElement>) {
  return <section className={cx("rounded-xl border border-slate-200 bg-white shadow-sm", className)} {...rest}>{children}</section>;
}

export function CardHeader({ title, subtitle, actions }: { title: ReactNode; subtitle?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 px-5 py-3.5">
      <div className="min-w-0">
        <h2 className="text-base font-semibold text-slate-900">{title}</h2>
        {subtitle && <p className="mt-0.5 text-[13px] text-slate-500">{subtitle}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}

export const CardBody = ({ className, children }: { className?: string; children: ReactNode }) => (
  <div className={cx("px-5 py-4", className)}>{children}</div>
);

/* ------------------------------------------------------------------- buttons */
type Variant = "primary" | "secondary" | "ghost";
export function buttonClass(variant: Variant = "secondary", small = false) {
  return cx(
    "inline-flex items-center justify-center gap-2 rounded-lg border font-semibold whitespace-nowrap transition-colors disabled:cursor-not-allowed disabled:opacity-55",
    small ? "h-8 px-3 text-[13px]" : "h-9 px-4",
    variant === "primary" && "border-blue-600 bg-blue-600 text-white hover:bg-blue-700",
    variant === "secondary" && "border-slate-300 bg-white text-slate-800 hover:bg-slate-50",
    variant === "ghost" && "border-transparent bg-transparent text-blue-700 hover:bg-blue-50",
  );
}

export function Button({ variant = "secondary", small, className, ...rest }:
  { variant?: Variant; small?: boolean } & ButtonHTMLAttributes<HTMLButtonElement>) {
  return <button type="button" className={cx(buttonClass(variant, small), className)} {...rest} />;
}

/* -------------------------------------------------------------------- badges */
export type Tone = "green" | "amber" | "red" | "blue" | "slate" | "violet" | "orange";
const TONES: Record<Tone, string> = {
  green: "bg-green-50 text-green-800 border-green-200",
  amber: "bg-amber-50 text-amber-800 border-amber-200",
  red: "bg-red-50 text-red-800 border-red-200",
  blue: "bg-blue-50 text-blue-800 border-blue-200",
  slate: "bg-slate-100 text-slate-700 border-slate-200",
  violet: "bg-violet-50 text-violet-800 border-violet-200",
  orange: "bg-orange-50 text-orange-800 border-orange-200",
};
export function Badge({ tone = "slate", children, title, className }: { tone?: Tone; children: ReactNode; title?: string; className?: string }) {
  return (
    <span title={title} className={cx("inline-flex items-center gap-1 whitespace-nowrap rounded-full border px-2.5 py-0.5 text-xs font-semibold", TONES[tone], className)}>
      {children}
    </span>
  );
}

/* ------------------------------------------------------------------- notices */
type NoticeTone = "info" | "warn" | "error" | "ok";
const NOTICE_STYLE: Record<NoticeTone, string> = {
  info: "border-blue-200 bg-blue-50 text-blue-900",
  warn: "border-amber-200 bg-amber-50 text-amber-900",
  error: "border-red-200 bg-red-50 text-red-900",
  ok: "border-green-200 bg-green-50 text-green-900",
};
const NOTICE_ICON: Record<NoticeTone, LucideIcon> = { info: Info, warn: AlertTriangle, error: OctagonX, ok: CheckCircle2 };

export function Notice({ tone = "info", title, children, action }: { tone?: NoticeTone; title?: string; children?: ReactNode; action?: ReactNode }) {
  const Icon = NOTICE_ICON[tone];
  return (
    <div role={tone === "error" ? "alert" : "status"} className={cx("flex items-start gap-3 rounded-lg border px-4 py-3", NOTICE_STYLE[tone])}>
      <Icon size={18} className="mt-0.5 shrink-0" aria-hidden />
      <div className="min-w-0 flex-1">
        {title && <strong className="block font-semibold">{title}</strong>}
        {children}
      </div>
      {action}
    </div>
  );
}

const ERROR_TITLE: Record<ApiError["kind"], string> = {
  offline: "Cannot reach the validation API",
  unauthorized: "The API key was not accepted",
  unconfigured: "The API is not ready",
  invalid: "The request was rejected",
  busy: "The gate is busy",
  notfound: "Not found",
  forbidden: "Not allowed on this server",
  toolarge: "The file is too large",
  server: "The API reported an error",
};

/** An API failure shown with what to do about it. */
export function ErrorNotice({ error, onRetry }: { error: ApiError; onRetry?: () => void }) {
  const { openDialog } = useConnection();
  const action =
    error.kind === "unauthorized" ? <Button small onClick={openDialog}>Enter API key</Button>
    : onRetry ? <Button small onClick={onRetry}>Try again</Button>
    : undefined;
  return (
    <Notice tone={error.kind === "busy" ? "warn" : "error"} title={ERROR_TITLE[error.kind]} action={action}>
      <span>{error.message}</span>
      {error.kind === "offline" && (
        <span className="mt-1 block text-[13px]">Start it with <code className="font-mono">python -m mqvg.api</code> (see docs/api.md).</span>
      )}
    </Notice>
  );
}

/** Shown on every page when the API cannot be used yet; nothing while connecting or connected. */
export function ConnectNotice() {
  const { status, message, openDialog } = useConnection();
  if (status === "connected" || status === "checking") return null;
  const title = status === "offline" ? "Cannot reach the validation API"
    : status === "unauthorized" ? "Enter the API key to continue" : "The API is not ready";
  return (
    <Notice tone={status === "offline" ? "error" : "warn"} title={title}
      action={status !== "offline" ? <Button small onClick={openDialog}>Connection settings</Button> : undefined}>
      <span>{message}</span>
      {status === "offline" && (
        <span className="mt-1 block text-[13px]">Start it with <code className="font-mono">python -m mqvg.api</code> (see docs/api.md).</span>
      )}
    </Notice>
  );
}

/* ----------------------------------------------------------- page furniture */
export function PageHeader({ title, subtitle, actions, crumb }: { title: string; subtitle?: ReactNode; actions?: ReactNode; crumb?: ReactNode }) {
  return (
    <header className="flex flex-wrap items-start justify-between gap-4">
      <div className="min-w-0">
        {crumb && <div className="mb-1.5 text-[13px] text-slate-500">{crumb}</div>}
        <h1 className="text-[22px] font-bold tracking-tight text-slate-900">{title}</h1>
        {subtitle && <p className="mt-1 max-w-3xl text-slate-500">{subtitle}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </header>
  );
}

export function EmptyState({ icon: Icon = SearchX, title, children, action }: { icon?: LucideIcon; title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="grid justify-items-center gap-2 px-5 py-12 text-center text-slate-500">
      <Icon size={36} className="text-blue-400" aria-hidden />
      <h3 className="text-base font-semibold text-slate-800">{title}</h3>
      {children && <p className="max-w-[56ch]">{children}</p>}
      {action}
    </div>
  );
}

/** A feature the API or the data cannot provide: say so, and what would enable it. Never a made-up value. */
export function Unavailable({ title, children, hint }: { title: string; children?: ReactNode; hint?: ReactNode }) {
  return (
    <div role="status" className="rounded-xl border border-dashed border-slate-300 bg-white px-5 py-8 text-center">
      <PlugZap size={30} className="mx-auto text-slate-400" aria-hidden />
      <h3 className="mt-2 text-base font-semibold text-slate-800">{title}</h3>
      {children && <p className="mx-auto mt-1 max-w-[62ch] text-slate-500">{children}</p>}
      {hint && <p className="mx-auto mt-2 max-w-[62ch] text-[13px] text-slate-500">{hint}</p>}
    </div>
  );
}

export const Spinner = ({ label }: { label?: string }) => (
  <span className="inline-flex items-center gap-2">
    <span className="h-[18px] w-[18px] animate-spin rounded-full border-[2.5px] border-blue-100 border-t-blue-600" aria-hidden />
    {label}
  </span>
);

export const Skeleton = ({ className }: { className?: string }) => <div className={cx("animate-pulse rounded-md bg-slate-200/80", className)} aria-hidden />;

export function CardSkeleton({ rows = 4 }: { rows?: number }) {
  return (
    <div className="grid gap-3 rounded-xl border border-slate-200 bg-white p-5" aria-busy="true" aria-label="Loading">
      <Skeleton className="h-5 w-2/5" />
      {Array.from({ length: rows }, (_, i) => <Skeleton key={i} className="h-4" />)}
    </div>
  );
}

export function StatCard({ icon: Icon, label, value, note }: { icon: LucideIcon; label: string; value: ReactNode; note?: ReactNode }) {
  return (
    <Card className="px-5 py-4">
      <div className="flex items-center gap-2 text-[13px] font-medium text-slate-500"><Icon size={16} className="text-blue-600" aria-hidden /><span data-stat-label>{label}</span></div>
      <div className="mt-1.5 text-3xl font-bold tracking-tight text-slate-900 tabular-nums">{value}</div>
      {note && <div className="mt-0.5 text-xs text-slate-500">{note}</div>}
    </Card>
  );
}

/* ---------------------------------------------------------------------- tabs */
export function Tabs<T extends string>({ tabs, value, onChange, label }: {
  tabs: { id: T; label: string; count?: number }[]; value: T; onChange: (id: T) => void; label: string;
}) {
  const list = useRef<HTMLDivElement>(null);
  const onKey = (e: React.KeyboardEvent) => {
    const i = tabs.findIndex((t) => t.id === value);
    const next = e.key === "ArrowRight" ? (i + 1) % tabs.length : e.key === "ArrowLeft" ? (i - 1 + tabs.length) % tabs.length : null;
    if (next === null) return;
    e.preventDefault();
    onChange(tabs[next].id);
    list.current?.querySelectorAll<HTMLButtonElement>("[role=tab]")[next]?.focus();
  };
  return (
    <div role="tablist" aria-label={label} ref={list} onKeyDown={onKey} className="flex gap-1 overflow-x-auto border-b border-slate-200">
      {tabs.map((t) => (
        <button key={t.id} role="tab" id={`tab-${t.id}`} aria-selected={value === t.id} aria-controls={`panel-${t.id}`}
          tabIndex={value === t.id ? 0 : -1} onClick={() => onChange(t.id)}
          className={cx("-mb-px inline-flex items-center gap-2 whitespace-nowrap border-b-2 px-3.5 py-2.5 font-semibold",
            value === t.id ? "border-blue-600 text-blue-700" : "border-transparent text-slate-500 hover:text-blue-800")}>
          {t.label}
          {t.count !== undefined && <span className="rounded-full bg-blue-50 px-2 text-[11.5px] font-bold text-blue-800">{t.count}</span>}
        </button>
      ))}
    </div>
  );
}

export const TabPanel = ({ id, children }: { id: string; children: ReactNode }) => (
  <div role="tabpanel" id={`panel-${id}`} aria-labelledby={`tab-${id}`} className="grid gap-4">{children}</div>
);

/* --------------------------------------------------------------- forms, tables */
export const inputClass =
  "h-9 w-full rounded-lg border border-slate-300 bg-white px-3 text-slate-900 placeholder:text-slate-400 hover:border-blue-400 focus:border-blue-600 focus:outline-2 focus:outline-blue-200 disabled:bg-slate-100";

export function Field({ id, label, optional, hint, className, children }: {
  id: string; label: string; optional?: boolean; hint?: ReactNode; className?: string; children: ReactNode;
}) {
  return (
    <div className={cx("grid content-start gap-1.5", className)}>
      <label htmlFor={id} className="text-[13px] font-semibold text-slate-700">
        {label}{optional && <span className="font-normal text-slate-500"> (optional)</span>}
      </label>
      {children}
      {hint && <span className="text-[12.5px] text-slate-500">{hint}</span>}
    </div>
  );
}

export const TableWrap = ({ children }: { children: ReactNode }) => <div className="overflow-x-auto">{children}</div>;
export const Table = ({ children, caption }: { children: ReactNode; caption?: string }) => (
  <table className="w-full min-w-160 border-collapse text-left">
    {caption && <caption className="sr-only">{caption}</caption>}
    {children}
  </table>
);
export const Th = ({ children, right, className }: { children?: ReactNode; right?: boolean; className?: string }) => (
  <th scope="col" className={cx("whitespace-nowrap border-b border-slate-200 bg-slate-50 px-4 py-2.5 text-xs font-semibold uppercase tracking-wide text-slate-500", right && "text-right", className)}>{children}</th>
);
export const Td = ({ children, right, className, colSpan }: { children?: ReactNode; right?: boolean; className?: string; colSpan?: number }) => (
  <td colSpan={colSpan} className={cx("border-b border-slate-100 px-4 py-2.5 align-top", right && "text-right tabular-nums", className)}>{children}</td>
);

export const Mono = ({ children, title }: { children: ReactNode; title?: string }) => <code title={title} className="font-mono text-[12.5px]">{children}</code>;
