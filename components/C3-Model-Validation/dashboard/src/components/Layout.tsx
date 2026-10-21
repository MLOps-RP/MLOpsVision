import {
  BarChart3, FileText, History, KeyRound, LayoutDashboard, Menu, PlayCircle, ScanSearch, ShieldCheck, ClipboardCheck, X,
} from "lucide-react";
import { Suspense, useEffect, useRef, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";

import { useConnection, type ConnectionStatus } from "../state/connection";
import { Badge, Button, buttonClass, CardSkeleton, cx, inputClass } from "./ui";

const NAV = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard, end: true },
  { to: "/validate", label: "Validate Model", icon: PlayCircle },
  { to: "/results", label: "Validation Results", icon: ClipboardCheck },
  { to: "/history", label: "Validation History", icon: History },
  { to: "/hidden-failures", label: "Hidden Failure Analysis", icon: ScanSearch },
  { to: "/reports", label: "Reports & Catalogue", icon: FileText },
];

const PILL: Record<ConnectionStatus, { label: string; dot: string }> = {
  checking: { label: "Connecting…", dot: "bg-amber-400" },
  connected: { label: "API connected", dot: "bg-green-500 ring-4 ring-green-100" },
  unauthorized: { label: "API key needed", dot: "bg-red-500 ring-4 ring-red-100" },
  unconfigured: { label: "API not ready", dot: "bg-red-500 ring-4 ring-red-100" },
  offline: { label: "API offline", dot: "bg-red-500 ring-4 ring-red-100" },
};

export function Layout() {
  const { status, openDialog } = useConnection();
  const [menu, setMenu] = useState(false);
  const location = useLocation();
  useEffect(() => setMenu(false), [location.pathname]);
  const pill = PILL[status];

  const sidebar = (
    <div className="flex h-full flex-col gap-6 px-3.5 py-5">
      <div className="flex items-center gap-3 px-2">
        <div className="grid size-9 place-items-center rounded-[10px] bg-linear-to-br from-blue-600 to-blue-800 text-white shadow-md shadow-blue-600/30"><ShieldCheck size={20} aria-hidden /></div>
        <div>
          <div className="text-[15px] font-bold tracking-tight">Validation Gate</div>
          <div className="text-xs text-slate-500">Model Quality · MLOps</div>
        </div>
      </div>
      <nav aria-label="Main" className="grid gap-1">
        {NAV.map(({ to, label, icon: Icon, end }) => (
          <NavLink key={to} to={to} end={end}
            className={({ isActive }) => cx("flex items-center gap-3 rounded-lg px-3 py-2 font-semibold",
              isActive ? "bg-blue-50 text-blue-700 shadow-[inset_3px_0_0_var(--color-blue-600)]" : "text-slate-600 hover:bg-blue-50 hover:text-blue-800")}>
            <Icon size={18} aria-hidden />{label}
          </NavLink>
        ))}
      </nav>
    </div>
  );

  return (
    <div className="min-h-full lg:grid lg:grid-cols-[256px_1fr]">
      <aside className="sticky top-0 hidden h-screen border-r border-slate-200 bg-white lg:block">{sidebar}</aside>
      {menu && (
        <div className="fixed inset-0 z-40 lg:hidden" role="dialog" aria-modal="true" aria-label="Navigation">
          <div className="absolute inset-0 bg-slate-900/45" onClick={() => setMenu(false)} />
          <div className="absolute inset-y-0 left-0 w-72 bg-white shadow-2xl">
            <button className="absolute right-3 top-3 rounded-md p-1.5 text-slate-500 hover:bg-slate-100" onClick={() => setMenu(false)} aria-label="Close navigation"><X size={18} /></button>
            {sidebar}
          </div>
        </div>
      )}
      <div className="flex min-w-0 flex-col">
        <header className="sticky top-0 z-30 flex items-center justify-between gap-3 border-b border-slate-200 bg-white/92 px-4 py-3 backdrop-blur sm:px-7">
          <div className="flex items-center gap-3">
            <button className="rounded-md p-1.5 text-slate-600 hover:bg-slate-100 lg:hidden" onClick={() => setMenu(true)} aria-label="Open navigation"><Menu size={20} /></button>
            <Badge tone="blue" className="hidden sm:inline-flex"><BarChart3 size={13} aria-hidden />Automated Model Quality &amp; Validation Gate</Badge>
          </div>
          <button onClick={openDialog} aria-label={`${pill.label}. Open connection settings`}
            className="inline-flex items-center gap-2 whitespace-nowrap rounded-full border border-slate-300 bg-white px-3 py-1.5 text-[13px] font-semibold hover:bg-slate-50">
            <span className={cx("size-2.5 rounded-full", pill.dot)} aria-hidden />{pill.label}<KeyRound size={14} aria-hidden />
          </button>
        </header>
        <main id="content" className="mx-auto grid w-full max-w-330 gap-5 px-4 pb-12 pt-6 sm:px-7"><Suspense fallback={<CardSkeleton rows={6} />}><Outlet /></Suspense></main>
      </div>
      <ConnectionDialog />
    </div>
  );
}

function ConnectionDialog() {
  const { dialogOpen, closeDialog, setKey, clearKey, hasKey, status, message } = useConnection();
  const [value, setValue] = useState("");
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!dialogOpen) return;
    setValue("");
    input.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && closeDialog();
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [dialogOpen, closeDialog]);

  if (!dialogOpen) return null;
  const save = (e: React.FormEvent) => {
    e.preventDefault();
    if (value.trim()) {
      setKey(value.trim());
      closeDialog();
    }
  };
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-slate-900/45 p-5" onMouseDown={(e) => e.target === e.currentTarget && closeDialog()}>
      <form className="grid w-full max-w-md gap-4 rounded-2xl bg-white p-6 shadow-2xl" role="dialog" aria-modal="true" aria-labelledby="conn-title" onSubmit={save}>
        <div>
          <h2 id="conn-title" className="text-base font-semibold">API connection</h2>
          <p className="mt-1 text-[13px] text-slate-500">Status: <strong>{PILL[status].label}</strong>{message ? ` — ${message}` : ""}</p>
        </div>
        <div className="grid gap-1.5">
          <label htmlFor="api-key" className="text-[13px] font-semibold text-slate-700">API key</label>
          <input id="api-key" ref={input} type="password" autoComplete="off" value={value} className={inputClass}
            onChange={(e) => setValue(e.target.value)} placeholder={hasKey ? "A key is saved for this tab" : "The server's MQVG_API_KEY"} />
          <span className="text-[12.5px] text-slate-500">Kept in this browser tab only (session storage) and cleared when the tab closes. It is never stored in the build.</span>
        </div>
        <div className="flex items-center justify-between gap-2">
          {hasKey ? <Button variant="ghost" onClick={() => { clearKey(); closeDialog(); }}>Forget key</Button> : <span />}
          <div className="flex gap-2">
            <Button onClick={closeDialog}>Cancel</Button>
            <button type="submit" className={buttonClass("primary")} disabled={!value.trim()}>Save and connect</button>
          </div>
        </div>
      </form>
    </div>
  );
}
