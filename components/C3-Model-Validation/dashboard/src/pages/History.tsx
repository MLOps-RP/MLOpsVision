import { ChevronLeft, ChevronRight, Download, History as HistoryIcon, Search } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { DECISIONS, type Decision } from "../api/types";
import { DecisionBadge } from "../components/badges";
import { Gate } from "../components/Gate";
import {
  Button, buttonClass, Card, ConnectNotice, cx, EmptyState, inputClass, Mono, PageHeader, Table, TableWrap, Td, Th,
} from "../components/ui";
import { toCsv } from "../lib/csv";
import { downloadText, fmtDate, fmtScore } from "../lib/format";
import { useModelNames } from "../lib/modelNames";
import { useRuns } from "../lib/runs";

const PAGE = 15;
type Sort = "newest" | "oldest" | "mqs-high" | "mqs-low";
const SORTS: [Sort, string][] = [["newest", "Newest first"], ["oldest", "Oldest first"], ["mqs-high", "MQS high to low"], ["mqs-low", "MQS low to high"]];

const chip = (on: boolean) => cx("rounded-full border px-3 py-1 text-[13px] font-semibold",
  on ? "border-blue-600 bg-blue-600 text-white" : "border-slate-300 bg-white text-slate-700 hover:border-blue-400");

export function History() {
  const { runs, loading, error, reload } = useRuns();
  const { nameOf } = useModelNames();
  const [decision, setDecision] = useState<Decision | "all">("all");
  const [profile, setProfile] = useState("all");
  const [sort, setSort] = useState<Sort>("newest");
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(0);
  const navigate = useNavigate();

  const profiles = useMemo(() => [...new Set(runs.map((r) => r.profile))].sort(), [runs]);
  const rows = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const filtered = runs
      .filter((r) => decision === "all" || r.decision === decision)
      .filter((r) => profile === "all" || r.profile === profile)
      .filter((r) => !needle || (r.candidate_id ?? "").toLowerCase().includes(needle) || nameOf(r.candidate_id, r.run_id).toLowerCase().includes(needle) || r.run_id.includes(needle)
        || r.profile.toLowerCase().includes(needle) || r.decision.toLowerCase().includes(needle));
    const time = (r: (typeof runs)[number]) => new Date(r.created_at_utc).getTime();
    const mqs = (r: (typeof runs)[number], missing: number) => (r.mqs === null ? missing : r.mqs);   // no score sorts last
    return [...filtered].sort(
      sort === "oldest" ? (a, b) => time(a) - time(b)
      : sort === "mqs-high" ? (a, b) => mqs(b, -1) - mqs(a, -1)
      : sort === "mqs-low" ? (a, b) => mqs(a, 1e9) - mqs(b, 1e9)
      : (a, b) => time(b) - time(a));
  }, [runs, decision, profile, query, sort]);

  useEffect(() => setPage(0), [decision, profile, query, sort]);
  const pages = Math.max(1, Math.ceil(rows.length / PAGE));
  const shown = rows.slice(page * PAGE, page * PAGE + PAGE);

  const exportCsv = () => downloadText("validation-history.csv",
    toCsv(["run_id", "model", "candidate", "profile", "decision", "mqs", "created_at_utc"],
      rows.map((r) => [r.run_id, nameOf(r.candidate_id, r.run_id), r.candidate_id, r.profile, r.decision, r.mqs === null ? "" : r.mqs.toFixed(2), r.created_at_utc])), "text/csv");

  return (
    <>
      <PageHeader title="Validation History" subtitle="Every validation the API has stored. Search, filter, sort and open one for its full result."
        actions={<Link to="/validate" className={buttonClass("primary")}>Validate a model</Link>} />
      <ConnectNotice />
      <Gate loading={loading} error={error} onRetry={reload} rows={6}>
        <Card>
          <div className="flex flex-wrap items-center gap-3 border-b border-slate-200 px-5 py-3.5">
            <label className="relative min-w-50 max-w-xs flex-1">
              <span className="sr-only">Search runs</span>
              <Search size={15} aria-hidden className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
              <input type="search" placeholder="Search model, run id, profile or decision" value={query} onChange={(e) => setQuery(e.target.value)}
                className={cx(inputClass, "pl-9")} />
            </label>
            <div className="flex flex-wrap gap-1.5" role="group" aria-label="Filter by decision">
              <button className={chip(decision === "all")} aria-pressed={decision === "all"} onClick={() => setDecision("all")}>All</button>
              {DECISIONS.map((d) => <button key={d} className={chip(decision === d)} aria-pressed={decision === d} onClick={() => setDecision(d)}>{d}</button>)}
            </div>
            <label className="flex items-center gap-2 text-[13px] text-slate-600">Profile
              <select className={cx(inputClass, "h-8 w-auto py-0")} value={profile} onChange={(e) => setProfile(e.target.value)}>
                <option value="all">All profiles</option>
                {profiles.map((p) => <option key={p} value={p}>{p}</option>)}
              </select>
            </label>
            <label className="flex items-center gap-2 text-[13px] text-slate-600">Sort
              <select className={cx(inputClass, "h-8 w-auto py-0")} value={sort} onChange={(e) => setSort(e.target.value as Sort)}>
                {SORTS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
              </select>
            </label>
            <span className="ml-auto text-[13px] text-slate-500">{rows.length} of {runs.length}</span>
            <Button small onClick={exportCsv} disabled={rows.length === 0}><Download size={14} aria-hidden />Export CSV</Button>
          </div>

          {rows.length === 0 ? (
            <EmptyState icon={HistoryIcon} title={runs.length === 0 ? "No validations yet" : "No runs match these filters"}>
              {runs.length === 0 ? "Validated models appear here." : "Try clearing the search or the filters."}
            </EmptyState>
          ) : (
            <>
              <TableWrap>
                <Table caption="Validation runs">
                  <thead><tr><Th>Model</Th><Th>Decision</Th><Th right>MQS</Th><Th>Profile</Th><Th>Run id</Th><Th>Created</Th></tr></thead>
                  <tbody>
                    {shown.map((r) => (
                      <tr key={r.run_id} className="cursor-pointer hover:bg-slate-50" onClick={() => navigate(`/results/${r.run_id}`)}>
                        <Td><Link to={`/results/${r.run_id}`} className="font-semibold text-blue-700 hover:underline" onClick={(e) => e.stopPropagation()}>{r.candidate_id ? nameOf(r.candidate_id, r.run_id) : "(unnamed)"}</Link></Td>
                        <Td><DecisionBadge decision={r.decision} /></Td>
                        <Td right>{fmtScore(r.mqs)}</Td>
                        <Td>{r.profile}</Td>
                        <Td><Mono>{r.run_id.slice(0, 12)}</Mono></Td>
                        <Td className="whitespace-nowrap text-xs text-slate-500">{fmtDate(r.created_at_utc)}</Td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
              </TableWrap>
              {pages > 1 && (
                <nav aria-label="Pages" className="flex items-center justify-between border-t border-slate-200 px-5 py-3 text-[13px]">
                  <span className="text-slate-500">Page {page + 1} of {pages}</span>
                  <div className="flex gap-2">
                    <Button small disabled={page === 0} onClick={() => setPage(page - 1)}><ChevronLeft size={14} aria-hidden />Previous</Button>
                    <Button small disabled={page >= pages - 1} onClick={() => setPage(page + 1)}>Next<ChevronRight size={14} aria-hidden /></Button>
                  </div>
                </nav>
              )}
            </>
          )}
        </Card>
      </Gate>
    </>
  );
}
