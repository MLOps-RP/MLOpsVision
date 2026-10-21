import { CheckCircle2, FileText, Play, ShieldCheck, Star } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

import { ApiError } from "../api/client";
import type { CandidateInfo, ProfileInfo } from "../api/types";
import { Gate } from "../components/Gate";
import {
  Badge, Button, buttonClass, Card, CardBody, CardHeader, ConnectNotice, cx, EmptyState, ErrorNotice, Notice, PageHeader, Spinner,
  type Tone, Unavailable,
} from "../components/ui";
import { shortPath } from "../lib/format";
import { modelNumbers } from "../lib/modelNames";
import { useAsync } from "../lib/useAsync";
import { useConnection } from "../state/connection";

const STRICTNESS: Record<ProfileInfo["strictness"], { label: string; tone: Tone }> = {
  lenient: { label: "Lenient", tone: "slate" }, standard: { label: "Standard", tone: "blue" }, strict: { label: "Strict", tone: "violet" },
};

/** "extended_hidden_groups" -> "Extended hidden groups". The name itself stays visible as secondary text. */
const title = (name: string) => name.charAt(0).toUpperCase() + name.slice(1).replaceAll("_", " ");

const selectable = (selected: boolean, disabled = false) => cx(
  "relative flex cursor-pointer gap-3 rounded-xl border bg-white p-4 shadow-sm transition focus-within:ring-2 focus-within:ring-blue-300",
  selected ? "border-blue-600 bg-blue-50/40 ring-2 ring-blue-200" : "border-slate-200 hover:border-blue-300",
  disabled && "cursor-not-allowed opacity-60 hover:border-slate-200");

export function ValidateModel() {
  const { client, status } = useConnection();
  const candidates = useAsync(() => (status === "connected" ? client.candidates() : Promise.resolve(null)), [client, status]);
  const profiles = useAsync(() => (status === "connected" ? client.profiles() : Promise.resolve(null)), [client, status]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [profileChoice, setProfileChoice] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<ApiError | null>(null);
  const abort = useRef<AbortController | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    if (!running) return;
    const started = Date.now();
    const timer = setInterval(() => setElapsed(Math.floor((Date.now() - started) / 1000)), 500);
    return () => clearInterval(timer);
  }, [running]);

  const list = useMemo(() => candidates.data?.candidates ?? [], [candidates.data]);
  const profileList = useMemo(() => profiles.data?.profiles ?? [], [profiles.data]);
  const allowed = profileList.filter((p) => p.allowed);
  const restricted = profileList.filter((p) => !p.allowed);
  const numbers = useMemo(() => modelNumbers(list), [list]);
  const label = (c: CandidateInfo) => numbers.get(c.id) ?? c.id;             // the same name every page uses
  const selected = list.find((c) => c.id === selectedId && c.available) ?? null;

  // The server's own default if it will run it; otherwise the first profile it will run.
  const defaultProfile = allowed.find((p) => p.name === profiles.data?.default_profile) ?? allowed[0] ?? null;
  const suggested = selected?.suggested_profile ? allowed.find((p) => p.name === selected.suggested_profile) ?? null : null;
  const effective = allowed.find((p) => p.name === profileChoice) ?? suggested ?? defaultProfile;

  const choose = (c: CandidateInfo) => { setSelectedId(c.id); setProfileChoice(null); setError(null); };
  const ready = selected !== null && effective !== null && !running && status === "connected";

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!ready || !selected || !effective) return;
    setError(null);
    setElapsed(0);
    setRunning(true);
    abort.current = new AbortController();
    try {
      // Exactly the files the server declares for this model; nothing is chosen or guessed here.
      const summary = await client.validate({
        candidate_id: selected.id, model_path: selected.model_path, eval_data_path: selected.eval_data_path,
        metadata_path: selected.metadata_path, train_row_hashes_path: selected.train_row_hashes_path, profile: effective.name,
      }, abort.current.signal);
      navigate(`/results/${summary.run_id}`);
    } catch (failure) {
      if (failure instanceof DOMException && failure.name === "AbortError") return;
      setError(failure instanceof ApiError ? failure : new ApiError("server", 0, String(failure)));
    } finally {
      setRunning(false);
    }
  }

  const usable = list.filter((c) => c.available);
  const unusable = list.filter((c) => !c.available);
  const reasons = new Map<string, string[]>();
  for (const p of restricted) reasons.set(p.unavailable_reason ?? "This server does not allow it.", [...(reasons.get(p.unavailable_reason ?? "This server does not allow it.") ?? []), p.name]);

  return (
    <>
      <PageHeader title="Validate Model"
        subtitle="Choose a model, choose how it should be checked, and run the validation. The answer is a recommendation; nothing is deployed." />
      <ConnectNotice />

      <Gate loading={candidates.loading || profiles.loading} error={candidates.error ?? profiles.error}
        onRetry={() => { candidates.reload(); profiles.reload(); }} rows={7}>
        <form onSubmit={submit} aria-label="Validation request" className="grid gap-4">
          <fieldset disabled={running} className="m-0 grid min-w-0 gap-4 border-0 p-0">
            {/* ------------------------------------------------------------ models */}
            <Card>
              <CardHeader title="1. Choose a model" subtitle="Each model comes with its own evaluation data and training details, set by the server." />
              <CardBody>
                {candidates.data?.error ? (
                  <Notice tone="error" title="The list of models could not be used">{candidates.data.error}</Notice>
                ) : !candidates.data?.catalogue_found ? (
                  <EmptyState title="No models are registered on this server">
                    The server has no list of models to validate (a <code className="font-mono">candidates.yaml</code> file in its artifact root).
                    An administrator can add one; see docs/api.md.
                  </EmptyState>
                ) : list.length === 0 ? (
                  <EmptyState title="The list of models is empty">Ask an administrator to add models to the server's candidate list.</EmptyState>
                ) : (
                  <div className="grid gap-4">
                    {usable.length === 0 && <Notice tone="warn" title="None of the listed models can be used right now">The reasons are shown on each model below.</Notice>}
                    <div role="radiogroup" aria-label="Models" className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                      {usable.map((c) => <CandidateCard key={c.id} c={c} name={label(c)} selected={c.id === selected?.id} onChoose={() => choose(c)} />)}
                    </div>
                    {unusable.length > 0 && (
                      <div className="grid gap-3">
                        <h3 className="text-[13px] font-semibold text-slate-600">Not available right now ({unusable.length})</h3>
                        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                          {unusable.map((c) => <CandidateCard key={c.id} c={c} name={label(c)} selected={false} onChoose={() => {}} />)}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </CardBody>
            </Card>

            {/* ---------------------------------------------------------- profiles */}
            <Card>
              <CardHeader title="2. Choose how to check it"
                subtitle="A profile decides which checks run and how demanding the verdict is. These are the ones this server will run." />
              <CardBody className="grid gap-4">
                {allowed.length === 0 ? (
                  <Unavailable title="No validation profile is available on this server"
                    hint="The server only runs profiles that scan the model file before loading it, and none is configured.">
                    A validation cannot be started until an administrator makes one available.
                  </Unavailable>
                ) : (
                  <>
                    {selected && suggested && (
                      <p className="flex items-center gap-2 text-[13px] text-slate-600">
                        <Star size={14} className="text-amber-500" aria-hidden />
                        Suggested for {label(selected)}: <strong>{title(suggested.name)}</strong>
                      </p>
                    )}
                    {selected?.suggested_profile && !suggested && (
                      <Notice tone="info">The profile suggested for {label(selected)} ({selected.suggested_profile}) is not available on this server.</Notice>
                    )}
                    <div role="radiogroup" aria-label="Validation profiles" className="grid gap-3 lg:grid-cols-2">
                      {allowed.map((p) => (
                        <ProfileCard key={p.name} p={p} selected={p.name === effective?.name} isDefault={p.name === profiles.data?.default_profile}
                          onChoose={() => setProfileChoice(p.name)} />
                      ))}
                    </div>
                  </>
                )}
                {restricted.length > 0 && (
                  <details className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-[13px]">
                    <summary className="font-semibold text-slate-700">Not available on this server ({restricted.length})</summary>
                    <ul className="mt-3 grid list-none gap-3 p-0">
                      {[...reasons].map(([reason, names]) => (
                        <li key={reason}>
                          <div className="flex flex-wrap gap-1.5">{names.map((n) => <Badge key={n} tone="slate">{n}</Badge>)}</div>
                          <p className="mt-1 text-slate-600">{reason}</p>
                        </li>
                      ))}
                    </ul>
                  </details>
                )}
              </CardBody>
            </Card>
          </fieldset>

          {error && <ErrorNotice error={error} />}

          <Card className="flex flex-wrap items-center justify-between gap-3 px-5 py-4">
            <div className="min-w-0 text-[13px] text-slate-600" aria-live="polite">
              {running ? <Spinner label={`Validating ${selected ? label(selected) : "the model"}… ${elapsed}s. This can take up to a minute.`} />
                : selected && effective ? <>Ready: <strong>{label(selected)}</strong> ({shortPath(selected.model_path, 2)}) with the <strong>{title(effective.name)}</strong> profile.</>
                : "Choose a model to continue."}
            </div>
            <div className="flex items-center gap-2">
              {running && <Button onClick={() => abort.current?.abort()}>Stop waiting</Button>}
              <button type="submit" className={buttonClass("primary")} disabled={!ready}><Play size={15} aria-hidden />Validate model</button>
            </div>
          </Card>
          {running && <p className="text-[13px] text-slate-500">Stopping only ends the wait; the server may still finish the run, and it will appear in the history.</p>}
        </form>
      </Gate>
    </>
  );
}

function CandidateCard({ c, name, selected, onChoose }: { c: CandidateInfo; name: string; selected: boolean; onChoose: () => void }) {
  const disabled = !c.available;
  return (
    <label className={selectable(selected, disabled)}>
      <input type="radio" name="candidate" value={c.id} checked={selected} disabled={disabled} onChange={onChoose} className="sr-only"
        aria-labelledby={`cand-${c.id}-name`} aria-describedby={`cand-${c.id}-desc cand-${c.id}-file`} />
      <div className="min-w-0 flex-1">
        <div className="flex items-center justify-between gap-2">
          <span id={`cand-${c.id}-name`} className="text-base font-semibold text-slate-900">{name}</span>
          {selected && <CheckCircle2 size={18} className="shrink-0 text-blue-600" aria-hidden />}
          {disabled && <Badge tone="amber">Unavailable</Badge>}
        </div>
        <p id={`cand-${c.id}-desc`} className="mt-1 text-[13.5px] text-slate-700">{c.description}</p>
        <p id={`cand-${c.id}-file`} className="mt-2 flex items-center gap-1.5 text-xs text-slate-500" title={`Model file: ${c.model_path}`}>
          <FileText size={13} aria-hidden /><code className="font-mono">{shortPath(c.model_path, 2)}</code>
        </p>
        {disabled && c.unavailable_reason && <p className="mt-1.5 text-xs font-medium text-amber-800">Not available: {c.unavailable_reason}.</p>}
      </div>
    </label>
  );
}

function ProfileCard({ p, selected, isDefault, onChoose }: { p: ProfileInfo; selected: boolean; isDefault: boolean; onChoose: () => void }) {
  const level = STRICTNESS[p.strictness];
  return (
    <label className={selectable(selected)}>
      <input type="radio" name="profile" value={p.name} checked={selected} onChange={onChoose} className="sr-only"
        aria-labelledby={`prof-${p.name}-name`} aria-describedby={`prof-${p.name}-desc`} />
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2">
          <span id={`prof-${p.name}-name`} className="text-base font-semibold text-slate-900">{title(p.name)}</span>
          <Badge tone={level.tone}>{level.label}</Badge>
          {isDefault && <Badge tone="green">Server default</Badge>}
          {selected && <CheckCircle2 size={18} className="ml-auto shrink-0 text-blue-600" aria-hidden />}
        </div>
        <div id={`prof-${p.name}-desc`}>
          <p className="mt-1 text-[13.5px] text-slate-700">{p.summary}</p>
          <p className="mt-1.5 text-[12.5px] text-slate-500">{p.strictness_text}</p>
          {p.highlights.length > 0 && (
            <ul className="mt-2 grid list-none gap-1 p-0 text-[12.5px] text-slate-600">
              {p.highlights.map((h) => (
                <li key={h} className="flex items-start gap-1.5"><ShieldCheck size={13} className="mt-0.5 shrink-0 text-blue-600" aria-hidden />{h}</li>
              ))}
            </ul>
          )}
        </div>
        <p className="mt-2 text-xs text-slate-400"><code className="font-mono">{p.name}</code></p>
      </div>
    </label>
  );
}
