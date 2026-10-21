import type { CheckResult } from "../api/types";
import { fmtPercent } from "../lib/format";
import type { GroupPoint } from "./charts";
import { StatusBadge } from "./badges";
import { Badge, Card, CardHeader, Mono, Notice, Table, TableWrap, Td, Th } from "./ui";

export const HIDDEN_CHECK = "behaviour.hidden_failure_groups";

export interface HiddenDetails {
  groups: GroupPoint[]; groups_confirmed: number; groups_tested: number; methods: string[];
  alpha: number; min_effect: number; caveats: string[]; features_used: string[];
  share_of_rows_in_confirmed_groups: number | null;
}

/** The check's details are untyped on the wire; accept them only if they have the expected shape. */
export function readHiddenDetails(raw: Record<string, unknown> | undefined): HiddenDetails | null {
  if (!raw || !Array.isArray(raw.groups) || typeof raw.groups_tested !== "number") return null;
  return raw as unknown as HiddenDetails;
}

const signed = (x: number) => `${x >= 0 ? "+" : ""}${(x * 100).toFixed(1)}%`;

export function PolicyLine({ policy }: { policy?: { consequence?: string; require_assessment?: boolean } }) {
  if (!policy?.consequence) return null;
  return (
    <p className="text-[13px]">
      <strong>Configured consequence: <code className="font-mono">{policy.consequence}</code>.</strong>{" "}
      {policy.consequence === "score_only" && "A confirmed group lowers the score only."}
      {policy.consequence === "hold" && "A confirmed group blocks promotion (HOLD)."}
      {policy.consequence === "critical" && "A severe group is a critical failure (ROLLBACK recommendation)."}{" "}
      {policy.require_assessment ? "A result that could not be assessed blocks promotion." : "A result that could not be assessed is never counted as a pass."}
    </p>
  );
}

export function HiddenGroups({ check, policy }: { check: CheckResult; policy?: { consequence?: string; require_assessment?: boolean } }) {
  const details = readHiddenDetails(check.details);
  return (
    <div className="grid gap-4">
      <Card className="grid gap-2.5 px-5 py-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h2 className="text-base font-semibold">Hidden failure groups</h2>
            <p className="text-[13px] text-slate-500">
              Groups of rows, defined by feature values, where the model makes more errors than its own probabilities predict.
              Found on some rows and tested on different, held-out rows.
            </p>
          </div>
          <StatusBadge status={check.status} />
        </div>
        <PolicyLine policy={policy} />
      </Card>

      {!details ? (
        <Notice tone="warn" title="Not assessed">{check.message} This is not a pass.</Notice>
      ) : (
        <Card>
          <CardHeader title={`${details.groups_confirmed} confirmed of ${details.groups_tested} groups tested`}
            subtitle={<>
              Methods: {details.methods.join(", ")} · false discovery rate {details.alpha} · minimum effect {fmtPercent(details.min_effect)}
              {details.share_of_rows_in_confirmed_groups != null && <> · confirmed groups cover {fmtPercent(details.share_of_rows_in_confirmed_groups, 1)} of rows</>}
            </>} />
          {details.groups.length === 0 ? (
            <p className="px-5 py-4 text-slate-500">No candidate group was found.</p>
          ) : (
            <TableWrap>
              <Table>
                <thead>
                  <tr><Th>#</Th><Th>Group</Th><Th>Method</Th><Th right>Rows</Th><Th right>Observed</Th><Th right>Expected</Th>
                    <Th right>Excess (95% CI)</Th><Th right>Adjusted p</Th><Th>Confirmed</Th></tr>
                </thead>
                <tbody>
                  {details.groups.map((g, i) => (
                    <tr key={i}>
                      <Td className="font-semibold text-slate-500">G{i + 1}</Td>
                      <Td className="max-w-sm"><Mono>{g.rule}</Mono></Td>
                      <Td>{g.method}</Td>
                      <Td right>{g.n_rows}</Td>
                      <Td right>{fmtPercent(g.observed_error_rate, 1)}</Td>
                      <Td right>{fmtPercent(g.expected_error_rate, 1)}</Td>
                      <Td right>{signed(g.effect)} <span className="text-slate-500">[{signed(g.effect_ci[0])}, {signed(g.effect_ci[1])}]</span></Td>
                      <Td right>{g.adjusted_p_value.toExponential(1)}</Td>
                      <Td>{g.confirmed ? <Badge tone="red">Confirmed</Badge> : <Badge tone="slate">No</Badge>}</Td>
                    </tr>
                  ))}
                </tbody>
              </Table>
            </TableWrap>
          )}
          <div className="grid gap-1.5 border-t border-slate-200 px-5 py-4 text-[13px] text-slate-500">
            <p><strong>Features searched:</strong> {details.features_used.join(", ")}</p>
            {details.caveats.map((c, i) => <p key={i}>{c}</p>)}
          </div>
        </Card>
      )}
    </div>
  );
}
