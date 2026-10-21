import { useCallback } from "react";

import type { CandidateInfo } from "../api/types";
import { useConnection } from "../state/connection";
import { useAsync } from "./useAsync";

/** "Model 1", "Model 2", ...: the position in the server's model list. The same rule everywhere, so a model has one name. */
export function modelNumbers(list: Pick<CandidateInfo, "id">[]): Map<string, string> {
  return new Map(list.map((c, i) => [c.id, `Model ${i + 1}`]));
}

/**
 * Turns the candidate id stored on a run into the friendly name shown to people. The id (and the files behind it) stay
 * what the API sees; this only changes what is displayed. A run whose id is not in the server's list keeps its own id,
 * so a name is never invented for a model the list does not contain.
 */
export function useModelNames() {
  const { client, status } = useConnection();
  const list = useAsync(() => (status === "connected" ? client.candidates() : Promise.resolve(null)), [client, status]);
  const names = modelNumbers(list.data?.candidates ?? []);
  const nameOf = useCallback(
    (candidateId: string | null | undefined, runId: string) => (candidateId ? names.get(candidateId) ?? candidateId : runId.slice(0, 8)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [list.data],
  );
  /** For records that carry only the id (the study files). */
  const nameFor = useCallback((candidateId: string) => names.get(candidateId) ?? candidateId,
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [list.data]);
  return { nameOf, nameFor, loading: list.loading };
}

const escape = (text: string) => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/**
 * Shows a model's friendly name in a validation report: in the report's title and in the "Candidate" row of its details
 * table, and nowhere else. Only those two lines are touched, and only when they hold exactly the model's id, so other
 * text (for example a longer id that merely contains it) is never rewritten. The gate's own report on disk is unchanged.
 */
export function nameReport(markdown: string, candidateId: string | null | undefined, name: string): string {
  if (!candidateId || candidateId === name) return markdown;
  const id = escape(candidateId);
  return markdown
    .replace(new RegExp(`^(# Model Quality Validation Report: )(")?${id}(")?[ \\t]*$`, "m"), (_, head: string, open = "", close = "") => `${head}${open}${name}${close}`)
    .replace(new RegExp(`^(\\| Candidate \\| )${id}( \\|)[ \\t]*$`, "m"), (_, head: string, tail: string) => `${head}${name}${tail}`);
}
