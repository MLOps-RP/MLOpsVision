import { useMemo } from "react";

import type { RunEntry, ValidationResult } from "../api/types";
import { useConnection } from "../state/connection";
import { useAsync } from "./useAsync";

/** The stored runs, newest first. Nothing is requested until the API connection works. */
export function useRuns() {
  const { client, status } = useConnection();
  const state = useAsync(() => (status === "connected" ? client.runs(200) : Promise.resolve(null)), [client, status]);
  const runs = useMemo<RunEntry[]>(() => state.data?.runs ?? [], [state.data]);
  return { ...state, runs };
}

/** One full result, or null while there is no id. */
export function useRun(id: string | undefined) {
  const { client, status } = useConnection();
  return useAsync<ValidationResult | null>(
    () => (status === "connected" && id ? client.run(id) : Promise.resolve(null)), [client, status, id]);
}
