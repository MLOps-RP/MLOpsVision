import { useMemo } from "react";

import type { CheckInfo } from "../api/types";
import { useConnection } from "../state/connection";
import { useAsync } from "./useAsync";

/** The check catalogue (ids, layers, units) keyed by check id. */
export function useCatalogue() {
  const { client, status } = useConnection();
  const state = useAsync(() => (status === "connected" ? client.checks() : Promise.resolve<CheckInfo[]>([])), [client, status]);
  const map = useMemo(() => new Map((state.data ?? []).map((c) => [c.id, c])), [state.data]);
  return { ...state, map };
}
