import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError } from "../api/client";

export interface AsyncState<T> {
  data: T | null;
  error: ApiError | null;
  loading: boolean;
  reload: () => void;
}

interface Settled<T> { data: T | null; error: ApiError | null; forDeps: unknown[] }

const same = (a: unknown[], b: unknown[]) => a.length === b.length && a.every((v, i) => Object.is(v, b[i]));

/**
 * Run `load` on mount and whenever `deps` change; ignore answers that arrive too late.
 *
 * `loading` is derived from the dependencies during render: a result only counts as current if it was
 * produced for the dependencies in force right now. Setting it later, in the effect, would leave one
 * committed render showing the previous result (or "no data") for the new dependencies, which is how
 * a page can flash the wrong state the instant a connection becomes ready.
 */
export function useAsync<T>(load: () => Promise<T>, deps: unknown[]): AsyncState<T> {
  const [tick, setTick] = useState(0);
  const [settled, setSettled] = useState<Settled<T> | null>(null);
  const latest = useRef(0);
  const current = [...deps, tick];
  const loading = settled === null || !same(settled.forDeps, current);

  useEffect(() => {
    const id = ++latest.current;
    const forDeps = [...deps, tick];
    load().then(
      (data) => { if (id === latest.current) setSettled({ data, error: null, forDeps }); },
      (failure: unknown) => {
        if (id === latest.current) {
          setSettled({ data: null, error: failure instanceof ApiError ? failure : new ApiError("server", 0, String(failure)), forDeps });
        }
      },
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);

  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { data: settled?.data ?? null, error: loading ? null : settled?.error ?? null, loading, reload };
}
