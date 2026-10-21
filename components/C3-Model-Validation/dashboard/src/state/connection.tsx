import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import { ApiError, createClient, type Client } from "../api/client";

export type ConnectionStatus = "checking" | "connected" | "unauthorized" | "unconfigured" | "offline";

interface Connection {
  client: Client;
  status: ConnectionStatus;
  hasKey: boolean;
  /** The message behind a non-connected status. */
  message: string | null;
  setKey: (key: string) => void;
  clearKey: () => void;
  recheck: () => void;
  dialogOpen: boolean;
  openDialog: () => void;
  closeDialog: () => void;
}

// sessionStorage, not localStorage: the key disappears when the tab is closed.
const STORAGE_KEY = "c3.apiKey";

function readKey(): string | null {
  try {
    return sessionStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

const ConnectionContext = createContext<Connection | null>(null);

export function ConnectionProvider({ children, base }: { children: ReactNode; base?: string }) {
  const keyRef = useRef<string | null>(readKey());
  const [hasKey, setHasKey] = useState(keyRef.current !== null);
  const [status, setStatus] = useState<ConnectionStatus>("checking");
  const [message, setMessage] = useState<string | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [tick, setTick] = useState(0);
  const client = useMemo(() => createClient(() => keyRef.current, base), [base]);

  useEffect(() => {
    let current = true;
    setStatus("checking");
    // /health is open; /v1/profiles needs the key, so it tells us whether the key works.
    client
      .health()
      .then(() => client.profiles())
      .then(() => {
        if (current) {
          setStatus("connected");
          setMessage(null);
        }
      })
      .catch((error: unknown) => {
        if (!current) return;
        const failure = error instanceof ApiError ? error : null;
        setMessage(failure?.message ?? "The API did not answer.");
        setStatus(
          failure?.kind === "unauthorized" ? "unauthorized"
          : failure?.kind === "unconfigured" ? "unconfigured"
          : "offline",
        );
      });
    return () => {
      current = false;
    };
  }, [client, tick, hasKey]);

  const setKey = useCallback((key: string) => {
    keyRef.current = key || null;
    try {
      if (key) sessionStorage.setItem(STORAGE_KEY, key);
      else sessionStorage.removeItem(STORAGE_KEY);
    } catch {
      /* storage can be blocked; the key then lasts for this page load only */
    }
    setHasKey(Boolean(key));
    setTick((t) => t + 1);
  }, []);

  const value = useMemo<Connection>(
    () => ({
      client, status, hasKey, message, setKey,
      clearKey: () => setKey(""),
      recheck: () => setTick((t) => t + 1),
      dialogOpen,
      openDialog: () => setDialogOpen(true),
      closeDialog: () => setDialogOpen(false),
    }),
    [client, status, hasKey, message, setKey, dialogOpen],
  );
  return <ConnectionContext.Provider value={value}>{children}</ConnectionContext.Provider>;
}

export function useConnection(): Connection {
  const value = useContext(ConnectionContext);
  if (!value) throw new Error("useConnection must be used inside ConnectionProvider");
  return value;
}
