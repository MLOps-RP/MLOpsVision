import type { ReactNode } from "react";

import type { ApiError } from "../api/client";
import { useConnection } from "../state/connection";
import { CardSkeleton, ErrorNotice } from "./ui";

/**
 * The one place that decides what to show before data exists: a skeleton while the connection is
 * being checked or the request is running, nothing when the API cannot be used (the page shows a
 * ConnectNotice), the error if the request failed, otherwise the data. Never an empty state too early.
 */
export function Gate({ loading, error, onRetry, rows = 5, children }: {
  loading: boolean; error: ApiError | null; onRetry?: () => void; rows?: number; children: ReactNode;
}) {
  const { status } = useConnection();
  if (status === "checking" || (status === "connected" && loading)) return <CardSkeleton rows={rows} />;
  if (status !== "connected") return null;
  if (error) return <ErrorNotice error={error} onRetry={onRetry} />;
  return <>{children}</>;
}
