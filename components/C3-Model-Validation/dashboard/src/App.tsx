import { lazy } from "react";
import { Link, Route, Routes } from "react-router-dom";

import { Layout } from "./components/Layout";
import { buttonClass, Card, EmptyState } from "./components/ui";
import { ConnectionProvider } from "./state/connection";

// The Model Comparison page (pages/Comparison.tsx) is not routed: it is hidden, not deleted.
// Each page is its own chunk, so the charts and the Markdown renderer load only where they are used.
const Dashboard = lazy(() => import("./pages/Dashboard").then((m) => ({ default: m.Dashboard })));
const ValidateModel = lazy(() => import("./pages/ValidateModel").then((m) => ({ default: m.ValidateModel })));
const Results = lazy(() => import("./pages/Results").then((m) => ({ default: m.Results })));
const ResultsIndex = lazy(() => import("./pages/Results").then((m) => ({ default: m.ResultsIndex })));
const History = lazy(() => import("./pages/History").then((m) => ({ default: m.History })));
const HiddenFailures = lazy(() => import("./pages/HiddenFailures").then((m) => ({ default: m.HiddenFailures })));
const Reports = lazy(() => import("./pages/Reports").then((m) => ({ default: m.Reports })));

export function App({ apiBase }: { apiBase?: string }) {
  return (
    <ConnectionProvider base={apiBase}>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="validate" element={<ValidateModel />} />
          <Route path="results" element={<ResultsIndex />} />
          <Route path="results/:runId" element={<Results />} />
          <Route path="history" element={<History />} />
          <Route path="hidden-failures" element={<HiddenFailures />} />
          <Route path="reports" element={<Reports />} />
          <Route path="*" element={
            <Card><EmptyState title="Page not found" action={<Link to="/" className={buttonClass("primary")}>Back to the dashboard</Link>} /></Card>} />
        </Route>
      </Routes>
    </ConnectionProvider>
  );
}
