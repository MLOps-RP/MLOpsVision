import { render } from "@testing-library/react";
import type { ReactElement } from "react";
import { MemoryRouter } from "react-router-dom";

import { App } from "../App";
import { ConnectionProvider } from "../state/connection";
import { KEY } from "./mockApi";

// The pages are lazy-loaded. Compile them once, up front, so a test that navigates to a page is not
// measuring the time it takes to build Recharts and the Markdown renderer.
await Promise.all([
  import("../pages/Dashboard"), import("../pages/ValidateModel"), import("../pages/Results"), import("../pages/History"),
  import("../pages/Comparison"), import("../pages/HiddenFailures"), import("../pages/Reports"),
]);

/** Render the whole app at a route, with the API key already saved (unless `key` is null). */
export function renderApp(path = "/", key: string | null = KEY) {
  if (key !== null) sessionStorage.setItem("c3.apiKey", key);
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App apiBase="/api" />
    </MemoryRouter>,
  );
}

/** Render one component (for example a hidden page that has no route) with the router and connection it needs. */
export function renderPage(element: ReactElement, path = "/", key: string | null = KEY) {
  if (key !== null) sessionStorage.setItem("c3.apiKey", key);
  return render(
    <MemoryRouter initialEntries={[path]}>
      <ConnectionProvider base="/api">{element}</ConnectionProvider>
    </MemoryRouter>,
  );
}
