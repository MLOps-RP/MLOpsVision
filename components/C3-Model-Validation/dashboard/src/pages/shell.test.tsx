import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { installApi, KEY } from "../test/mockApi";
import { renderApp } from "../test/render";

afterEach(() => vi.unstubAllGlobals());

describe("Navigation", () => {
  it("offers the six pages in the sidebar", async () => {
    installApi({ key: KEY });
    renderApp("/history");
    await screen.findByRole("heading", { level: 1, name: "Validation History" });
    const nav = within(screen.getByRole("navigation", { name: "Main" }));
    const links = nav.getAllByRole("link");
    expect(links.map((l) => l.textContent)).toEqual([
      "Dashboard", "Validate Model", "Validation Results", "Validation History", "Hidden Failure Analysis", "Reports & Catalogue"]);
    expect(links.map((l) => l.getAttribute("href"))).toEqual(["/", "/validate", "/results", "/history", "/hidden-failures", "/reports"]);
    expect(nav.getByRole("link", { name: "Validation History" })).toHaveAttribute("aria-current", "page");
  });

  it("is named Validation Gate, without a C3 prefix, in the sidebar and the page title", async () => {
    installApi({ key: KEY });
    renderApp("/");
    await screen.findByRole("heading", { level: 1, name: "Dashboard" });
    expect(screen.getAllByText("Validation Gate").length).toBeGreaterThan(0);
    expect(screen.queryByText(/C3/)).not.toBeInTheDocument();
    const html = (await import("../../index.html?raw")).default;
    expect(html).toContain("<title>Validation Gate</title>");
    expect(html).not.toMatch(/C3/);
  });

  it("does not offer Model Comparison: no link, and its address is a not-found page", async () => {
    installApi({ key: KEY });
    renderApp("/comparison");
    expect(await screen.findByText("Page not found")).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Model Comparison" })).not.toBeInTheDocument();
    expect(screen.queryByText("Side by side")).not.toBeInTheDocument();
  });

  it("moves between pages", async () => {
    installApi({ key: KEY });
    const user = userEvent.setup();
    renderApp("/");
    await user.click(await screen.findByRole("link", { name: "Reports & Catalogue" }));
    expect(await screen.findByRole("heading", { level: 1, name: "Reports & Check Catalogue" })).toBeInTheDocument();
    await user.click(screen.getByRole("link", { name: "Hidden Failure Analysis" }));
    expect(await screen.findByRole("heading", { level: 1, name: "Hidden Failure Analysis" })).toBeInTheDocument();
  });

  it("has a menu for small screens", async () => {
    installApi({ key: KEY });
    const user = userEvent.setup();
    renderApp("/");
    await user.click(await screen.findByRole("button", { name: "Open navigation" }));
    const menu = screen.getByRole("dialog", { name: "Navigation" });
    expect(within(menu).getAllByRole("link")).toHaveLength(6);
    await user.click(within(menu).getByRole("link", { name: "Validation History" }));
    expect(await screen.findByRole("heading", { level: 1, name: "Validation History" })).toBeInTheDocument();
    expect(screen.queryByRole("dialog", { name: "Navigation" })).not.toBeInTheDocument();   // closes after choosing
  });

  it("shows a not-found page for an unknown route", async () => {
    installApi({ key: KEY });
    renderApp("/nope");
    expect(await screen.findByText("Page not found")).toBeInTheDocument();
  });
});

describe("API connection", () => {
  it("asks for a key, keeps it for the tab only, and then connects", async () => {
    const calls = installApi({ key: KEY });
    const user = userEvent.setup();
    renderApp("/", null);
    await user.click(await screen.findByRole("button", { name: /API key needed/ }));
    const dialog = screen.getByRole("dialog", { name: "API connection" });
    expect(within(dialog).getByText(/never stored in the build/)).toBeInTheDocument();
    await user.type(within(dialog).getByLabelText("API key"), KEY);
    await user.click(within(dialog).getByRole("button", { name: "Save and connect" }));
    expect(await screen.findByRole("button", { name: /API connected/ })).toBeInTheDocument();
    expect(sessionStorage.getItem("c3.apiKey")).toBe(KEY);
    expect(localStorage.getItem("c3.apiKey")).toBeNull();                       // never persisted beyond the tab
    expect(calls.at(-1)!.headers.get("X-API-Key")).toBe(KEY);
  });

  it("does not accept a wrong key", async () => {
    installApi({ key: KEY });
    const user = userEvent.setup();
    renderApp("/", "wrong-key");
    await user.click(await screen.findByRole("button", { name: /API key needed/ }));
    expect(screen.getByRole("dialog")).toHaveTextContent(/missing or wrong API key/);
  });

  it("closes the dialog with Escape and can forget the key", async () => {
    installApi({ key: KEY });
    const user = userEvent.setup();
    renderApp("/");
    await user.click(await screen.findByRole("button", { name: /API connected/ }));
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /API connected/ }));
    await user.click(screen.getByRole("button", { name: "Forget key" }));
    expect(sessionStorage.getItem("c3.apiKey")).toBeNull();
    expect(await screen.findByRole("button", { name: /API key needed/ })).toBeInTheDocument();
  });

  it("says plainly when the API cannot be reached", async () => {
    installApi({ offline: true });
    renderApp("/");
    expect(await screen.findByRole("button", { name: /API offline/ })).toBeInTheDocument();
    await waitFor(() => expect(screen.getAllByText("Cannot reach the validation API").length).toBeGreaterThan(0));
  });

  it("says when the server has no API key configured", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => String(input).endsWith("/health")
      ? new Response(JSON.stringify({ status: "ok", gate_version: "0.1.0", authentication: "not configured" }), { status: 200, headers: { "Content-Type": "application/json" } })
      : new Response(JSON.stringify({ detail: "the API key is not configured: set MQVG_API_KEY" }), { status: 503, headers: { "Content-Type": "application/json" } })));
    renderApp("/history");
    expect(await screen.findByRole("button", { name: /API not ready/ })).toBeInTheDocument();
    expect(screen.getByText(/MQVG_API_KEY/)).toBeInTheDocument();
  });
});

describe("Sidebar", () => {
  it("has no 'Recommendation only' note (the decision banners still say nothing was deployed)", async () => {
    installApi({ key: KEY });
    renderApp("/");
    await screen.findByRole("heading", { level: 1, name: "Dashboard" });
    expect(screen.queryByText(/Recommendation only/)).not.toBeInTheDocument();
    expect(screen.queryByText(/never deploys/)).not.toBeInTheDocument();
  });
});
