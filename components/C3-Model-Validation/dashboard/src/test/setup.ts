import "@testing-library/jest-dom/vitest";
import { cleanup, configure } from "@testing-library/react";
import { afterEach, beforeEach, vi } from "vitest";

// Pages are lazy-loaded, so the first visit to a page includes compiling it; give async queries room.
configure({ asyncUtilTimeout: 6000 });

// jsdom has no ResizeObserver (Recharts' container uses one). The charts are given an initial size instead.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = ResizeObserverStub as unknown as typeof ResizeObserver;
// jsdom lays nothing out, so every element measures 0 x 0 and Recharts would draw nothing. Give them a size.
for (const [prop, value] of [["clientWidth", 640], ["clientHeight", 300], ["offsetWidth", 640], ["offsetHeight", 300]] as const) {
  Object.defineProperty(HTMLElement.prototype, prop, { configurable: true, get: () => value });
}
Element.prototype.getBoundingClientRect = () =>
  ({ width: 640, height: 300, top: 0, left: 0, right: 640, bottom: 300, x: 0, y: 0, toJSON: () => ({}) }) as DOMRect;

// jsdom has no Blob URLs and does not implement following a download link: record the blobs instead.
export const downloads: { name: string; blob: Blob }[] = [];

beforeEach(() => {
  downloads.length = 0;
  let last: Blob | null = null;
  URL.createObjectURL = vi.fn((blob: Blob) => { last = blob; return "blob:test"; });
  URL.revokeObjectURL = vi.fn();
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) {
    if (last) downloads.push({ name: this.download, blob: last });
  });
});

afterEach(() => {
  cleanup();
  sessionStorage.clear();
  vi.restoreAllMocks();
});
