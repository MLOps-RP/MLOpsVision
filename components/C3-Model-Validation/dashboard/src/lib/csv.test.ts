import { describe, expect, it } from "vitest";

import { csvCell, toCsv } from "./csv";

describe("csvCell", () => {
  it("leaves plain values alone and quotes what needs quoting", () => {
    expect(csvCell("good")).toBe("good");
    expect(csvCell(86.9)).toBe("86.9");
    expect(csvCell(null)).toBe("");
    expect(csvCell(undefined)).toBe("");
    expect(csvCell("a,b")).toBe('"a,b"');
    expect(csvCell('say "hi"')).toBe('"say ""hi"""');
    expect(csvCell("two\nlines")).toBe('"two\nlines"');
  });

  it.each(["=SUM(A1:A9)", "+1+1", "-2+3", "@cmd", "\tx", "=HYPERLINK(\"http://evil\")"])(
    "neutralises text a spreadsheet would run as a formula: %s", (text) => {
      const cell = csvCell(text);
      expect(cell.replace(/^"/, "").startsWith("'")).toBe(true);
    });

  it("does not alter a negative number (only text is neutralised)", () => {
    expect(csvCell(-3.5)).toBe("-3.5");
  });
});

describe("toCsv", () => {
  it("writes a header and CRLF-separated rows", () => {
    expect(toCsv(["a", "b"], [["x", 1], ["y", null]])).toBe("a,b\r\nx,1\r\ny,\r\n");
  });
});
