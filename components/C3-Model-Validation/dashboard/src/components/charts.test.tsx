import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { parseBaseline, parseSensitivity } from "../api/studies";
import { BASELINE, RESULTS, RUNS, SENSITIVITY } from "../test/mockApi";
import {
  BaselineChart, CheckScoreBars, DecisionDonut, GroupEffectChart, GroupErrorChart, LayerBars, MqsByRunChart, RunCompareChart,
  SensitivityChart, StatusDonut, type GroupPoint,
} from "./charts";
import { readHiddenDetails, HIDDEN_CHECK } from "./HiddenGroups";

/** The rows of a chart's "view data as a table", the accessible copy of what is drawn. */
function tableRows(name: string): string[][] {
  const figure = screen.getByRole("figure", { name });
  return within(figure).getAllByRole("row").slice(1).map((r) => within(r).getAllByRole("cell").map((c) => c.textContent ?? ""));
}

describe("every chart", () => {
  it("is drawn (an SVG) and is also available as a table", () => {
    const { container } = render(<DecisionDonut counts={{ PROMOTE: 2, HOLD: 4, ROLLBACK: 4 }} />);
    expect(container.querySelector("svg.recharts-surface")).not.toBeNull();
    expect(screen.getByRole("img", { name: /Recommendations\. The data is in the table/ })).toBeInTheDocument();
    expect(tableRows("Recommendations")).toEqual([["PROMOTE", "2"], ["HOLD", "4"], ["ROLLBACK", "4"]]);
  });
});

describe("MqsByRunChart", () => {
  it("plots only runs with a quotable score, oldest to newest, and says how many are left out", () => {
    render(<MqsByRunChart runs={RUNS.runs} />);
    const rows = tableRows("Model Quality Score by run");
    const withScore = RUNS.runs.filter((r) => r.mqs !== null);
    expect(rows).toHaveLength(withScore.length);
    expect(rows[0][0]).toBe(withScore[withScore.length - 1].candidate_id);        // the oldest on the left
    expect(rows.every((r) => Number.isFinite(Number(r[1])))).toBe(true);           // never a made-up number
    expect(screen.getByText(/3 runs without a quotable score \(too little evidence\) are not plotted/)).toBeInTheDocument();
  });

  it("does not mention omissions when every run has a score", () => {
    render(<MqsByRunChart runs={RUNS.runs.filter((r) => r.mqs !== null)} />);
    expect(screen.queryByText(/not plotted/)).not.toBeInTheDocument();
  });
});

describe("result charts", () => {
  it("draws layer scores and says that a layer without a score is not drawn as zero", () => {
    render(<LayerBars dimensions={RESULTS.corrupt.scorecard.dimensions} />);
    expect(tableRows("Layer scores")).toEqual([["TVS", "99.9", "80%"], ["MBS", "n/a", "0%"], ["ORS", "n/a", "33%"]]);
    expect(screen.getByText(/not drawn/)).toBeInTheDocument();
  });

  it("shows the lowest-scoring checks first and footnotes the checks that have no score", () => {
    render(<CheckScoreBars checks={RESULTS.corrupt.checks} />);
    const rows = tableRows("Lowest-scoring checks");
    expect(rows.length).toBeLessThanOrEqual(10);
    expect(rows[0][2]).toBe("Fail");                                             // operational.model_loadable
    expect(screen.getByText(/without evidence have no score and are not drawn \(never counted as a pass\)/)).toBeInTheDocument();
  });

  it("counts check statuses", () => {
    render(<StatusDonut counts={{ PASS: 22, WARN: 3 }} />);
    expect(tableRows("Check statuses")).toEqual([["Pass", "22"], ["Warning", "3"]]);
  });
});

describe("hidden failure charts (real planted-group run)", () => {
  const details = readHiddenDetails(RESULTS.hidden_group.checks.find((c) => c.id === HIDDEN_CHECK)!.details)!;
  const groups = details.groups as GroupPoint[];

  it("compares observed with expected error for every group", () => {
    render(<GroupErrorChart groups={groups} />);
    const rows = tableRows("Error rate inside each group");
    expect(rows).toHaveLength(groups.length);
    expect(rows[0]).toEqual(["G1", "76.5%", "27.0%"]);                            // the planted group
  });

  it("shows excess error with its interval and which groups were confirmed", () => {
    render(<GroupEffectChart groups={groups} minEffect={details.min_effect} />);
    const rows = tableRows("Excess error with its 95% interval");
    expect(rows[0]).toEqual(["G1", "49.5%", "[33.0%, 60.6%]", "yes"]);
    expect(rows.filter((r) => r[3] === "yes")).toHaveLength(6);
    expect(rows.filter((r) => r[3] === "no")).toHaveLength(groups.length - 6);
  });
});

describe("comparison charts", () => {
  it("shows an unavailable layer as n/a, not zero", () => {
    render(<RunCompareChart rows={[{ label: "a", tvs: 99.9, mbs: null, ors: null, mqs: null }, { label: "b", tvs: 80, mbs: 70, ors: 90, mqs: 78 }]} />);
    expect(tableRows("Scores side by side")).toEqual([["a", "99.9", "n/a", "n/a", "n/a"], ["b", "80.0", "70.0", "90.0", "78.0"]]);
  });

  it("charts the real baseline study", () => {
    render(<BaselineChart study={parseBaseline(BASELINE)!} />);
    expect(tableRows("Defects stopped and false alarms, by gate")).toEqual([
      ["F1 only", "1", "8", "0", "2"], ["metrics only", "5", "8", "1", "2"],
      ["full gate (default)", "7", "8", "0", "2"], ["full gate (hidden_groups)", "8", "8", "0", "2"]]);
  });

  it("charts the real sensitivity study, one row per model", () => {
    const study = parseSensitivity(SENSITIVITY)!;
    render(<SensitivityChart cases={study.cases} />);
    const rows = tableRows("Share of perturbations that keep the decision");
    expect(rows).toHaveLength(study.cases.length);
    const slow = rows.find((r) => r[0] === "slow")!;
    expect(slow[1]).toBe("HOLD");
    expect(Number.parseFloat(slow[2])).toBeLessThan(100);                         // it flips under some perturbations
    expect(rows.find((r) => r[0] === "overfit")![2]).toMatch(/^100%/);
  });
});
