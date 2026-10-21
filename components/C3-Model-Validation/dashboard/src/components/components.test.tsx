import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import type { CheckContribution, CheckResult } from "../api/types";
import { CHECKS, RESULTS } from "../test/mockApi";
import { DecisionBadge, StatusBadge } from "./badges";
import { ChecksTable } from "./ChecksTable";
import { DecisionBanner } from "./DecisionBanner";
import { HIDDEN_CHECK, HiddenGroups } from "./HiddenGroups";
import { LayerCard } from "./LayerCard";
import { EvidenceGaps } from "./RulesPanel";
import { ScoreGauge } from "./ScoreGauge";

const catalogue = new Map(CHECKS.map((c) => [c.id, c]));
const contributionsOf = (name: keyof typeof RESULTS) => {
  const map = new Map<string, CheckContribution>();
  for (const dim of Object.values(RESULTS[name].scorecard.dimensions)) for (const c of dim.contributions ?? []) map.set(c.check_id, c);
  return map;
};
const table = (name: keyof typeof RESULTS) => (
  <ChecksTable checks={RESULTS[name].checks} contributions={contributionsOf(name)} catalogue={catalogue} />
);
const bodyRows = () => within(screen.getByRole("table")).getAllByRole("row").slice(1).filter((r) => !r.classList.contains("detail-row"));

describe("ScoreGauge", () => {
  it("shows the score with an accessible description", () => {
    render(<ScoreGauge value={86.9} promoteAt={75} rejectBelow={50} />);
    expect(screen.getByRole("img", { name: "Model Quality Score 86.9 out of 100" })).toBeInTheDocument();
    expect(screen.getByText("86.9")).toBeInTheDocument();
    expect(screen.getByText("75")).toBeInTheDocument();     // the profile's promotion threshold is marked
    expect(screen.getByText("50")).toBeInTheDocument();
  });

  it("draws an arc of at most 180 degrees (never an SVG large-arc)", () => {
    const { container } = render(<ScoreGauge value={86.9} />);
    const paths = [...container.querySelectorAll("path")].map((p) => p.getAttribute("d")!);
    expect(paths).toHaveLength(2);                           // track + value
    for (const d of paths) expect(d).toMatch(/A 92 92 0 0 1 /);
  });

  it("shows n/a, never a number, when the score cannot be quoted", () => {
    render(<ScoreGauge value={null} lowerBound={24.0} provisional />);
    expect(screen.getByRole("img", { name: "Model Quality Score not available" })).toBeInTheDocument();
    expect(screen.getByText("n/a")).toBeInTheDocument();
    expect(screen.getByText(/not enough evidence/i)).toBeInTheDocument();
    expect(screen.getByText("24.0")).toBeInTheDocument();    // the worst case is what to read instead
  });

  it("flags a provisional score", () => {
    render(<ScoreGauge value={70} provisional />);
    expect(screen.getByText(/provisional/i)).toBeInTheDocument();
  });
});

describe("LayerCard", () => {
  const c = RESULTS.corrupt.scorecard;
  it("shows the nominal weight and says when a layer was not counted", () => {
    render(<LayerCard dimension="model_behaviour" score={c.dimensions.model_behaviour} weight={c.weights.model_behaviour}
      effectiveWeight={c.effective_weights?.model_behaviour} />);
    const card = screen.getByRole("region", { name: /Model Behaviour score/ });
    expect(within(card).getByText("n/a")).toBeInTheDocument();
    expect(card).toHaveTextContent("weight 50%");
    expect(card).toHaveTextContent("counted as 0%");
    expect(card).toHaveTextContent("0 of 8 checks with evidence");
  });

  it("does not mention counting when every layer counts as configured", () => {
    const g = RESULTS.good.scorecard;
    render(<LayerCard dimension="training_validity" score={g.dimensions.training_validity} weight={g.weights.training_validity}
      effectiveWeight={g.effective_weights?.training_validity} />);
    expect(screen.getByRole("region", { name: /Training Validity score/ })).not.toHaveTextContent("counted as");
  });
});

describe("badges and banner", () => {
  it("labels statuses with text, not colour alone", () => {
    render(<><StatusBadge status="UNAVAILABLE" /><StatusBadge status="WARN" /><DecisionBadge decision="ROLLBACK" /></>);
    expect(screen.getByText("No evidence")).toBeInTheDocument();
    expect(screen.getByText("Warning")).toBeInTheDocument();
    expect(screen.getByText("ROLLBACK")).toBeInTheDocument();
  });

  it("states that the decision is only a recommendation and when a critical rule overrode the score", () => {
    render(<DecisionBanner decision="ROLLBACK" summary="why" overriddenByCritical profile="extended" notice="Nothing was deployed." />);
    const banner = screen.getByRole("region", { name: "Decision" });
    expect(banner).toHaveTextContent("ROLLBACK");
    expect(banner).toHaveTextContent("Recommendation");
    expect(banner).toHaveTextContent("Nothing was deployed.");
    expect(banner).toHaveTextContent("A critical rule overrode the score");
    expect(banner).toHaveTextContent("Profile: extended");
  });
});

describe("ChecksTable (real leaky run)", () => {
  it("puts what needs attention first", () => {
    render(table("leaky"));
    const first = bodyRows()[0];
    expect(first).toHaveTextContent("leakage.single_feature_predictiveness");
    expect(first).toHaveTextContent("Fail");
    expect(first).toHaveTextContent("0.968");
    expect(bodyRows()).toHaveLength(25);
  });

  it("filters by status, layer and text", async () => {
    const user = userEvent.setup();
    render(table("leaky"));
    await user.click(screen.getByRole("button", { name: "Fail" }));
    expect(bodyRows()).toHaveLength(1);
    await user.click(screen.getByRole("button", { name: "Needs attention" }));
    expect(bodyRows()).toHaveLength(3);                       // 1 fail + 2 warnings
    await user.click(screen.getByRole("button", { name: "All" }));
    await user.click(screen.getByRole("button", { name: "ORS" }));
    expect(bodyRows().every((r) => r.textContent!.includes("operational."))).toBe(true);
    await user.click(screen.getByRole("button", { name: "All layers" }));
    await user.type(screen.getByRole("searchbox", { name: "Search checks" }), "latency");
    expect(bodyRows()).toHaveLength(1);
    await user.clear(screen.getByRole("searchbox", { name: "Search checks" }));
    await user.type(screen.getByRole("searchbox", { name: "Search checks" }), "no-such-check");
    expect(screen.getByText("No checks match these filters")).toBeInTheDocument();
  });

  it("expands a row to show the message, thresholds and how it entered the score", async () => {
    const user = userEvent.setup();
    render(table("leaky"));
    await user.click(screen.getByRole("button", { name: "Show details for leakage.single_feature_predictiveness" }));
    const detail = screen.getByText(/pass at 0\.9/).closest("tr")!;
    expect(detail).toHaveTextContent("target 0.8 · pass at 0.9 · fail at 0.95 · floor 1");
    expect(detail).toHaveTextContent(/prior_high_income_flag/);
    expect(detail).toHaveTextContent("counted");
    expect(screen.getByRole("button", { name: "Hide details for leakage.single_feature_predictiveness" })).toHaveAttribute("aria-expanded", "true");
  });

  it("shows missing evidence as 'no score', never as a number", () => {
    render(table("corrupt"));
    const row = screen.getByText("behaviour.f1").closest("tr")!;
    expect(row).toHaveTextContent("No evidence");
    expect(row).toHaveTextContent("no score");
  });
});

describe("EvidenceGaps", () => {
  it("lists a reason once when many checks share it", () => {
    render(<EvidenceGaps checks={RESULTS.corrupt.checks} />);
    const gaps = RESULTS.corrupt.checks.filter((c) => c.status === "UNAVAILABLE" || c.status === "ERROR").length;
    expect(gaps).toBeGreaterThan(10);
    const rows = within(screen.getByRole("table")).getAllByRole("row").length - 1;
    expect(rows).toBeLessThan(gaps / 2);
    const summaries = screen.getAllByText(/^\d+ checks$/);
    expect(summaries.length).toBeGreaterThan(0);              // grouped rows say how many checks share the reason
    expect(Math.max(...summaries.map((s) => parseInt(s.textContent!, 10)))).toBeGreaterThanOrEqual(5);
  });

  it("says so when there are none", () => {
    render(<EvidenceGaps checks={RESULTS.good.checks} />);
    expect(screen.getByText(/no gaps/i)).toBeInTheDocument();
  });
});

describe("HiddenGroups (real planted-group run)", () => {
  const check = RESULTS.hidden_group.checks.find((c) => c.id === HIDDEN_CHECK) as CheckResult;

  it("shows each group's statistics and the configured consequence", () => {
    render(<HiddenGroups check={check} policy={{ consequence: "hold", require_assessment: true }} />);
    expect(screen.getByRole("heading", { name: "6 confirmed of 15 groups tested" })).toBeInTheDocument();
    expect(screen.getByText(/segment = d AND tenure_years > 6\.57 AND income_k > 57/)).toBeInTheDocument();
    expect(screen.getAllByText("Confirmed", { selector: "span" })).toHaveLength(6);
    expect(screen.getByText("76.5%")).toBeInTheDocument();    // observed error of the worst group
    expect(screen.getByText("27.0%")).toBeInTheDocument();    // what the model's own probabilities predicted
    expect(screen.getByText(/Configured consequence/)).toHaveTextContent("hold");
    expect(screen.getByText(/proxies for sensitive characteristics/)).toBeInTheDocument();
    expect(screen.getByText(/association found in the evaluation data, not a causal explanation/)).toBeInTheDocument();
  });

  it("says 'not assessed' (not a pass) when the check could not run", () => {
    render(<HiddenGroups check={{ ...check, status: "UNAVAILABLE", score: null, details: {}, message: "only 700 evaluation rows" }} />);
    expect(screen.getByText("Not assessed")).toBeInTheDocument();
    expect(screen.getByText(/only 700 evaluation rows/)).toBeInTheDocument();
    expect(screen.getByText(/This is not a pass/)).toBeInTheDocument();
  });
});
