// Types come from the API's own OpenAPI contract (docs/api_openapi.json) via `npm run types`.
import type { components } from "./schema";

type S = components["schemas"];

export type ValidationSummary = S["ValidationSummary"];
export type ValidationResult = S["ValidationResult"];
export type ValidationRequest = S["ValidationRequest"];
export type CheckResult = S["CheckResult"];
export type CheckInfo = S["CheckInfo"];
export type ScoreCard = S["ScoreCard"];
export type DimensionScore = S["DimensionScore"];
export type CheckContribution = S["CheckContribution"];
export type TriggeredRule = S["TriggeredRule"];
export type ProfileInfo = S["ProfileInfo"];
export type ProfileList = S["ProfileList"];
export type RunEntry = S["RunEntry"];
export type RunList = S["RunList"];
export type Health = S["Health"];
export type CandidateInfo = S["CandidateInfo"];
export type CandidateList = S["CandidateList"];
export type StudyInfo = S["StudyInfo"];
export type StudyList = S["StudyList"];
export type StudyData = S["StudyData"];

export type Decision = S["Decision"];
export type CheckStatus = S["CheckStatus"];
export type Dimension = S["Dimension"];
export type EvidenceSource = S["EvidenceSource"];
export type RuleKind = S["RuleKind"];

export const DIMENSIONS: Dimension[] = ["training_validity", "model_behaviour", "operational_readiness"];
export const DECISIONS: Decision[] = ["PROMOTE", "HOLD", "ROLLBACK"];
export const STATUSES: CheckStatus[] = ["PASS", "WARN", "FAIL", "UNAVAILABLE", "ERROR", "NOT_APPLICABLE"];
export const NO_EVIDENCE: CheckStatus[] = ["UNAVAILABLE", "ERROR"];
