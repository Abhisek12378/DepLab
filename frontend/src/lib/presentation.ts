import type { AlternativeCategory, PairRisk } from "../types";

export const categoryLabels: Record<AlternativeCategory, string> = {
  achieves_requested_change: "Achieves your requested change",
  keeps_current_version: "Keeps your current target version",
  different_upgrade_fallback: "Different upgrade fallback",
  downgrade_fallback: "Downgrade fallback — does not achieve your goal",
};

export function riskPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export function riskTone(value: number): "low" | "medium" | "high" {
  if (value >= 0.7) return "high";
  if (value >= 0.3) return "medium";
  return "low";
}

export function evidenceLabel(risk: PairRisk): string {
  if (risk.evidence_type === "published_constraint_conflict") {
    return "Published constraint";
  }
  if (risk.evidence_type === "structured_ranking_signal") {
    return "Ranking signal";
  }
  if (risk.evidence_type === "post_install_prediction") {
    return "Post-install prediction";
  }
  return "Model prediction";
}

export function formatCoverageWarning(warning: string): string {
  const requestsVersion = warning.match(/with requests==([^\s]+) on Python/i)?.[1];
  if (requestsVersion) return `requests ${requestsVersion}: unrelated to this change, not scored.`;
  return warning;
}

export function shortConversationId(id: string): string {
  return id.slice(0, 8).toUpperCase();
}
