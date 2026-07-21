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
  return risk.evidence_type === "published_constraint_conflict"
    ? "Published constraint"
    : "Model prediction";
}

export function shortConversationId(id: string): string {
  return id.slice(0, 8).toUpperCase();
}
