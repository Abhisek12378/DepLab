export type MessageRole = "user" | "assistant";

export interface ConstraintConflict {
  evidence_type: "published_constraint_conflict";
  certainty: "deterministic";
  declaring_package: string;
  declaring_version: string;
  dependency_package: string;
  dependency_version: string;
  requirement: string;
  blocking_specifiers: string[];
}

export interface PairRisk {
  family: string;
  target_package: string;
  target_version: string;
  related_package: string;
  related_version: string;
  risk_score: number;
  predicted_failure: boolean;
  likely_stage: string;
  published_constraints_allow: boolean;
  risk_detected: boolean;
  evidence_type: "published_constraint_conflict" | "model_prediction";
  constraint_conflicts: ConstraintConflict[];
}

export interface SuggestedChange {
  package: string;
  from_version: string;
  to_version: string;
}

export type AlternativeCategory =
  | "achieves_requested_change"
  | "keeps_current_version"
  | "different_upgrade_fallback"
  | "downgrade_fallback";

export interface Alternative {
  target_version: string;
  keeps_requested_target: boolean;
  category: AlternativeCategory;
  changes: SuggestedChange[];
  maximum_risk_score: number;
  predicted_failure: boolean;
  covered_pairs: number;
  reason: string;
}

export interface AdvisoryResult {
  status: "risk_found" | "no_risk_predicted" | "coverage_unavailable" | "input_error";
  summary: string;
  pair_risks: PairRisk[];
  alternatives: Alternative[];
  warnings: string[];
  errors: string[];
  answer: string;
  model_scope: "prediction_only" | "deterministic_constraints_and_prediction";
  verification_status: "not_runtime_verified";
}

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  created_at: string;
  result: AdvisoryResult | null;
}

export interface Conversation {
  id: string;
  requirements_text: string;
  python_version: "3.10" | "3.11" | "3.12";
  platform: "linux-x86_64";
  created_at: string;
  expires_at: string;
  messages: ChatMessage[];
}

export interface Exchange {
  conversation: Conversation;
  user_message: ChatMessage;
  assistant_message: ChatMessage;
}

export interface CreateConversationInput {
  requirements_text: string;
  python_version: Conversation["python_version"];
  platform: Conversation["platform"];
}
