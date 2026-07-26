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
  evidence_type:
    | "published_constraint_conflict"
    | "model_prediction"
    | "structured_ranking_signal"
    | "post_install_prediction";
  constraint_conflicts: ConstraintConflict[];
  import_probability?: number | null;
  smoke_probability?: number | null;
  model_coverage?: "covered" | "not_scored";
}

export interface ResolverCheck {
  environment: Record<string, string>;
  status:
    | "resolved"
    | "unresolvable"
    | "direct_constraint_blocked"
    | "timeout"
    | "unavailable";
  resolvable: boolean | null;
  duration_seconds: number;
  cache_hit: boolean;
  explanation: string;
  evidence_type: "uv_resolution";
  installed: false;
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
  resolver_status?: ResolverCheck["status"] | "not_checked";
  resolver_duration_seconds?: number | null;
  post_install_risk?: number | null;
  verification_status?: "resolver_verified" | "not_checked";
}

export interface AdvisoryResult {
  status:
    | "risk_found"
    | "no_risk_predicted"
    | "coverage_unavailable"
    | "resolver_unavailable"
    | "input_error";
  summary: string;
  pair_risks: PairRisk[];
  resolver_checks?: ResolverCheck[];
  alternatives: Alternative[];
  warnings: string[];
  errors: string[];
  answer: string;
  model_scope:
    | "prediction_only"
    | "deterministic_constraints_and_prediction"
    | "resolver_and_post_install_prediction";
  verification_status:
    | "not_runtime_verified"
    | "resolver_verified"
    | "resolver_rejected"
    | "resolver_unavailable";
  candidates_considered?: number;
  candidates_resolver_checked?: number;
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
