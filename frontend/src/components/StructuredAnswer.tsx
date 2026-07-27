import {
  ArrowRight,
  Ban,
  CheckCircle2,
  ChevronRight,
  CircleX,
  PackageCheck,
  ShieldCheck,
  Sparkles,
  TriangleAlert,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import {
  categoryLabels,
  formatCoverageWarning,
  riskPercent,
} from "../lib/presentation";
import type {
  AdvisoryResult,
  Alternative,
  ConstraintConflict,
} from "../types";

interface StructuredAnswerProps {
  result: AdvisoryResult;
  fallback: string;
}

export function StructuredAnswer({ result, fallback }: StructuredAnswerProps) {
  const conflicts = result.pair_risks.flatMap(
    (risk) => risk.constraint_conflicts,
  );
  const hasStructuredContent =
    Boolean(result.resolver_checks?.[0]) ||
    conflicts.length > 0 ||
    result.alternatives.length > 0;

  if (!hasStructuredContent) return <ReactMarkdown>{fallback}</ReactMarkdown>;

  const [primary, ...additional] = result.alternatives;
  return (
    <div className="structured-answer">
      <AnswerVerdict
        primary={primary}
        conflictCount={conflicts.length}
        alternatives={result.alternatives}
      />
      {primary && <PrimaryRecommendation alternative={primary} />}
      {additional.length > 0 && (
        <AdditionalRecommendations alternatives={additional} />
      )}
      {conflicts.length > 0 && <ConstraintSummary conflicts={conflicts} />}
      {result.warnings.map((warning) => (
        <p className="answer-coverage-note" key={warning}>
          {formatCoverageWarning(warning)}
        </p>
      ))}
      <VerificationNote result={result} />
    </div>
  );
}

function AnswerVerdict({
  primary,
  conflictCount,
  alternatives,
}: {
  primary: Alternative | undefined;
  conflictCount: number;
  alternatives: Alternative[];
}) {
  const goalCount = alternatives.filter(
    (item) => item.category === "achieves_requested_change",
  ).length;
  const presentation = verdictPresentation(primary, conflictCount, goalCount);
  const Icon = presentation.icon;
  return (
    <section className={`answer-verdict ${presentation.tone}`}>
      <span className="verdict-icon">
        <Icon size={21} />
      </span>
      <div>
        <small>{presentation.eyebrow}</small>
        <h3>{presentation.title}</h3>
        <p>{presentation.description}</p>
      </div>
      <span className="verdict-status">{presentation.status}</span>
    </section>
  );
}

function verdictPresentation(
  primary: Alternative | undefined,
  conflictCount: number,
  goalCount: number,
) {
  if (primary?.keeps_requested_target) {
    return {
      tone: "success",
      eyebrow: "Recommended next step",
      title: "A verified upgrade path is available",
      description:
        `The current pins have ${conflictCount} blocking conflict(s), but ` +
        `DepLab found ${goalCount} complete environment(s) that achieve your change.`,
      status: "READY",
      icon: Sparkles,
    };
  }
  if (primary) {
    return {
      tone: "caution",
      eyebrow: "Safer alternative found",
      title: "The exact request is blocked",
      description:
        "DepLab could not keep the requested target, but it found a resolver-verified fallback.",
      status: "FALLBACK",
      icon: TriangleAlert,
    };
  }
  if (conflictCount > 0) {
    return {
      tone: "blocked",
      eyebrow: "Action required",
      title: "The current environment will not resolve",
      description:
        "Published requirements block this change, and no verified alternative is currently available.",
      status: "BLOCKED",
      icon: CircleX,
    };
  }
  return {
    tone: "neutral",
    eyebrow: "Analysis complete",
    title: "Review the compatibility evidence",
    description:
      "DepLab completed the available deterministic and predictive checks.",
    status: "REVIEW",
    icon: ShieldCheck,
  };
}

function PrimaryRecommendation({
  alternative,
}: {
  alternative: Alternative;
}) {
  return (
    <section
      className="primary-recommendation"
      aria-label="Best verified environment"
    >
      <div className="primary-recommendation-header">
        <span className="recommendation-medallion">
          <CheckCircle2 size={20} />
        </span>
        <div>
          <small>Best verified environment</small>
          <h4>{categoryLabels[alternative.category]}</h4>
        </div>
        <span className="verified-badge">
          <PackageCheck size={13} />
          uv verified
        </span>
      </div>
      <EnvironmentChanges alternative={alternative} prominent />
      <div className="recommendation-proof">
        <span>
          <CheckCircle2 size={14} />
          Complete dependency set resolves
        </span>
        <span>
          <ShieldCheck size={14} />
          {riskPercent(alternative.maximum_risk_score)} maximum predicted
          post-install risk
        </span>
        <small>No packages were installed</small>
      </div>
    </section>
  );
}

function AdditionalRecommendations({
  alternatives,
}: {
  alternatives: Alternative[];
}) {
  return (
    <details className="answer-options">
      <summary>
        <span>
          <CheckCircle2 size={15} />
          {alternatives.length} other verified{" "}
          {alternatives.length === 1 ? "option" : "options"}
        </span>
        <ChevronRight size={16} />
      </summary>
      <div className="answer-options-list">
        {alternatives.map((alternative, index) => (
          <article
            className={`secondary-recommendation ${recommendationTone(
              alternative,
            )}`}
            key={`${alternative.category}-${alternative.target_version}-${index}`}
          >
            <div>
              <strong>Option {index + 2}</strong>
              <span>{categoryLabels[alternative.category]}</span>
            </div>
            <EnvironmentChanges alternative={alternative} />
          </article>
        ))}
      </div>
    </details>
  );
}

function EnvironmentChanges({
  alternative,
  prominent = false,
}: {
  alternative: Alternative;
  prominent?: boolean;
}) {
  if (alternative.changes.length === 0) {
    return <p className="keep-environment">Keep the current pinned environment.</p>;
  }
  return (
    <div className={prominent ? "primary-change-grid" : "compact-change-list"}>
      {alternative.changes.map((change) => (
        <div
          className={prominent ? "primary-change" : "compact-change"}
          key={change.package}
        >
          <span>{change.package}</span>
          <div>
            <del>{change.from_version}</del>
            <ArrowRight size={14} />
            <strong>{change.to_version}</strong>
          </div>
        </div>
      ))}
    </div>
  );
}

function ConstraintSummary({
  conflicts,
}: {
  conflicts: ConstraintConflict[];
}) {
  return (
    <details
      className="answer-conflicts"
      aria-label="Published constraint conflicts"
    >
      <summary>
        <span>
          <Ban size={15} />
          Why the current pins are blocked
        </span>
        <span className="fact-count">{conflicts.length} facts</span>
        <ChevronRight size={16} />
      </summary>
      <div className="answer-conflict-list">
        {conflicts.map((conflict) => (
          <article className="answer-conflict" key={conflictKey(conflict)}>
            <strong>
              {conflict.declaring_package} {conflict.declaring_version} blocks{" "}
              {conflict.dependency_package} {conflict.dependency_version}
            </strong>
            <p>
              Published requirement <code>{conflict.requirement}</code>
            </p>
            <div className="blocking-specifier">
              <span>Blocking specifier</span>
              <code>{blockingSpecifier(conflict)}</code>
            </div>
          </article>
        ))}
      </div>
    </details>
  );
}

function VerificationNote({ result }: { result: AdvisoryResult }) {
  const verifiedAlternative = result.alternatives.some(
    (item) => item.verification_status === "resolver_verified",
  );
  return (
    <p className="answer-verification">
      <ShieldCheck size={13} />
      {verificationCopy(result, verifiedAlternative)}
    </p>
  );
}

function verificationCopy(
  result: AdvisoryResult,
  verifiedAlternative: boolean,
): string {
  if (verifiedAlternative) {
    return "Recommended environments were verified by uv. Post-install findings are model predictions; nothing was installed.";
  }
  if (result.verification_status === "resolver_verified") {
    return "uv verified dependency resolution. Post-install findings are model predictions; nothing was installed.";
  }
  if (result.verification_status === "resolver_rejected") {
    return "The resolution failure is a published-constraint or uv fact. Nothing was installed.";
  }
  if (result.verification_status === "resolver_unavailable") {
    return "The resolver check was unavailable. Nothing was installed.";
  }
  return "Prediction and published evidence only. Nothing was installed.";
}

function recommendationTone(
  alternative: Alternative,
): "goal" | "current" | "fallback" {
  if (alternative.category === "achieves_requested_change") return "goal";
  if (alternative.category === "keeps_current_version") return "current";
  return "fallback";
}

function blockingSpecifier(conflict: ConstraintConflict): string {
  return conflict.blocking_specifiers.join(", ") || conflict.requirement;
}

function conflictKey(conflict: ConstraintConflict): string {
  return `${conflict.declaring_package}-${conflict.declaring_version}-${conflict.dependency_package}-${conflict.dependency_version}`;
}
