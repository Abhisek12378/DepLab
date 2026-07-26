import {
  ArrowRight,
  CheckCircle2,
  ChevronRight,
  CircleX,
  Database,
  PackageCheck,
  ShieldCheck,
} from "lucide-react";
import {
  categoryLabels,
  evidenceLabel,
  formatCoverageWarning,
  riskPercent,
  riskTone,
} from "../lib/presentation";
import type {
  AdvisoryResult,
  Alternative,
  PairRisk,
  ResolverCheck,
} from "../types";

export function ResultDetails({ result }: { result: AdvisoryResult }) {
  const proposedResolver = result.resolver_checks?.[0];
  return (
    <div className="result-details">
      <EvidenceBanner result={result} />
      <EvidenceLegend />
      {proposedResolver && <ResolverCard check={proposedResolver} />}
      {result.pair_risks.length > 0 && (
        <details className="result-section" open>
          <summary>
            <span>Compatibility evidence</span>
            <ChevronRight size={16} />
          </summary>
          <div className="risk-grid">
            {result.pair_risks.map((risk) => (
              <RiskCard
                key={`${risk.family}-${risk.related_version}`}
                risk={risk}
              />
            ))}
          </div>
        </details>
      )}
      {result.alternatives.length > 0 && (
        <details className="result-section" open>
          <summary>
            <span>Recommended environments</span>
            <ChevronRight size={16} />
          </summary>
          <div className="alternative-list">
            {result.alternatives.map((item, index) => (
              <AlternativeCard
                key={`${item.target_version}-${item.category}`}
                item={item}
                index={index}
              />
            ))}
          </div>
        </details>
      )}
      {result.warnings.length > 0 && (
        <details className="result-section coverage-section">
          <summary>
            <span>Coverage notes · {result.warnings.length}</span>
            <ChevronRight size={16} />
          </summary>
          <ul>
            {result.warnings.map((warning) => (
              <li key={warning}>{formatCoverageWarning(warning)}</li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

function EvidenceBanner({ result }: { result: AdvisoryResult }) {
  if (result.model_scope === "resolver_and_post_install_prediction") {
    return (
      <div className="evidence-banner resolver">
        <PackageCheck size={17} />
        <span>
          uv checks whether the pins resolve. DepLab separately predicts import
          and smoke-test risk. No packages are installed.
        </span>
      </div>
    );
  }
  const deterministic = result.pair_risks.some(
    (risk) => risk.evidence_type === "published_constraint_conflict",
  );
  return (
    <div
      className={`evidence-banner ${
        deterministic ? "deterministic" : "predictive"
      }`}
    >
      {deterministic ? <ShieldCheck size={17} /> : <Database size={17} />}
      <span>
        {deterministic
          ? "Published constraints establish part of this answer. Additional scores are predictions."
          : "This answer is based on model predictions; published constraints do not block the scored pairs."}
      </span>
    </div>
  );
}

function EvidenceLegend() {
  return (
    <div className="evidence-legend" aria-label="Evidence legend">
      <span className="legend-fact">
        <i />
        BLOCKED = published constraint (fact)
      </span>
      <span className="legend-resolver">
        <i />
        RESOLVED = uv dependency check
      </span>
      <span className="legend-prediction">
        <i />% = post-install model prediction
      </span>
    </div>
  );
}

function ResolverCard({ check }: { check: ResolverCheck }) {
  const presentation = resolverPresentation(check);
  const Icon = presentation.icon;
  return (
    <article className={`resolver-card ${presentation.tone}`}>
      <div>
        <Icon size={17} />
        <strong>uv dependency resolution</strong>
        <span>{presentation.label}</span>
      </div>
      <p>{check.explanation}</p>
      <small>
        {check.cache_hit ? "Cached result" : `${check.duration_seconds.toFixed(2)}s`}
        {" · "}Resolution only · nothing installed
      </small>
    </article>
  );
}

function resolverPresentation(check: ResolverCheck) {
  if (check.resolvable === true) {
    return { tone: "resolved", label: "RESOLVED", icon: CheckCircle2 };
  }
  if (check.resolvable === false) {
    return { tone: "blocked", label: "WILL NOT RESOLVE", icon: CircleX };
  }
  return { tone: "unavailable", label: "CHECK UNAVAILABLE", icon: Database };
}

function RiskCard({ risk }: { risk: PairRisk }) {
  const deterministic = risk.evidence_type === "published_constraint_conflict";
  const rankingOnly = risk.evidence_type === "structured_ranking_signal";
  const conflict = risk.constraint_conflicts[0];
  return (
    <article
      className={`risk-card ${
        deterministic
          ? "constraint-card"
          : rankingOnly
            ? "ranking-card"
            : "prediction-card"
      }`}
    >
      <div className="risk-card-top">
        <span
          className={`evidence-pill ${
            deterministic ? "fact" : "prediction"
          }`}
        >
          {deterministic ? (
            <ShieldCheck size={13} />
          ) : (
            <Database size={13} />
          )}
          {evidenceLabel(risk)}
        </span>
        <RiskBadge
          deterministic={deterministic}
          rankingOnly={rankingOnly}
          score={risk.risk_score}
        />
      </div>
      <h4>
        {risk.target_package} <span>×</span> {risk.related_package}
      </h4>
      <RiskExplanation
        risk={risk}
        deterministic={deterministic}
        rankingOnly={rankingOnly}
        conflict={conflict}
      />
    </article>
  );
}

function RiskBadge({
  deterministic,
  rankingOnly,
  score,
}: {
  deterministic: boolean;
  rankingOnly: boolean;
  score: number;
}) {
  if (deterministic) return <span className="blocked-label">BLOCKED</span>;
  if (rankingOnly) return <span className="ranking-label">RANKING ONLY</span>;
  return (
    <span className={`risk-score ${riskTone(score)}`}>
      {riskPercent(score)} risk
    </span>
  );
}

function RiskExplanation({
  risk,
  deterministic,
  rankingOnly,
  conflict,
}: {
  risk: PairRisk;
  deterministic: boolean;
  rankingOnly: boolean;
  conflict: PairRisk["constraint_conflicts"][number] | undefined;
}) {
  if (deterministic && conflict) {
    return (
      <p>
        {conflict.declaring_package} {conflict.declaring_version} declares{" "}
        <code>{conflict.requirement}</code>
        <span className="inline-blocker">
          blocks{" "}
          <code>
            {conflict.blocking_specifiers.join(", ") || conflict.requirement}
          </code>
        </span>
      </p>
    );
  }
  if (rankingOnly) {
    return (
      <p>
        This structured score only prioritized resolver checks. It is not proof
        of failure.
      </p>
    );
  }
  return (
    <p>
      After uv resolved the pins, DepLab predicted a possible {risk.likely_stage}{" "}
      failure.
    </p>
  );
}

function AlternativeCard({
  item,
  index,
}: {
  item: Alternative;
  index: number;
}) {
  const tone = alternativeTone(item);
  const recommended = tone === "goal";
  return (
    <article className={`alternative-card ${tone}`}>
      <div className="alternative-rank">
        {recommended ? <CheckCircle2 size={18} /> : index + 1}
      </div>
      <div className="alternative-body">
        <div className="alternative-heading">
          <strong>{categoryLabels[item.category]}</strong>
          <span>{alternativeTag(tone)}</span>
        </div>
        <div className="change-list">
          {item.changes.length === 0 ? (
            <span className="no-change">No package changes</span>
          ) : (
            item.changes.map((change) => (
              <span className="change-chip" key={change.package}>
                <span>{change.package}</span>
                <small>{change.from_version}</small>
                <ArrowRight size={13} />
                <b>{change.to_version}</b>
              </span>
            ))
          )}
        </div>
      </div>
      <div className="alternative-risk">
        <span>{riskPercent(item.maximum_risk_score)}</span>
        <small>max post-install risk</small>
      </div>
    </article>
  );
}

function alternativeTone(item: Alternative): "goal" | "current" | "fallback" {
  if (item.category === "achieves_requested_change") return "goal";
  if (item.category === "keeps_current_version") return "current";
  return "fallback";
}

function alternativeTag(tone: "goal" | "current" | "fallback"): string {
  if (tone === "goal") return "Best match";
  if (tone === "current") return "No change";
  return "Fallback";
}
