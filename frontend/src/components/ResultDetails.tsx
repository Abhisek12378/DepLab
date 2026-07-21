import { ArrowRight, CheckCircle2, ChevronRight, Database, ShieldCheck } from "lucide-react";
import { categoryLabels, evidenceLabel, formatCoverageWarning, riskPercent, riskTone } from "../lib/presentation";
import type { AdvisoryResult, Alternative, PairRisk } from "../types";

export function ResultDetails({ result }: { result: AdvisoryResult }) {
  return (
    <div className="result-details">
      <EvidenceBanner result={result} />
      <EvidenceLegend />
      {result.pair_risks.length > 0 && (
        <details className="result-section" open>
          <summary><span>Compatibility evidence</span><ChevronRight size={16} /></summary>
          <div className="risk-grid">{result.pair_risks.map((risk) => <RiskCard key={`${risk.family}-${risk.related_version}`} risk={risk} />)}</div>
        </details>
      )}
      {result.alternatives.length > 0 && (
        <details className="result-section" open>
          <summary><span>Recommended environments</span><ChevronRight size={16} /></summary>
          <div className="alternative-list">{result.alternatives.map((item, index) => <AlternativeCard key={`${item.target_version}-${item.category}`} item={item} index={index} />)}</div>
        </details>
      )}
      {result.warnings.length > 0 && (
        <details className="result-section coverage-section">
          <summary><span>Coverage notes · {result.warnings.length}</span><ChevronRight size={16} /></summary>
          <ul>{result.warnings.map((warning) => <li key={warning}>{formatCoverageWarning(warning)}</li>)}</ul>
        </details>
      )}
    </div>
  );
}

function EvidenceBanner({ result }: { result: AdvisoryResult }) {
  const deterministic = result.model_scope === "deterministic_constraints_and_prediction";
  return (
    <div className={`evidence-banner ${deterministic ? "deterministic" : "predictive"}`}>
      {deterministic ? <ShieldCheck size={17} /> : <Database size={17} />}
      <span>{deterministic ? "Published constraints establish part of this answer. Additional scores are predictions." : "This answer is based on model predictions; published constraints do not block the scored pairs."}</span>
    </div>
  );
}

function EvidenceLegend() {
  return (
    <div className="evidence-legend" aria-label="Evidence legend">
      <span className="legend-fact"><i />BLOCKED = published constraint (fact)</span>
      <span className="legend-prediction"><i />% = DepLab model prediction</span>
    </div>
  );
}

function RiskCard({ risk }: { risk: PairRisk }) {
  const deterministic = risk.evidence_type === "published_constraint_conflict";
  const conflict = risk.constraint_conflicts[0];
  return (
    <article className={`risk-card ${deterministic ? "constraint-card" : "prediction-card"}`}>
      <div className="risk-card-top">
        <span className={`evidence-pill ${deterministic ? "fact" : "prediction"}`}>
          {deterministic ? <ShieldCheck size={13} /> : <Database size={13} />}{evidenceLabel(risk)}
        </span>
        {deterministic ? <span className="blocked-label">BLOCKED</span> : <span className={`risk-score ${riskTone(risk.risk_score)}`}>{riskPercent(risk.risk_score)} risk</span>}
      </div>
      <h4>{risk.target_package} <span>×</span> {risk.related_package}</h4>
      {deterministic && conflict ? (
        <p>{conflict.declaring_package} {conflict.declaring_version} declares <code>{conflict.requirement}</code><span className="inline-blocker">blocks <code>{conflict.blocking_specifiers.join(", ") || conflict.requirement}</code></span></p>
      ) : <p>DepLab warns about a possible {risk.likely_stage} failure.</p>}
    </article>
  );
}

function AlternativeCard({ item, index }: { item: Alternative; index: number }) {
  const tone = alternativeTone(item);
  const recommended = tone === "goal";
  return (
    <article className={`alternative-card ${tone}`}>
      <div className="alternative-rank">{recommended ? <CheckCircle2 size={18} /> : index + 1}</div>
      <div className="alternative-body">
        <div className="alternative-heading"><strong>{categoryLabels[item.category]}</strong><span>{alternativeTag(tone)}</span></div>
        <div className="change-list">
          {item.changes.length === 0 ? <span className="no-change">No package changes</span> : item.changes.map((change) => (
            <span className="change-chip" key={change.package}><span>{change.package}</span><small>{change.from_version}</small><ArrowRight size={13} /><b>{change.to_version}</b></span>
          ))}
        </div>
      </div>
      <div className="alternative-risk"><span>{riskPercent(item.maximum_risk_score)}</span><small>max predicted risk</small></div>
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
