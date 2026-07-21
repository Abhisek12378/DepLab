import { AlertTriangle, ArrowRight, CheckCircle2, ChevronRight, Database, ShieldCheck } from "lucide-react";
import { categoryLabels, evidenceLabel, riskPercent, riskTone } from "../lib/presentation";
import type { AdvisoryResult, Alternative, PairRisk } from "../types";

export function ResultDetails({ result }: { result: AdvisoryResult }) {
  return (
    <div className="result-details">
      <EvidenceBanner result={result} />
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
          <ul>{result.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>
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

function RiskCard({ risk }: { risk: PairRisk }) {
  const deterministic = risk.evidence_type === "published_constraint_conflict";
  const conflict = risk.constraint_conflicts[0];
  return (
    <article className={`risk-card ${deterministic ? "constraint-card" : "prediction-card"}`}>
      <div className="risk-card-top">
        <span className={`evidence-pill ${deterministic ? "fact" : "prediction"}`}>
          {deterministic ? <ShieldCheck size={13} /> : <Database size={13} />}{evidenceLabel(risk)}
        </span>
        <span className={`risk-score ${riskTone(risk.risk_score)}`}>{riskPercent(risk.risk_score)} risk</span>
      </div>
      <h4>{risk.target_package} <span>×</span> {risk.related_package}</h4>
      <p>{deterministic && conflict ? `${conflict.declaring_package} ${conflict.declaring_version} declares ${conflict.dependency_package}${conflict.blocking_specifiers.join(", ") || conflict.requirement}.` : `DepLab warns about a possible ${risk.likely_stage} failure.`}</p>
    </article>
  );
}

function AlternativeCard({ item, index }: { item: Alternative; index: number }) {
  const recommended = item.category === "achieves_requested_change";
  return (
    <article className={`alternative-card ${recommended ? "recommended" : ""}`}>
      <div className="alternative-rank">{recommended ? <CheckCircle2 size={18} /> : index + 1}</div>
      <div className="alternative-body">
        <div className="alternative-heading"><strong>{categoryLabels[item.category]}</strong>{recommended && <span>Best match</span>}</div>
        <div className="change-list">
          {item.changes.length === 0 ? <span className="no-change">No package changes</span> : item.changes.map((change) => (
            <span className="change-chip" key={change.package}>{change.package} <small>{change.from_version}</small><ArrowRight size={12} /><b>{change.to_version}</b></span>
          ))}
        </div>
      </div>
      <div className="alternative-risk"><span>{riskPercent(item.maximum_risk_score)}</span><small>max risk</small></div>
    </article>
  );
}
