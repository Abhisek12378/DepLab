import { ArrowRight, Ban, CheckCircle2, MinusCircle, TriangleAlert } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { categoryLabels, formatCoverageWarning } from "../lib/presentation";
import type { AdvisoryResult, Alternative, ConstraintConflict } from "../types";

interface StructuredAnswerProps {
  result: AdvisoryResult;
  fallback: string;
}

export function StructuredAnswer({ result, fallback }: StructuredAnswerProps) {
  const conflicts = result.pair_risks.flatMap((risk) => risk.constraint_conflicts);
  const hasStructuredContent = conflicts.length > 0 || result.alternatives.length > 0;

  if (!hasStructuredContent) return <ReactMarkdown>{fallback}</ReactMarkdown>;

  return (
    <div className="structured-answer">
      {conflicts.length > 0 ? <ConstraintSummary conflicts={conflicts} /> : <p>{result.summary}</p>}
      {result.alternatives.length > 0 && <RecommendationSummary alternatives={result.alternatives} />}
      {result.warnings.map((warning) => (
        <p className="answer-coverage-note" key={warning}>{formatCoverageWarning(warning)}</p>
      ))}
      <p className="answer-verification">Prediction and published evidence only. No environment was installed.</p>
    </div>
  );
}

function ConstraintSummary({ conflicts }: { conflicts: ConstraintConflict[] }) {
  return (
    <section className="answer-conflicts" aria-label="Published constraint conflicts">
      <div className="answer-section-title"><Ban size={15} /><strong>{conflicts.length} published-constraint conflicts</strong><span>Facts</span></div>
      <div className="answer-conflict-list">
        {conflicts.map((conflict) => (
          <article className="answer-conflict" key={conflictKey(conflict)}>
            <strong>{conflict.declaring_package} {conflict.declaring_version} blocks {conflict.dependency_package} {conflict.dependency_version}</strong>
            <p>Published requirement <code>{conflict.requirement}</code></p>
            <div className="blocking-specifier"><span>Blocking specifier</span><code>{blockingSpecifier(conflict)}</code></div>
          </article>
        ))}
      </div>
    </section>
  );
}

function RecommendationSummary({ alternatives }: { alternatives: Alternative[] }) {
  return (
    <section className="answer-recommendations" aria-label="Recommended changes">
      {alternatives.map((alternative) => (
        <RecommendationBlock alternative={alternative} key={`${alternative.category}-${alternative.target_version}`} />
      ))}
    </section>
  );
}

function RecommendationBlock({ alternative }: { alternative: Alternative }) {
  const presentation = recommendationPresentation(alternative);
  const Icon = presentation.icon;
  return (
    <article className={`answer-recommendation ${presentation.tone}`}>
      <div className="recommendation-header">
        <strong>{categoryLabels[alternative.category]}</strong>
        <span><Icon size={12} />{presentation.tag}</span>
      </div>
      <div className="recommendation-changes">
        {alternative.changes.length === 0 ? (
          <p>Keep the current pinned environment.</p>
        ) : alternative.changes.map((change) => (
          <div className="recommendation-change" key={change.package}>
            <span>{change.package}</span>
            <del>{change.from_version}</del>
            <ArrowRight size={13} />
            <strong>{change.to_version}</strong>
          </div>
        ))}
      </div>
    </article>
  );
}

function recommendationPresentation(alternative: Alternative) {
  if (alternative.category === "achieves_requested_change") {
    return { tone: "goal", tag: "Achieves your goal", icon: CheckCircle2 };
  }
  if (alternative.category === "keeps_current_version") {
    return { tone: "current", tag: "No change", icon: MinusCircle };
  }
  return { tone: "fallback", tag: "Fallback · does not achieve goal", icon: TriangleAlert };
}

function blockingSpecifier(conflict: ConstraintConflict): string {
  return conflict.blocking_specifiers.join(", ") || conflict.requirement;
}

function conflictKey(conflict: ConstraintConflict): string {
  return `${conflict.declaring_package}-${conflict.declaring_version}-${conflict.dependency_package}-${conflict.dependency_version}`;
}
