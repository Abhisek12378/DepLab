from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deplab.advisor import AnalysisRequest, build_default_service  # noqa: E402


st.set_page_config(page_title="DepLab Advisor", page_icon="D", layout="wide")


@st.cache_resource
def service():
    return build_default_service(PROJECT_ROOT)


st.title("DepLab dependency advisor")
st.caption(
    "Ask about a proposed Python package version change. DepLab checks published constraints "
    "and predicts additional compatibility risk. It does not install packages."
)

with st.sidebar:
    st.header("Environment")
    python_version = st.selectbox("Python version", ["3.10", "3.11", "3.12"], index=0)
    platform = st.selectbox("Platform", ["linux-x86_64"], index=0)
    st.info("Current model coverage: frozen DepLab package families and exact versions.")

uploaded = st.file_uploader("Upload requirements.txt", type=["txt"])
default_requirements = "numpy==1.21.6\npandas==1.3.5"
if uploaded:
    requirements_text = uploaded.getvalue().decode("utf-8", errors="replace")
else:
    requirements_text = st.text_area(
        "Or paste requirements.txt",
        value=default_requirements,
        height=160,
    )

question = st.text_input(
    "Question",
    value="Can I upgrade NumPy to 1.24.4?",
    placeholder="Can I upgrade NumPy to 1.24.4?",
)

if st.button("Analyze change", type="primary", use_container_width=True):
    if not requirements_text.strip() or not question.strip():
        st.error("Please provide requirements.txt and a question.")
    else:
        with st.spinner("GPT is parsing the request and DepLab is scoring covered pairs..."):
            result = service().analyze(
                AnalysisRequest(
                    requirements_text=requirements_text,
                    question=question,
                    python_version=python_version,
                    platform=platform,
                )
            )

        if result.status == "risk_found":
            st.warning(result.summary)
        elif result.status == "no_risk_predicted":
            st.success(result.summary)
        elif result.status == "coverage_unavailable":
            st.info(result.summary)
        else:
            st.error(result.summary)

        st.subheader("DepLab answer")
        st.write(result.answer or "No explanation was generated.")
        if result.model_scope == "deterministic_constraints_and_prediction":
            st.caption(
                "Evidence: published-constraint conflicts are deterministic. Model risk scores "
                "and compatibility warnings remain predictions. No environment was installed."
            )
        else:
            st.caption("Evidence: model prediction only. No environment was installed.")

        if result.errors:
            for message in result.errors:
                st.error(message)
        if result.warnings:
            with st.expander("Coverage notes", expanded=True):
                for message in result.warnings:
                    st.write(f"- {message}")

        if result.pair_risks:
            st.subheader("Scored package pairs")
            st.dataframe(
                [
                    {
                        "Package pair": item.family,
                        "Related version": f"{item.related_package}=={item.related_version}",
                        "Risk score": round(item.risk_score, 3),
                        "Model also warns": item.predicted_failure,
                        "Published constraints allow": item.published_constraints_allow,
                        "Evidence": item.evidence_type.replace("_", " "),
                        "Blocking constraint": "; ".join(
                            ", ".join(conflict.blocking_specifiers) or conflict.requirement
                            for conflict in item.constraint_conflicts
                        ),
                        "Likely stage": item.likely_stage,
                    }
                    for item in result.pair_risks
                ],
                use_container_width=True,
                hide_index=True,
            )

        if result.alternatives:
            st.subheader("Recommended environments")
            st.dataframe(
                [
                    {
                        "Recommendation type": {
                            "achieves_requested_change": "Achieves requested change",
                            "keeps_current_version": "Keeps current version",
                            "different_upgrade_fallback": "Different upgrade fallback",
                            "downgrade_fallback": "Downgrade fallback — does not achieve your goal",
                        }[item.category],
                        "Suggested changes": ", ".join(
                            f"{change.package} {change.from_version} -> {change.to_version}"
                            for change in item.changes
                        ) or "No version changes",
                        "Maximum risk score": round(item.maximum_risk_score, 3),
                        "Model warning": item.predicted_failure,
                        "Covered pairs": item.covered_pairs,
                    }
                    for item in result.alternatives
                ],
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("Structured backend result"):
            st.code(json.dumps(result.to_dict(), indent=2), language="json")

st.divider()
st.caption(
    "Current hybrid validation accuracy: 78.81% on 840 held-out rows. "
    "Those rows became validation data, so this is not an untouched final-test claim."
)
