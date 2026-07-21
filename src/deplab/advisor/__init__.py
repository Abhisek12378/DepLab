"""Reusable advisory backend for the DepLab chatbot and API."""

from .contracts import AnalysisRequest, AdvisoryResult
from .service import AdvisorService, build_default_service

__all__ = [
    "AnalysisRequest",
    "AdvisoryResult",
    "AdvisorService",
    "build_default_service",
]
