"""Modul evaluasi otomatis untuk bahan Bab 4 (detektor, classifier, end-to-end, benchmark)."""

from attire_compliance.evaluation.benchmark import run_latency_benchmark
from attire_compliance.evaluation.classifier import evaluate_classifier
from attire_compliance.evaluation.detector import evaluate_detector
from attire_compliance.evaluation.end_to_end import evaluate_end_to_end
from attire_compliance.evaluation.runner import run_full_evaluation

__all__ = [
    "evaluate_classifier",
    "evaluate_detector",
    "evaluate_end_to_end",
    "run_full_evaluation",
    "run_latency_benchmark",
]
