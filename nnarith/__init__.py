from __future__ import annotations

from nnarith.analysis.heatmaps import (
    HeatmapAnalysis,
    HeatmapAnalysisConfig,
    HeatmapNormalizer,
    absolute_error_normalizer,
)
from nnarith.analysis.loss_curves import LossCurveAnalysis, LossCurveAnalysisConfig
from nnarith.analysis.planning import AnalysisPlan
from nnarith.architecture import enumerate_architectures
from nnarith.config import ArchConfig, DataSweep, SplitConfig, TrainingConfig
from nnarith.encoding import compute_encoding, required_digits
from nnarith.history import HistoryRecord, LossSeries, PlotRequest, history_to_series
from nnarith.model import build_mlp
from nnarith.operations import add, multiply, subtract
from nnarith.records import (
    ArchitectureRecord,
    ArchitectureSummary,
    ArtifactRecord,
    HistorySummary,
    RangeSummary,
    RunContext,
    RunRecord,
    RunRecorder,
    TrainingConfigSummary,
)
from nnarith.training import evaluate, resolve_device, run_experiments, train_epoch
from nnarith.utils import plot_filename, sanitize_label

__all__ = [
    "AnalysisPlan",
    "ArchConfig",
    "ArchitectureRecord",
    "ArchitectureSummary",
    "ArtifactRecord",
    "DataSweep",
    "HeatmapAnalysis",
    "HeatmapAnalysisConfig",
    "HeatmapNormalizer",
    "HistoryRecord",
    "HistorySummary",
    "LossCurveAnalysis",
    "LossCurveAnalysisConfig",
    "LossSeries",
    "PlotRequest",
    "RangeSummary",
    "RunContext",
    "RunRecord",
    "RunRecorder",
    "TrainingConfig",
    "TrainingConfigSummary",
    "absolute_error_normalizer",
    "add",
    "build_mlp",
    "compute_encoding",
    "enumerate_architectures",
    "evaluate",
    "history_to_series",
    "multiply",
    "plot_filename",
    "required_digits",
    "resolve_device",
    "run_experiments",
    "sanitize_label",
    "SplitConfig",
    "subtract",
    "train_epoch",
]
