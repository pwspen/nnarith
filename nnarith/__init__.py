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
from nnarith.datasets import (
    ArithmeticDatasets,
    generate_arithmetic_datasets,
    generate_dataset_for_range,
)
from nnarith.encoding import (
    EncodingSpec,
    compute_encoding,
    decode_number,
    encode_number,
    required_digits,
)
from nnarith.presets import (
    Scenario,
    build_default_scenario,
    default_analysis_plan,
    run_default_experiment,
)
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
    "ArithmeticDatasets",
    "Scenario",
    "DataSweep",
    "EncodingSpec",
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
    "build_default_scenario",
    "build_mlp",
    "compute_encoding",
    "decode_number",
    "default_analysis_plan",
    "encode_number",
    "enumerate_architectures",
    "evaluate",
    "generate_arithmetic_datasets",
    "generate_dataset_for_range",
    "history_to_series",
    "multiply",
    "plot_filename",
    "required_digits",
    "resolve_device",
    "run_default_experiment",
    "run_experiments",
    "sanitize_label",
    "SplitConfig",
    "subtract",
    "train_epoch",
]
