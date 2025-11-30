from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from nnarith.analysis.heatmaps import HeatmapAnalysisConfig
from nnarith.analysis.loss_curves import LossCurveAnalysisConfig


@dataclass
class AnalysisPlan:
    loss_curves: Optional[LossCurveAnalysisConfig] = None
    network_heatmaps: Optional[HeatmapAnalysisConfig] = None


__all__ = ["AnalysisPlan"]
