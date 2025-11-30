from __future__ import annotations

from typing import Sequence

from torch import nn

from nnarith.config import ArchConfig
from nnarith.history import HistoryRecord
from nnarith.records import ArtifactRecord, RunContext


class Analysis:
    """Base class for experiment-time observers."""

    def begin_run(self, context: RunContext) -> None:
        return None

    def observe_model(
        self,
        context: RunContext,
        arch_name: str,
        arch: ArchConfig,
        model: nn.Module,
        history: HistoryRecord,
    ) -> Sequence[ArtifactRecord]:
        return ()

    def finalize_run(self, context: RunContext) -> Sequence[ArtifactRecord]:
        return ()

    def finalize_experiment(self) -> None:
        return None


__all__ = ["Analysis"]
