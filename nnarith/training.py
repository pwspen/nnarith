from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from gen import generate_dataset_for_range

from nnarith.analysis.base import Analysis
from nnarith.analysis.heatmaps import HeatmapAnalysis
from nnarith.analysis.loss_curves import LossCurveAnalysis, LossCurveAnalysisConfig
from nnarith.analysis.planning import AnalysisPlan
from nnarith.config import ArchConfig, DataSweep, TrainingConfig
from nnarith.encoding import compute_encoding
from nnarith.history import HistoryRecord
from nnarith.model import build_mlp
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
from nnarith.training_ops import evaluate, resolve_device, train_epoch
def run_experiments(
    data_sweeps: Dict[str, DataSweep],
    training_configs: Dict[str, TrainingConfig],
    architectures: Dict[str, ArchConfig],
    *,
    results_dir: str = "results",
    torch_seed: Optional[int] = 0,
    analysis_plan: Optional[AnalysisPlan] = None,
    write_run_records: bool = True,
) -> None:
    if torch_seed is not None:
        torch.manual_seed(torch_seed)

    os.makedirs(results_dir, exist_ok=True)
    if analysis_plan is None:
        analysis_plan = AnalysisPlan(loss_curves=LossCurveAnalysisConfig())

    analyses: List[Analysis] = []
    loss_curve_analysis: Optional[LossCurveAnalysis] = None
    if analysis_plan.loss_curves is not None:
        loss_curve_analysis = LossCurveAnalysis(analysis_plan.loss_curves, results_dir)
        analyses.append(loss_curve_analysis)
    if analysis_plan.network_heatmaps is not None:
        analyses.append(HeatmapAnalysis(analysis_plan.network_heatmaps, results_dir))

    recorder = RunRecorder(results_dir) if write_run_records else None
    run_records: Dict[Tuple[str, str], RunRecord] = {}

    for sweep_name, sweep in data_sweeps.items():
        encoding = compute_encoding(sweep)
        train_dataset, _, train_operand_max, train_result_max = generate_dataset_for_range(
            sweep.train.minimum,
            sweep.train.maximum,
            sweep.operations,
            sweep.base,
            samples=sweep.train.samples,
            seed=sweep.seed,
            encoding=encoding,
        )

        print(f"\n=== Data sweep: {sweep_name} ===")
        print(
            f"Train range=[{sweep.train.minimum}, {sweep.train.maximum}] "
            f"samples={sweep.train.samples}"
        )
        for eval_name, split in sweep.evaluations.items():
            print(
                f"  Eval[{eval_name}] range=[{split.minimum}, {split.maximum}] "
                f"samples={split.samples}"
            )
        print(
            f"Operand digits={encoding.operand_digits}, "
            f"Result digits={encoding.result_digits}, "
            f"input_size={encoding.input_size}, target_size={encoding.target_size}"
        )

        eval_datasets: Dict[str, Tuple[Dataset, int, int]] = {}
        for index, (eval_name, split) in enumerate(sweep.evaluations.items(), start=1):
            eval_seed = sweep.seed + index
            dataset, _, operand_max_abs, result_max_abs = generate_dataset_for_range(
                split.minimum,
                split.maximum,
                sweep.operations,
                sweep.base,
                samples=split.samples,
                seed=eval_seed,
                encoding=encoding,
            )
            eval_datasets[eval_name] = (dataset, operand_max_abs, result_max_abs)

        dataset_operand_max = max(
            [train_operand_max, *[stats[1] for stats in eval_datasets.values()]]
        )
        dataset_result_max = max(
            [train_result_max, *[stats[2] for stats in eval_datasets.values()]]
        )
        print(f"operand_max_abs={dataset_operand_max}, result_max_abs={dataset_result_max}")

        train_range_summary = RangeSummary.from_split(sweep.train)
        evaluation_summaries = {
            name: RangeSummary.from_split(split) for name, split in sweep.evaluations.items()
        }
        combined_min = sweep.train.minimum
        combined_max = sweep.train.maximum
        for split in sweep.evaluations.values():
            combined_min = min(combined_min, split.minimum)
            combined_max = max(combined_max, split.maximum)
        combined_range_summary = RangeSummary.from_bounds(combined_min, combined_max)
        operation_names = tuple(
            getattr(operation, "__name__", repr(operation)) for operation in sweep.operations
        )

        for training_name, training_config in training_configs.items():
            device = resolve_device(training_config)

            print(f"\n--- Training config: {training_name} (device={device}) ---")

            train_loader = DataLoader(
                train_dataset,
                batch_size=training_config.batch_size,
                shuffle=True,
            )
            evaluation_loaders: Dict[str, DataLoader] = {
                eval_name: DataLoader(
                    dataset,
                    batch_size=training_config.batch_size,
                    shuffle=False,
                )
                for eval_name, (dataset, _, _) in eval_datasets.items()
            }

            criterion = nn.MSELoss()
            results = []

            training_summary = TrainingConfigSummary.from_config(training_config, device)
            run_context = RunContext(
                data_name=sweep_name,
                training_name=training_name,
                sweep=sweep,
                training_config=training_config,
                encoding=encoding,
                results_dir=results_dir,
                train_range=train_range_summary,
                combined_range=combined_range_summary,
            )
            for analysis in analyses:
                analysis.begin_run(run_context)

            run_key = (sweep_name, training_name)
            run_record = RunRecord(
                data_name=sweep_name,
                training_name=training_name,
                base=sweep.base,
                operations=operation_names,
                train_range=train_range_summary,
                evaluation_ranges=dict(evaluation_summaries),
                combined_range=combined_range_summary,
                training=training_summary,
            )
            run_records[run_key] = run_record

            for arch_name, arch in architectures.items():
                model = build_mlp(
                    encoding.input_size,
                    encoding.target_size,
                    arch.hidden_layers,
                    dropout=arch.dropout,
                ).to(device)
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=training_config.learning_rate, weight_decay=arch.l2
                )
                history = HistoryRecord(
                    train=[],
                    evaluations={name: [] for name in evaluation_loaders},
                )
                for epoch in range(training_config.epochs):
                    train_loss = train_epoch(
                        model, train_loader, optimizer, criterion, device
                    )
                    history.train.append(train_loss)
                    eval_losses: Dict[str, float] = {}
                    for eval_name, loader in evaluation_loaders.items():
                        loss_value = evaluate(model, loader, criterion, device)
                        history.evaluations[eval_name].append(loss_value)
                        eval_losses[eval_name] = loss_value
                    eval_summary = ", ".join(
                        f"{name}={loss:.6f}" for name, loss in eval_losses.items()
                    )
                    suffix = f" eval[{eval_summary}]" if eval_summary else ""
                    print(
                        f"{arch_name} ({sweep_name} | {training_name}): "
                        f"epoch={epoch + 1} train_loss={train_loss:.6f}{suffix}"
                    )
                final_train_loss = history.train[-1] if history.train else float("inf")
                final_eval_losses = {
                    name: losses[-1] for name, losses in history.evaluations.items()
                }
                mean_eval_loss = (
                    sum(final_eval_losses.values()) / max(1, len(final_eval_losses))
                    if final_eval_losses
                    else float("inf")
                )
                results.append((arch_name, final_train_loss, final_eval_losses, mean_eval_loss))

                history_summary = HistorySummary.from_history(history)
                arch_summary = ArchitectureSummary.from_config(arch)
                arch_record = ArchitectureRecord(
                    config=arch_summary,
                    history=history_summary,
                    final_train_loss=final_train_loss,
                    final_eval_losses=final_eval_losses,
                    mean_eval_loss=mean_eval_loss,
                )

                for analysis in analyses:
                    arch_artifacts = analysis.observe_model(
                        run_context, arch_name, arch, model, history
                    )
                    if arch_artifacts:
                        arch_record.artifacts.extend(arch_artifacts)

                run_record.architectures[arch_name] = arch_record

            if results:
                ranked = sorted(results, key=lambda item: item[3])
                print(f"\nRanked by mean eval loss ({sweep_name} | {training_name}):")
                for rank, (name, train_loss, eval_losses, _) in enumerate(
                    ranked, start=1
                ):
                    eval_report = ", ".join(
                        f"{eval_name}={loss:.6f}" for eval_name, loss in eval_losses.items()
                    )
                    print(
                        f"{rank}. {name}: train_loss={train_loss:.6f}"
                        + (f", eval[{eval_report}]" if eval_report else "")
                    )

            run_level_artifacts: List[ArtifactRecord] = []
            for analysis in analyses:
                run_level_artifacts.extend(analysis.finalize_run(run_context))
            if run_level_artifacts:
                run_record.artifacts.extend(run_level_artifacts)

    for analysis in analyses:
        analysis.finalize_experiment()

    if loss_curve_analysis is not None:
        for run_key, artifacts in loss_curve_analysis.artifacts_by_run.items():
            if not artifacts:
                continue
            record = run_records.get(run_key)
            if record is not None:
                record.artifacts.extend(artifacts)

    if recorder is not None:
        for run_key in sorted(run_records):
            recorder.write(run_records[run_key])


__all__ = ["evaluate", "resolve_device", "run_experiments", "train_epoch"]
