"""Experiment tracking with Comet.

A thin wrapper: every method is a no-op when tracking is off, so the training
code reads the same either way. Set COMET_API_KEY (and optionally
COMET_WORKSPACE) in the environment to log online.
"""

from __future__ import annotations

from pathlib import Path


class Tracker:
    def __init__(self, experiment=None):
        self.experiment = experiment

    @classmethod
    def start(cls, enabled: bool, project: str, run_name: str, tags: list[str] | None = None,
              offline_dir: str | None = None, experiment_key: str | None = None) -> "Tracker":
        if not enabled:
            return cls(None)
        import comet_ml

        config = comet_ml.ExperimentConfig(name=run_name, tags=tags, offline_directory=offline_dir,
                                           auto_param_logging=False, log_code=True)
        experiment = comet_ml.start(
            project_name=project,
            online=offline_dir is None,
            experiment_key=experiment_key,
            mode="get" if experiment_key else "create",
            experiment_config=config,
        )
        return cls(experiment)

    @property
    def key(self) -> str | None:
        return self.experiment.get_key() if self.experiment else None

    def log_params(self, params: dict) -> None:
        if self.experiment:
            self.experiment.log_parameters(params)

    def log_metrics(self, metrics: dict, step: int | None = None, epoch: int | None = None, prefix: str | None = None) -> None:
        if self.experiment:
            flat = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            self.experiment.log_metrics(flat, prefix=prefix, step=step, epoch=epoch)

    def log_summary(self, summary: dict, prefix: str, step: int | None = None, epoch: int | None = None) -> None:
        """Log a metrics.summarize() result: overall, per type, and per question."""
        self.log_metrics(summary["overall"], step, epoch, prefix=prefix)
        for name, metrics in summary["by_type"].items():
            self.log_metrics(metrics, step, epoch, prefix=f"{prefix}/{name}")
        for qid, metrics in summary["by_question"].items():
            self.log_metrics(metrics, step, epoch, prefix=f"{prefix}/q/{qid}")

    def log_reliability_diagram(self, name: str, bins: list[dict], step: int | None = None) -> None:
        if not self.experiment:
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))
        centers = [(b["lower"] + b["upper"]) / 2 for b in bins]
        ax.bar(centers, [b["accuracy"] for b in bins], width=0.09, alpha=0.8, label="accuracy")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
        ax.set_xlabel("confidence of top answer")
        ax.set_ylabel("accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(name)
        ax.legend(loc="upper left")
        self.experiment.log_figure(figure_name=name, figure=fig, step=step)
        plt.close(fig)

    def log_confusion_matrix(self, name: str, labels: list[str], y_true: list[int], y_pred: list[int], step: int | None = None) -> None:
        if self.experiment:
            self.experiment.log_confusion_matrix(y_true=y_true, y_predicted=y_pred, labels=labels, title=name,
                                                 file_name=f"{name.replace('/', '__')}.json", step=step)

    def log_dataset_artifact(self, name: str, files: list[Path], metadata: dict) -> None:
        if not self.experiment:
            return
        import comet_ml

        artifact = comet_ml.Artifact(name=name, artifact_type="dataset", metadata=metadata)
        for path in files:
            artifact.add(str(path))
        try:
            self.experiment.log_artifact(artifact)
        except NotImplementedError:  # offline experiments can't log artifacts; keep the files as assets
            for path in files:
                self.experiment.log_asset(str(path), metadata=metadata)

    def log_model(self, name: str, folder: Path, metadata: dict) -> None:
        if self.experiment:
            self.experiment.log_model(name, str(folder), metadata=metadata)

    def log_other(self, key: str, value) -> None:
        if self.experiment:
            self.experiment.log_other(key, value)

    def end(self) -> None:
        if self.experiment:
            self.experiment.end()
