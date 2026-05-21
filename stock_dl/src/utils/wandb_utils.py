from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def wandb_enabled(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("wandb", {}).get("enabled", False))


def init_wandb(
    cfg: dict[str, Any],
    *,
    job_type: str,
    extra_config: dict[str, Any] | None = None,
):
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None
    try:
        import wandb
    except ImportError:
        print("W&B is enabled but wandb is not installed; skip logging.")
        return None

    run_config = dict(cfg)
    if extra_config:
        run_config = {**run_config, **extra_config}

    kwargs = {
        "entity": wandb_cfg.get("entity"),
        "project": wandb_cfg.get("project"),
        "name": os.environ.get("WANDB_RUN_NAME", wandb_cfg.get("run_name")),
        "group": wandb_cfg.get("group"),
        "job_type": job_type,
        "tags": wandb_cfg.get("tags"),
        "notes": wandb_cfg.get("notes"),
        "mode": os.environ.get("WANDB_MODE", wandb_cfg.get("mode", "online")),
        "config": run_config,
        "reinit": True,
    }
    run_id = os.environ.get("WANDB_RUN_ID", wandb_cfg.get("run_id"))
    if run_id:
        kwargs["id"] = run_id
        kwargs["resume"] = os.environ.get("WANDB_RESUME", str(wandb_cfg.get("resume", "allow")))
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    try:
        return wandb.init(**kwargs)
    except Exception as exc:
        print(f"W&B init failed; continue without logging: {exc}")
        return None


def wandb_log(run, data: dict[str, Any], *, step: int | None = None) -> None:
    if run is None:
        return
    try:
        run.log(data, step=step)
    except Exception as exc:
        print(f"W&B log failed: {exc}")


def wandb_summary_update(run, data: dict[str, Any]) -> None:
    if run is None:
        return
    try:
        run.summary.update(data)
    except Exception as exc:
        print(f"W&B summary update failed: {exc}")


def wandb_log_artifact(
    run,
    path: str | Path,
    *,
    name: str,
    artifact_type: str,
) -> None:
    if run is None:
        return
    path = Path(path)
    if not path.exists():
        return
    try:
        import wandb

        artifact = wandb.Artifact(name=name, type=artifact_type)
        if path.is_dir():
            artifact.add_dir(str(path))
        else:
            artifact.add_file(str(path))
        run.log_artifact(artifact)
    except Exception as exc:
        print(f"W&B artifact logging failed for {path}: {exc}")


def wandb_log_images(run, image_dir: str | Path, *, prefix: str = "figures") -> None:
    if run is None:
        return
    image_dir = Path(image_dir)
    if not image_dir.exists():
        return
    try:
        import wandb

        payload = {}
        for fp in sorted(image_dir.glob("*.png")):
            payload[f"{prefix}/{fp.stem}"] = wandb.Image(str(fp))
        if payload:
            run.log(payload)
    except Exception as exc:
        print(f"W&B image logging failed: {exc}")


def finish_wandb(run) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception as exc:
        print(f"W&B finish failed: {exc}")
