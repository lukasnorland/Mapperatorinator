import os
from pathlib import Path

import torch
from accelerate.utils import set_seed

from ..config import TrainConfig


def _resolve_data_paths_to_hydra_orig_cwd(args: TrainConfig) -> None:
    """Hydra changes cwd to outputs under logs/; relative dataset paths must stay repo-rooted."""
    try:
        from hydra.utils import get_original_cwd
    except ImportError:
        return
    try:
        root = Path(get_original_cwd()).resolve()
    except ValueError:
        return
    data = args.data
    for name in ("train_dataset_path", "test_dataset_path"):
        p = getattr(data, name, None)
        if p and not Path(p).expanduser().is_absolute():
            setattr(data, name, str((root / p).resolve()))
    sap = getattr(data, "snapbeat_audio_path", None)
    if sap and str(sap).strip() and not Path(sap).expanduser().is_absolute():
        data.snapbeat_audio_path = str((root / sap).resolve())


def check_args_and_env(args: TrainConfig) -> None:
    assert args.optim.batch_size % args.optim.grad_acc == 0
    # Train log must happen before eval log
    assert args.eval.every_steps % args.logging.every_steps == 0

    if args.device == "gpu":
        assert torch.cuda.is_available(), "We use GPU to train/eval the model"


def opti_flags(args: TrainConfig) -> None:
    # This lines reduce training step by 2.4x
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def update_args_with_env_info(args: TrainConfig) -> None:
    slurm_id = os.getenv("SLURM_JOB_ID")

    if slurm_id is not None:
        args.slurm_id = slurm_id
    else:
        args.slurm_id = "none"

    args.working_dir = os.getcwd()


def setup_args(args: TrainConfig) -> None:
    check_args_and_env(args)
    update_args_with_env_info(args)
    opti_flags(args)
    _resolve_data_paths_to_hydra_orig_cwd(args)

    if args.seed is not None:
        set_seed(args.seed)
