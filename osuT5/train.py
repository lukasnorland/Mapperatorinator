import sys

import hydra
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration
from omegaconf import OmegaConf

from osuT5.config import TrainConfig
from osuT5.utils import (
    setup_args,
    train,
    train_profiling,
    load_model,
    get_scheduler,
    get_optimizer,
    get_dataloaders,
    get_shared_training_state,
)


def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())  # Total parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # Trainable params
    frozen_params = total_params - trainable_params  # Non-trainable (frozen) params

    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Frozen Parameters: {frozen_params:,}")


@hydra.main(config_path="../configs/train", config_name="v29", version_base="1.1")
def main(args: TrainConfig):
    args: TrainConfig = OmegaConf.to_object(args)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        cpu=args.device == "cpu",
        mixed_precision=args.precision,
        gradient_accumulation_steps=args.optim.grad_acc,
        log_with=args.logging.log_with,
        project_config=ProjectConfiguration(
            project_dir="..", logging_dir="tensorboard_logs"
        ),
        kwargs_handlers=[ddp_kwargs],
    )
    wandb_init = {
        "job_type": "training",
        "sync_tensorboard": args.profile.do_profile,
    }
    # Do not pass mode="online" from defaults — it overrides W&B's interactive
    # "offline" choice and env (WANDB_MODE), causing 401 when not logged in.
    if getattr(args.logging, "mode", None) not in (None, "online"):
        wandb_init["mode"] = args.logging.mode

    accelerator.init_trackers(
        "osuT5",
        init_kwargs={"wandb": wandb_init},
    )

    setup_args(args)

    shared = get_shared_training_state()
    model, tokenizer = load_model(
        args.pretrained_path,
        args,
        device=accelerator.device,
        # Ignore precision argument because that is handled by accelerator
        attn_implementation=args.attn_implementation,
        eval_mode=False
    )
    train_dataloader, test_dataloader = get_dataloaders(tokenizer, args, shared)

    if args.enable_lora:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(**args.lora)
        model = get_peft_model(model, lora_config)
        # lora_params = {n: p for n, p in model.named_parameters() if "lora" in n}
        # for n, p in lora_params.items():
        #     print(n, p.sum())
        model.print_trainable_parameters()

        if args.lora_resume_path:
            import os
            from safetensors.torch import load_file
            from peft.utils.save_and_load import set_peft_model_state_dict
            adapter_state = load_file(os.path.join(args.lora_resume_path, "adapter_model.safetensors"))
            load_result = set_peft_model_state_dict(model, adapter_state)
            missing = getattr(load_result, "missing_keys", [])
            unexpected = getattr(load_result, "unexpected_keys", [])
            non_lora_missing = [k for k in missing if "lora_" in k]
            assert not non_lora_missing, f"LoRA keys missing from adapter: {non_lora_missing[:5]}"
            assert not unexpected, f"Unexpected adapter keys: {unexpected[:5]}"
            print(f"Resumed LoRA adapter weights from {args.lora_resume_path}")

    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args, accelerator)

    if args.model.manual_norm_weights:
        print("Manually normalizing model weights")
        model.transformer.register_step_post_hook(optimizer)
        model.transformer.norm_weights_()

    print(model)
    print_model_parameters(model)

    # noinspection PyTypeChecker
    (
        model,
        optimizer,
        scheduler,
        train_dataloader,
        test_dataloader,
    ) = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader, test_dataloader
    )

    accelerator.register_for_checkpointing(tokenizer)

    if args.checkpoint_path:
        accelerator.load_state(args.checkpoint_path)
        shared.current_train_step = scheduler.scheduler.last_epoch // accelerator.num_processes + 1

    if args.compile and sys.platform == "win32" and args.device != "cpu":
        # torch.compile → Inductor needs Triton; CUDA Triton is not shipped for Windows.
        print("torch.compile disabled on Windows+CUDA (no Triton); using eager mode.")
        args.compile = False

    if args.compile:
        model = torch.compile(model)

    func = train_profiling if args.profile.do_profile else train

    func(
        model,
        train_dataloader,
        test_dataloader,
        accelerator,
        scheduler,
        optimizer,
        tokenizer,
        args,
        shared,
    )


if __name__ == "__main__":
    main()
