# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from safetensors.torch import save_file, load_file
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.fsdp import shard_model, apply_ac
from distributed.util import (
    _configure_model, 
    init_distributed, 
)
from einops import rearrange
from modules.utils import (
    load_transformer,
)
from utils import (
    init_logger, 
    logger, 
    get_mesh_id, 
    sample_timestep_id,
    data_seq_to_patch,
    warmup_constant_lambda,
    FlowMatchScheduler
)

from dataset import MultiLatentLeRobotDataset
from dataset import BucketedDistributedBatchSampler
import gc


class Trainer:
    def __init__(self, config):
        if config.enable_wandb and config.rank == 0:
            try:
                import wandb
            except ImportError as exc:
                raise RuntimeError(
                    "enable_wandb=True requires the `wandb` package. "
                    "Install the post-training extras or disable WandB."
                ) from exc

            for env_name in ("WANDB_API_KEY", "WANDB_BASE_URL", "WANDB_TEAM_NAME"):
                if os.environ.get(env_name) == "":
                    os.environ.pop(env_name, None)

            login_kwargs = {}
            wandb_base_url = os.environ.get("WANDB_BASE_URL")
            wandb_api_key = os.environ.get("WANDB_API_KEY")
            wandb_mode = os.environ.get("WANDB_MODE", "online")
            wandb_entity = os.environ.get("WANDB_TEAM_NAME") or None
            if wandb_base_url:
                login_kwargs["host"] = wandb_base_url
            if wandb_api_key:
                login_kwargs["key"] = wandb_api_key
            if wandb_mode == "online" and login_kwargs:
                wandb.login(**login_kwargs)
            self.wandb = wandb
            init_kwargs = dict(
                project=os.getenv("WANDB_PROJECT", "va_robotwin"),
                config=config,
                mode=wandb_mode,
                name=os.getenv("WANDB_RUN_NAME", "robotwin_train"),
            )
            if wandb_entity:
                init_kwargs["entity"] = wandb_entity
            self.wandb.init(**init_kwargs)
            logger.info(f"WandB logging enabled (mode={wandb_mode})")
        else:
            self.wandb = None
        self.step = 0
        self.config = config
        self.device = torch.device(f"cuda:{config.local_rank}")
        self.dtype = config.param_dtype
        self.patch_size = config.patch_size

        # Load models
        logger.info("Loading models...")

        # Load and shard transformer with FSDP
        logger.info("Loading transformer...")

        if hasattr(config, 'resume_from') and config.resume_from:
            transformer_path = os.path.join(config.resume_from, 'transformer')
            if config.rank == 0:
                logger.info(f"Resuming from checkpoint: {transformer_path}")
        else:
            transformer_path = os.path.join(config.wan22_pretrained_model_name_or_path, 'transformer')

        model_overrides = {
            "enable_action_residual_adapter": config.enable_action_residual_adapter,
            "action_adapter_dim": config.action_adapter_dim,
            "action_adapter_dropout": config.action_adapter_dropout,
        }
        self.transformer = load_transformer(
            transformer_path,
            torch_dtype=torch.float32,
            torch_device='cpu',
            model_overrides=model_overrides,
        )

        logger.info("Setting up activation checkpointing ...")
        apply_ac(self.transformer)

        logger.info(
            "Setting up distributed model (%s)...",
            os.environ.get("LINGBOT_VA_DIST_STRATEGY", "fsdp"),
        )
        shard_fn = shard_model
        self.transformer = _configure_model(
            model=self.transformer,
            shard_fn=shard_fn,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=False,
        )
        self.transformer.train()
        self.configure_trainable_modules()
        if self.config.rank == 0:
            self.log_trainable_params()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in self.transformer.parameters() if p.requires_grad],
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
            weight_decay=config.weight_decay,
            fused=True,
            foreach=False,
        )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, 
            lr_lambda=lambda step: warmup_constant_lambda(step, warmup_steps=config.warmup_steps))

        # Setup dataloaders
        logger.info("Setting up datasets...")
        train_dataset = MultiLatentLeRobotDataset(config=config)
        if config.batch_size > 1:
            batch_sampler = BucketedDistributedBatchSampler(
                train_dataset.bucket_keys,
                batch_size=config.batch_size,
                num_replicas=config.world_size,
                rank=config.rank,
                shuffle=True,
                seed=42,
                pad_batches=True,
            )
            self.train_loader = DataLoader(
                train_dataset,
                batch_sampler=batch_sampler,
                num_workers=config.load_worker,
            )
        else:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=config.world_size,
                rank=config.rank,
                shuffle=True,
                seed=42
            ) if config.world_size > 1 else None
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=(train_sampler is None), 
                num_workers=config.load_worker,
                sampler=train_sampler,
            )

        self.train_scheduler_latent = FlowMatchScheduler(shift=self.config.snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_latent.set_timesteps(1000, training=True)
        self.train_scheduler_action = FlowMatchScheduler(shift=self.config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_action.set_timesteps(1000, training=True)

        self.save_dir = Path(config.save_root) / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        self.train_loader_iter = None
        # if hasattr(config, 'resume_from') and config.resume_from:
        #     self._load_training_state(config.resume_from)

    def _base_transformer(self):
        return self.transformer.module if hasattr(self.transformer, "module") else self.transformer

    def configure_trainable_modules(self):
        base_transformer = self._base_transformer()
        base_transformer.requires_grad_(False)

        if not self.config.freeze_backbone:
            base_transformer.blocks.requires_grad_(True)
            base_transformer.norm_out.requires_grad_(True)

        if not self.config.freeze_embeddings:
            for module_name in [
                "patch_embedding_mlp",
                "action_embedder",
                "condition_embedder",
                "condition_embedder_action",
            ]:
                module = getattr(base_transformer, module_name, None)
                if module is not None:
                    module.requires_grad_(True)

        if self.config.train_action_adapter:
            adapters = getattr(base_transformer, "action_residual_adapters", None)
            if adapters is not None:
                adapters.requires_grad_(True)

        if self.config.train_action_head:
            base_transformer.action_proj_out.requires_grad_(True)

        if self.config.train_video_heads:
            base_transformer.proj_out.requires_grad_(True)

        if self.config.train_time_embedder:
            for name in ["condition_embedder", "condition_embedder_action", "scale_shift_table"]:
                module = getattr(base_transformer, name, None)
                if module is not None:
                    module.requires_grad_(True)

    def log_trainable_params(self):
        base_transformer = self._base_transformer()
        total = 0
        trainable = 0
        trainable_names = []
        for name, param in base_transformer.named_parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
                trainable_names.append(name)

        logger.info("Trainable params: %d / %d", trainable, total)
        for name in trainable_names[:50]:
            logger.info("trainable: %s", name)

    def _get_next_batch(self):
        """Get next batch from iterator, reset if epoch is finished."""
        if self.train_loader_iter is None:
            self.train_loader_iter = iter(self.train_loader)
        
        try:
            batch = next(self.train_loader_iter)
        except StopIteration:
            # Reset sampler and iterator when epoch finishes
            epoch_owner = None
            if hasattr(self.train_loader, "batch_sampler") and hasattr(self.train_loader.batch_sampler, "set_epoch"):
                epoch_owner = self.train_loader.batch_sampler
            elif hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "set_epoch"):
                epoch_owner = self.train_loader.sampler
            if epoch_owner is not None:
                epoch_owner.set_epoch(getattr(epoch_owner, "epoch", 0) + 1)
            self.train_loader_iter = iter(self.train_loader)
            batch = next(self.train_loader_iter)
        
        return batch

    @torch.no_grad()
    def _add_noise(self, latent, train_scheduler, action_mask=False, action_mode=False, noisy_cond_prob=0.):
        B, C, F, H, W = latent.shape

        timestep_ids = sample_timestep_id(batch_size=F, num_train_timesteps=train_scheduler.num_train_timesteps)
        noise = torch.zeros_like(latent).normal_()
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=self.device)
        noisy_latents =train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets =train_scheduler.training_target(latent, noise, timesteps)

        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1
        
        latent_grid_id = get_mesh_id(
            latent.shape[-3] // patch_f,  # F
            latent.shape[-2] // patch_h,  # H
            latent.shape[-1] // patch_w,  # W
            t=1 if action_mode else 0,  # 1 for action mode (0 for latent), not used
            f_w=1,
            f_shift=0,
            action=action_mode
        ).to(self.device)  # shape: [4, seq_len]
        latent_grid_id = latent_grid_id[None].repeat(B, 1, 1)

        if torch.rand(1).item() < noisy_cond_prob:
            cond_timestep_ids = sample_timestep_id(
                    batch_size=F,
                    min_timestep_bd=0.5, 
                    max_timestep_bd=1.0, 
                    num_train_timesteps=train_scheduler.num_train_timesteps,
                )
            noise = torch.zeros_like(latent).normal_()
            cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(device=self.device)
            latent = train_scheduler.add_noise(latent, noise, cond_timesteps, t_dim=2)
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        if action_mask is not None:
            noisy_latents *= action_mask.float()
            targets *= action_mask.float()
            latent *= action_mask.float()

        return dict(
            timesteps=timesteps[None].repeat(B, 1),
            noisy_latents=noisy_latents,
            targets=targets,
            latent=latent,
            cond_timesteps=cond_timesteps[None].repeat(B, 1),
            grid_id=latent_grid_id,
        )

    @torch.no_grad()
    def _prepare_input_dict(self, batch_dict):
        """Prepare input dict following infer code pattern from wan_va_server.py."""
        # Generate grid_id following infer code (no batch dimension yet)
        # For action mode: get_mesh_id(shape[-3], shape[-2], shape[-1], t=1, f_w=1, f_shift, action=True)
        latent_dict = self._add_noise(
            latent=batch_dict['latents'], 
            train_scheduler=self.train_scheduler_latent, 
            action_mask=None, 
            action_mode=False,
            noisy_cond_prob=0.5)
        
        action_dict = self._add_noise(
            latent=batch_dict['actions'], 
            train_scheduler=self.train_scheduler_action, 
            action_mask=batch_dict['actions_mask'], 
            action_mode=True,
            noisy_cond_prob=0.0)

        latent_dict['text_emb'] = batch_dict['text_emb']
        action_dict['text_emb'] = batch_dict['text_emb']
        action_dict['actions_mask'] = batch_dict['actions_mask']

        input_dict = {
            'latent_dict': latent_dict,
            'action_dict': action_dict,
            'chunk_size': torch.randint(1, 5, (1,)).item(),
            'window_size': torch.randint(4, 65, (1,)).item(),
        }
        return input_dict

    def convert_input_format(self, input_dict):
        """Convert input dict to match transformer input format if needed."""
        for key, value in input_dict.items():
            input_dict[key] = value.to(self.device)#.to(self.dtype)
        return input_dict

    def compute_loss(self,
        input_dict,
        pred
    ):
        latent_pred, action_pred = pred
        action_pred = rearrange(action_pred, 'b (f n) c -> b c f n 1', f=input_dict['action_dict']['targets'].shape[-3])
        latent_pred = data_seq_to_patch(
                        self.patch_size, latent_pred,
                        input_dict['latent_dict']['targets'].shape[-3], input_dict['latent_dict']['targets'].shape[-2],
                        input_dict['latent_dict']['targets'].shape[-1], batch_size=latent_pred.shape[0])
        Bn, Fn = input_dict['latent_dict']['timesteps'].shape
        latent_loss_weight = self.train_scheduler_latent.training_weight(input_dict['latent_dict']['timesteps'].flatten()).reshape(Bn, Fn)
        action_loss_weight = self.train_scheduler_action.training_weight(input_dict['action_dict']['timesteps'].flatten()).reshape(Bn, Fn)

        # Frame-wise video loss calculation
        latent_loss = F.mse_loss(latent_pred.float(), input_dict['latent_dict']['targets'].float().detach(), reduction='none')
        latent_loss = latent_loss * latent_loss_weight[:, None, :, None, None]
        # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        latent_loss = latent_loss.permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        latent_loss = latent_loss.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        # Sum per frame and compute mask per frame
        latent_loss_per_frame = latent_loss.sum(dim=1)  # (B*F,)
        latent_mask_per_frame = torch.ones_like(latent_loss).sum(dim=1)  # (B*F,)
        latent_loss = (latent_loss_per_frame / (latent_mask_per_frame + 1e-6)).mean()

        # Frame-wise action loss calculation
        action_loss = F.mse_loss(action_pred.float(), input_dict['action_dict']['targets'].float().detach(), reduction='none')
        action_loss = action_loss * action_loss_weight[:, None, :, None, None]
        action_loss = action_loss * input_dict['action_dict']['actions_mask'].float()
        # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        action_loss = action_loss.permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_mask = input_dict['action_dict']['actions_mask'].float().permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_loss = action_loss.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        action_mask = action_mask.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        # Sum per frame and normalize by mask per frame
        action_loss_per_frame = action_loss.sum(dim=1)  # (B*F,)
        action_mask_per_frame = action_mask.sum(dim=1)  # (B*F,)
        action_loss = (action_loss_per_frame / (action_mask_per_frame + 1e-6)).mean()

        return latent_loss / self.gradient_accumulation_steps, action_loss / self.gradient_accumulation_steps

    def _train_step(self, batch, batch_idx):
        """Train a single batch, returns losses for logging."""
        debug_first_step = self.config.rank == 0 and self.step == 0
        step_t0 = time.perf_counter()
        batch = self.convert_input_format(batch)
        if debug_first_step:
            logger.info("First train step: batch moved to device in %.2fs", time.perf_counter() - step_t0)
        input_dict = self._prepare_input_dict(batch)
        if debug_first_step:
            logger.info("First train step: input_dict prepared in %.2fs", time.perf_counter() - step_t0)
        
        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        sync_context = nullcontext()
        if dist.is_initialized() and not should_sync:
            if hasattr(self.transformer, "set_requires_gradient_sync"):
                self.transformer.set_requires_gradient_sync(False)
            elif hasattr(self.transformer, "no_sync"):
                sync_context = self.transformer.no_sync()
        elif hasattr(self.transformer, "set_requires_gradient_sync"):
            self.transformer.set_requires_gradient_sync(True)

        with sync_context:
            forward_t0 = time.perf_counter()
            output = self.transformer(input_dict, train_mode=True)
            if debug_first_step:
                logger.info("First train step: transformer forward finished in %.2fs", time.perf_counter() - forward_t0)
            loss_t0 = time.perf_counter()
            latent_loss, action_loss = self.compute_loss(input_dict, output)
            if debug_first_step:
                logger.info("First train step: loss computed in %.2fs", time.perf_counter() - loss_t0)
            loss = latent_loss + action_loss
            backward_t0 = time.perf_counter()
            loss.backward()
            if debug_first_step:
                logger.info("First train step: backward finished in %.2fs", time.perf_counter() - backward_t0)

        losses = {'latent_loss': latent_loss.detach(), 'action_loss': action_loss.detach()}
        
        # Only update weights after accumulating gradients
        if should_sync:
            optim_t0 = time.perf_counter()
            total_norm = torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 2.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            if debug_first_step:
                logger.info("First train step: optimizer phase finished in %.2fs", time.perf_counter() - optim_t0)
            
            losses['total_norm'] = total_norm
            losses['should_log'] = True
        else:
            losses['should_log'] = False

        return losses

    def save_checkpoint(self,):
        """Save model checkpoint in the same format as pretrained model."""
        try:
            base_transformer = self.transformer.module if hasattr(self.transformer, "module") else self.transformer
            if hasattr(self.transformer, "module"):
                state_dict = base_transformer.state_dict()
            else:
                state_dict = get_model_state_dict(
                    self.transformer,
                    options=StateDictOptions(full_state_dict=True, cpu_offload=True),
                )
            state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
            # optim_state = get_optimizer_state_dict(
            #         self.transformer, self.optimizer,
            #         options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            #     )

            # Only rank 0 saves the checkpoint
            if self.config.rank == 0:
                checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                # Save transformer in the same format as pretrained model
                transformer_dir = checkpoint_dir / "transformer"
                transformer_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"Saving transformer to {transformer_dir}")

                # Manually save in diffusers format (outside FSDP context to avoid deadlock)
                # Save model weights
                model_file = transformer_dir / "diffusion_pytorch_model.safetensors"
                save_file(state_dict_bf16, model_file)

                # Save config (copy from original transformer config and update _name_or_path)
                config_file = transformer_dir / "config.json"
                config_dict = dict(base_transformer.config)
                config_dict.pop('_name_or_path', None)
                with open(config_file, 'w') as f:
                    json.dump(config_dict, f, indent=2)

                # # Save optimizer state and training metadata in PyTorch format
                # training_state_path = checkpoint_dir / "training_state.pt"
                # logger.info(f"Saving training state to {training_state_path}")
                # torch.save({
                #     'step': self.step,
                #     'optimizer_state_dict': optim_state,
                #     'config': vars(self.config),
                # }, training_state_path)

                logger.info(f"Checkpoint saved successfully at step {self.step}")

        except Exception as e:
            if self.config.rank == 0:
                logger.error(f"Failed to save checkpoint: {e}")
                import traceback
                logger.error(traceback.format_exc())

    def _load_training_state(self, checkpoint_path):
        """Load training state (optimizer + step) after FSDP and optimizer creation."""
        checkpoint_dir = Path(checkpoint_path)
        training_state_path = checkpoint_dir / "training_state.pt"

        if not training_state_path.exists():
            if self.config.rank == 0:
                logger.warning(f"Training state not found: {training_state_path}, starting from step 0")
            return

        if self.config.rank == 0:
            logger.info(f"Loading training state from {training_state_path}")

        # All ranks load the training state directly
        training_state = torch.load(training_state_path, map_location='cpu', weights_only=False)

        # All ranks load optimizer state (required for FSDP)
        set_optimizer_state_dict(
            self.transformer, self.optimizer,
            optim_state_dict=training_state['optimizer_state_dict'],
            options=StateDictOptions(full_state_dict=True, strict=False)
        )
        self.step = training_state.get('step', 0)

        if self.config.rank == 0:
            logger.info(f"Training state loaded, resuming from step {self.step}")

    def train(self):
        """Main training loop - train by steps instead of epochs."""
        logger.info(f"Starting training for {self.config.num_steps} steps...")
        self.transformer.train()

        progress_bar = tqdm(
            total=self.config.num_steps,
            desc="Training",
            disable=(self.config.rank != 0),
            leave=True,
            dynamic_ncols=True,
            initial=self.step
        )

        self.optimizer.zero_grad()
        accumulated_latent_losses = []
        accumulated_action_losses = []
        step_in_accumulation = 0

        while self.step < self.config.num_steps:
            # Get next batch (handles epoch reset automatically)
            if self.config.rank == 0 and self.step == 0 and step_in_accumulation == 0:
                logger.info("Fetching first batch...")
            batch_fetch_start = time.perf_counter()
            batch = self._get_next_batch()
            batch_fetch_elapsed = time.perf_counter() - batch_fetch_start
            if self.config.rank == 0 and self.step == 0 and step_in_accumulation == 0:
                logger.info(f"First batch fetched in {batch_fetch_elapsed:.2f}s")
            
            if self.config.rank == 0 and self.step == 0 and step_in_accumulation == 0:
                logger.info("Running first train step...")
            train_step_start = time.perf_counter()
            losses = self._train_step(batch, step_in_accumulation)
            train_step_elapsed = time.perf_counter() - train_step_start
            if self.config.rank == 0 and self.step == 0:
                logger.info(
                    "Train micro-step finished: accum_idx=%d fetch=%.2fs step=%.2fs",
                    step_in_accumulation,
                    batch_fetch_elapsed,
                    train_step_elapsed,
                )
            
            # Accumulate losses for logging
            accumulated_latent_losses.append(losses['latent_loss'])
            accumulated_action_losses.append(losses['action_loss'])
            step_in_accumulation += 1

            # Log and checkpoint when optimizer steps
            if losses['should_log']:
                global_step = self.step + 1
                lr = self.lr_scheduler.get_last_lr()[0]

                local_loss_sums = torch.stack(
                    [
                        torch.stack(accumulated_latent_losses).sum(),
                        torch.stack(accumulated_action_losses).sum(),
                    ]
                )
                mean_loss_sums = local_loss_sums.clone()
                max_loss_sums = local_loss_sums.clone()
                if dist.is_initialized():
                    dist.all_reduce(mean_loss_sums, op=dist.ReduceOp.AVG)
                    dist.all_reduce(max_loss_sums, op=dist.ReduceOp.MAX)
                latent_loss_show = mean_loss_sums[0].detach().cpu().item()
                action_loss_show = mean_loss_sums[1].detach().cpu().item()
                max_latent_loss_show = max_loss_sums[0].detach().cpu().item()
                max_action_loss_show = max_loss_sums[1].detach().cpu().item()

                # Clear accumulated losses
                accumulated_latent_losses = []
                accumulated_action_losses = []
                step_in_accumulation = 0

                if self.config.gc_interval > 0 and global_step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                if self.config.rank == 0:
                    total_norm = losses['total_norm']
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        'latent_loss': f'{latent_loss_show:.4f}',
                        'action_loss': f'{action_loss_show:.4f}',
                        'step': global_step,
                        'grad_norm': f'{total_norm.item():.2f}',
                        'lr': f'{lr:.2e}'
                    })
                    if self.config.enable_wandb:
                        self.wandb.log({
                            'loss_metrics/global_avg_video_loss': latent_loss_show,
                            'loss_metrics/global_avg_action_loss': action_loss_show,
                            'loss_metrics/global_max_video_loss': max_latent_loss_show,
                            'loss_metrics/global_max_action_loss': max_action_loss_show,
                            'grad_norm': total_norm.item(),
                            'lr': lr,
                        }, step=global_step)
                
                self.step = global_step
                
                if self.step % self.config.save_interval == 0:
                    if self.config.rank == 0:
                        logger.info(f"Starting save model at step {self.step}")
                    self.save_checkpoint()

        progress_bar.close()
        logger.info("Training completed!")


def run(args):
    """Main entry point."""
    config = VA_CONFIGS[args.config_name]

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    init_distributed(world_size, local_rank, rank)

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    if args.save_root is not None:
        config.save_root = args.save_root

    if rank == 0:
        logger.info(f"Using config: {args.config_name}")
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")
        logger.info(f"Dist strategy: {os.environ.get('LINGBOT_VA_DIST_STRATEGY', 'fsdp')}")

    trainer = Trainer(config)
    trainer.train()
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Train WAN model for robotics")
    parser.add_argument(
        "--config-name",
        type=str,
        default='robotwin_train',
        help="Config name",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default=None,
        help="Root directory for saving checkpoints",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()
