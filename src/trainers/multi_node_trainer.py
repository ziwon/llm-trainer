import logging
import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from .base_trainer import BaseTrainer
from ..utils.monitoring import GPUTrainingMonitor, get_gpu_utilization

logger = logging.getLogger(__name__)

class DeepSpeedTrainer(Trainer):
    """Custom Trainer with advanced GPU monitoring for DeepSpeed."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu_monitor = GPUTrainingMonitor(log_interval_steps=self.args.logging_steps)
        
    def training_step(self, model, inputs, num_items_in_batch=None):
        step = self.state.global_step
        
        # Log GPU status before training step
        gpu_metrics = get_gpu_utilization()
        if gpu_metrics and step % self.args.logging_steps == 0:
            primary_gpu = list(gpu_metrics.keys())[0]
            mem_used = gpu_metrics[primary_gpu]["memory_used_gb"]
            mem_total = gpu_metrics[primary_gpu]["memory_total_gb"]
            logger.info(f"Step {step} - GPU Memory: {mem_used:.1f}GB/{mem_total:.1f}GB")
        
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()
        
        # Clean up GPU memory
        torch.cuda.empty_cache()
        
        self.accelerator.backward(loss)
        torch.cuda.empty_cache()
        
        return loss.detach() / self.args.gradient_accumulation_steps
    
    def log(self, logs):
        """Enhanced logging with GPU metrics."""
        super().log(logs)
        
        # Add GPU monitoring
        if hasattr(self, 'gpu_monitor') and 'loss' in logs:
            current_lr = self.get_lr()
            if isinstance(current_lr, list):
                current_lr = current_lr[0] if current_lr else 0.0
                
            self.gpu_monitor.log_metrics(
                step=self.state.global_step,
                loss=logs['loss'],
                learning_rate=current_lr,
                epoch=int(self.state.epoch) if self.state.epoch else 0
            )
    
    def on_save_checkpoint(self, checkpoint_dir: str):
        """Save GPU metrics alongside model checkpoint."""
        super().on_save_checkpoint(checkpoint_dir)
        if hasattr(self, 'gpu_monitor'):
            self.gpu_monitor.save_metrics(Path(checkpoint_dir))

class MultiNodeTrainer(BaseTrainer):
    """Trainer for multi-node training using DeepSpeed native integration."""

    def __init__(self, config_path: str):
        super().__init__(config_path)
        
        # Set compute dtype and attention implementation based on hardware
        if torch.cuda.is_bf16_supported():
            self.compute_dtype = torch.bfloat16
            self.attn_implementation = 'flash_attention_2'
        else:
            self.compute_dtype = torch.float16
            self.attn_implementation = 'sdpa'
            
        logger.info(f"Compute dtype: {self.compute_dtype}")
        logger.info(f"Attention implementation: {self.attn_implementation}")
        
        # Initialize model with optimized settings
        self._prepare_model()

    def _prepare_model(self):
        """Load and prepare model with DeepSpeed optimizations."""
        # Load model with flash attention and optimal dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model_name"],
            torch_dtype=self.compute_dtype,
            attn_implementation=self.attn_implementation,
            device_map="cuda"
        )
        
        # Enable gradient checkpointing to save memory
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False
        
        logger.info("Model prepared with gradient checkpointing and flash attention")

    def train(self):
        """Execute training using HuggingFace Trainer with DeepSpeed and advanced monitoring."""
        logger.info("Starting multi-node training with DeepSpeed.")
        
        # Log initial GPU status
        gpu_metrics = get_gpu_utilization()
        if gpu_metrics:
            logger.info("Initial GPU Status:")
            for gpu_id, metrics in gpu_metrics.items():
                logger.info(f"  {gpu_id}: {metrics['name']} - "
                           f"Memory: {metrics['memory_used_gb']:.1f}GB/"
                           f"{metrics['memory_total_gb']:.1f}GB "
                           f"({metrics['memory_utilization_percent']:.1f}%)")
        
        # Prepare dataset
        dataset = self.prepare_dataset()
        
        # Setup training arguments with DeepSpeed
        training_args = TrainingArguments(
            output_dir=self.config["output_dir"],
            learning_rate=self.config["learning_rate"],
            warmup_steps=self.config["warmup_steps"],
            lr_scheduler_type='cosine',
            per_device_train_batch_size=self.config["batch_size_per_device"],
            gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
            num_train_epochs=self.config["num_epochs"],
            save_steps=self.config["save_steps"],
            logging_steps=self.config["logging_steps"],
            save_strategy="steps",
            eval_strategy="no",
            weight_decay=0.01,
            
            # DeepSpeed configuration
            deepspeed=self.config["deepspeed_config"],
            
            # Mixed precision settings
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            
            # Other optimizations
            remove_unused_columns=True,
            dataloader_num_workers=4,
            report_to='none'
        )
        
        # Setup tokenizer padding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create trainer with enhanced monitoring
        trainer = DeepSpeedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, 
                mlm=False
            )
        )
        
        # Start training with monitoring
        logger.info("Starting training with DeepSpeed and GPU monitoring...")
        try:
            trainer.train()
            
            # Save final GPU metrics
            if hasattr(trainer, 'gpu_monitor'):
                trainer.gpu_monitor.save_metrics(Path(self.config["output_dir"]))
                
        except Exception as e:
            logger.error(f"Training failed: {e}")
            
            # Log GPU status during failure
            gpu_metrics = get_gpu_utilization()
            if gpu_metrics:
                logger.error("GPU Status at failure:")
                for gpu_id, metrics in gpu_metrics.items():
                    logger.error(f"  {gpu_id}: Memory: {metrics['memory_used_gb']:.1f}GB/"
                                f"{metrics['memory_total_gb']:.1f}GB, "
                                f"Utilization: {metrics['gpu_utilization_percent']}%")
            raise
        
        # Save final model
        trainer.save_model()
        logger.info("Multi-node training completed successfully!")