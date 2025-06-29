import torch
import logging
import time
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForLanguageModeling, 
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    Trainer,
    TrainingArguments
)
from .base_trainer import BaseTrainer
from ..utils.monitoring import GPUTrainingMonitor

logger = logging.getLogger(__name__)

class SingleGPUTrainer(BaseTrainer):
    """Enhanced single-GPU trainer with optimized training loop and comprehensive logging."""

    def __init__(self, config_path: str):
        super().__init__(config_path)
        # Model is already on the right device via base class
        if hasattr(self.model, 'to'):
            self.model.to(self.device)
        logger.info(f"SingleGPUTrainer initialized with device: {self.device}")

    def train(self):
        """Execute optimized single-GPU training with choice between Trainer and custom loop."""
        logger.info("🚀 Starting single-GPU training")
        start_time = time.time()
        
        try:
            # Choose training method based on configuration
            if self.config.get("use_custom_training_loop", False):
                logger.info("Using custom high-performance training loop")
                self.train_custom_loop()
            else:
                logger.info("Using Hugging Face Trainer")
                self._train_with_hf_trainer()
            
            # Log completion
            training_time = time.time() - start_time
            logger.info(f"✅ Single-GPU training completed in {training_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"❌ Single-GPU training failed: {e}")
            raise
    
    def _train_with_hf_trainer(self):
        """Training using Hugging Face Trainer (fallback method)."""
        # Prepare dataset
        dataset = self.prepare_dataset()
        logger.info(f"Dataset prepared: {len(dataset):,} examples")
        
        # Use Hugging Face Trainer
        training_args = self.get_training_arguments()
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=False,
            pad_to_multiple_of=8 if self.config.get("mixed_precision") == "bf16" else None
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            processing_class=self.tokenizer,  # Updated parameter name
        )
        
        # Resume from checkpoint if specified
        resume_from_checkpoint = self.config.get("resume_from_checkpoint")
        if resume_from_checkpoint:
            logger.info(f"Resuming training from: {resume_from_checkpoint}")
        
        # Start training
        logger.info("Starting training loop")
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save final model
        trainer.save_model()
        trainer.save_state()
        
        # Save training metrics
        metrics = {
            "training_loss": train_result.training_loss,
            "global_step": train_result.global_step,
            "total_flos": train_result.total_flos,
        }
        
        metrics_path = self.output_dir / "training_metrics.json"
        import json
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Training metrics saved to: {metrics_path}")
    
    def train_custom_loop(self):
        """High-performance custom training loop with GPU monitoring."""
        logger.info("🚀 Starting custom high-performance training loop")
        
        dataset = self.prepare_dataset()
        
        # Optimized data collator and dataloader
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=False,
            pad_to_multiple_of=8 if self.config.get("mixed_precision") in ["bf16", "fp16"] else None,
            return_tensors="pt"
        )
        
        # Optimized dataloader settings for high throughput
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config["batch_size_per_device"], 
            collate_fn=data_collator, 
            shuffle=True,
            num_workers=0,  # Disable multiprocessing temporarily to fix data collation
            pin_memory=self.config.get("dataloader_pin_memory", True),
            drop_last=True  # Ensures consistent batch sizes for better GPU utilization
        )
        
        # Optimizer with optimized settings
        optimizer_class = getattr(torch.optim, self.config.get("optimizer_class", "AdamW"))
        optimizer = optimizer_class(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            betas=(self.config.get("adam_beta1", 0.9), self.config.get("adam_beta2", 0.95)),
            eps=self.config.get("adam_epsilon", 1e-8),
            weight_decay=self.config.get("weight_decay", 0.1)
        )
        
        # Learning rate scheduler
        total_steps = len(dataloader) * self.config["num_epochs"]
        warmup_steps = int(total_steps * self.config.get("warmup_ratio", 0.03))
        
        if self.config.get("lr_scheduler_type", "cosine") == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        
        # Mixed precision scaler
        scaler = None
        if self.config.get("mixed_precision") == "fp16":
            scaler = torch.cuda.amp.GradScaler()
        
        # Initialize GPU monitoring
        gpu_monitor = None
        if self.config.get("gpu_utilization_monitoring", False):
            monitor_interval = self.config.get("monitor_interval_steps", 50)
            gpu_monitor = GPUTrainingMonitor(log_interval_steps=monitor_interval)
            logger.info(f"🔍 GPU monitoring enabled (interval: {monitor_interval} steps)")
        
        # Training loop
        self.model.train()
        global_step = 0
        best_loss = float('inf')
        
        logger.info(f"Starting training: {self.config['num_epochs']} epochs, {len(dataloader)} steps per epoch")
        logger.info(f"Effective batch size: {self.config['batch_size_per_device']} × {self.config.get('gradient_accumulation_steps', 1)} = {self.config['batch_size_per_device'] * self.config.get('gradient_accumulation_steps', 1)}")
        
        for epoch in range(self.config["num_epochs"]):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            
            logger.info(f"Starting epoch {epoch + 1}/{self.config['num_epochs']}")
            
            for step, batch in enumerate(dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass with mixed precision
                if self.config.get("mixed_precision") == "fp16" and scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                        loss = outputs.loss
                elif self.config.get("mixed_precision") == "bf16":
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        outputs = self.model(**batch)
                        loss = outputs.loss
                else:
                    outputs = self.model(**batch)
                    loss = outputs.loss
                
                # Gradient accumulation
                loss = loss / self.config.get("gradient_accumulation_steps", 1)
                
                # Backward pass
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                epoch_loss += loss.item()
                
                # Gradient update
                if (step + 1) % self.config.get("gradient_accumulation_steps", 1) == 0:
                    # Gradient clipping
                    if self.config.get("max_grad_norm", 1.0) > 0:
                        if scaler:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])
                    
                    # Optimizer step
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                # Logging with GPU monitoring
                if global_step % self.config["logging_steps"] == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    actual_loss = loss.item() * self.config.get("gradient_accumulation_steps", 1)
                    
                    if gpu_monitor:
                        gpu_monitor.log_metrics(global_step, actual_loss, current_lr, epoch + 1)
                    else:
                        logger.info(
                            f"Step {global_step:,}, Loss: {actual_loss:.4f}, "
                            f"LR: {current_lr:.2e}, Epoch: {epoch + 1}"
                        )
                
                # Save checkpoint
                if global_step > 0 and global_step % self.config["save_steps"] == 0:
                    self.save_model(f"checkpoint-{global_step}")
                    
                    # Save best model
                    avg_loss = epoch_loss / (step + 1)
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        self.save_model("best")
                        logger.info(f"Saved new best model with loss: {best_loss:.4f}")
            
            # End of epoch logging
            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / len(dataloader)
            logger.info(
                f"Epoch {epoch + 1} completed in {epoch_time:.2f}s, "
                f"Average loss: {avg_epoch_loss:.4f}"
            )
        
        # Save final model and GPU metrics
        self.save_model("final")
        
        if gpu_monitor:
            gpu_monitor.save_metrics(self.output_dir)
        
        logger.info("✅ Custom single-GPU training completed")