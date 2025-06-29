
import logging
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from accelerate import Accelerator
from accelerate.utils import set_seed
from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)

class MultiGPUTrainer(BaseTrainer):
    """Trainer for multi-GPU training using Accelerate."""

    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.accelerator = Accelerator(
            mixed_precision=self.config["mixed_precision"],
            gradient_accumulation_steps=self.config["gradient_accumulation_steps"]
        )
        set_seed(42)

    def train(self):
        """Executes the training loop for multiple GPUs."""
        logger.info("Starting multi-GPU training with Accelerate.")
        dataset = self.prepare_dataset()
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        dataloader = DataLoader(dataset, batch_size=self.config["batch_size_per_device"], collate_fn=data_collator, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["learning_rate"])
        
        self.model, optimizer, dataloader = self.accelerator.prepare(self.model, optimizer, dataloader)

        self.model.train()
        step = 0
        for epoch in range(self.config["num_epochs"]):
            if self.accelerator.is_main_process:
                logger.info(f"Starting epoch {epoch + 1}/{self.config['num_epochs']}")
            
            for batch in dataloader:
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    self.accelerator.backward(loss)
                    
                    optimizer.step()
                    optimizer.zero_grad()

                if self.accelerator.is_main_process and step % self.config["logging_steps"] == 0:
                    logger.info(f"Step {step}, Loss: {loss.item():.4f}")
                
                if self.accelerator.is_main_process and step > 0 and step % self.config["save_steps"] == 0:
                    self.save_model(f"checkpoint-{step}")
                
                step += 1
        
        if self.accelerator.is_main_process:
            self.save_model("final")
            logger.info("Multi-GPU training finished.")
