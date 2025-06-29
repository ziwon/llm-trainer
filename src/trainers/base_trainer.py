import os
import json
import yaml
import torch
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

logger = logging.getLogger(__name__)

class BaseTrainer(ABC):
    """Base class for all trainers with DGX B200 optimizations and comprehensive configuration support."""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._setup_output_dir()
        self._apply_dgx_optimizations()
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.device = self._get_device()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration from JSON or YAML file."""
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.suffix == '.json':
                    config = json.load(f)
                else:
                    config = yaml.safe_load(f)
            
            # Set default values for missing keys
            default_config = {
                "mixed_precision": "bf16",
                "use_flash_attention_2": True,
                "gradient_checkpointing": True,
                "dataloader_num_workers": 8,
                "dataloader_pin_memory": True,
                "max_length": 2048,
                "padding_side": "right",
                "use_fast_tokenizer": True,
                "torch_compile": False,
                "optim": "adamw_torch_fused",
                "adam_beta1": 0.9,
                "adam_beta2": 0.95,
                "adam_epsilon": 1e-8,
                "weight_decay": 0.1,
                "max_grad_norm": 1.0,
                "warmup_ratio": 0.03,
                "lr_scheduler_type": "cosine",
                "logging_steps": 10,
                "save_steps": 500,
                "eval_steps": 500,
                "save_total_limit": 2,
                "load_best_model_at_end": True,
                "metric_for_best_model": "loss",
                "greater_is_better": False,
                "report_to": []
            }
            
            # Merge with defaults
            for key, value in default_config.items():
                config.setdefault(key, value)
            
            return config
            
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise ValueError(f"Error parsing configuration file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading configuration: {e}")

    def _apply_dgx_optimizations(self):
        """Apply DGX B200 specific optimizations based on testing results."""
        # Set environment variables for optimal DGX B200 performance
        os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
        os.environ["NCCL_IB_DISABLE"] = "0"
        os.environ["NCCL_IB_GID_INDEX"] = "3"
        os.environ["NCCL_NET_GDR_LEVEL"] = "3"
        os.environ["NCCL_IB_QPS_PER_CONNECTION"] = "4"
        os.environ["NCCL_IB_TC"] = "136"
        
        # Memory optimization for large models
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        
        # Optimize for BF16 training
        if self.config.get("mixed_precision") == "bf16":
            os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
        
        logger.info("Applied DGX B200 optimizations")

    def _setup_logging(self):
        """Configure comprehensive logging for training process."""
        log_level = getattr(logging, self.config.get("log_level", "INFO").upper())
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Setup root logger
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Setup file logging if specified
        if self.config.get("log_file"):
            file_handler = logging.FileHandler(self.config["log_file"])
            file_handler.setFormatter(formatter)
            logging.getLogger().addHandler(file_handler)
        
        logger.info(f"Logging configured at {logging.getLevelName(log_level)} level")

    def _setup_output_dir(self):
        """Create output directory structure and save configuration."""
        self.output_dir = Path(self.config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "configs").mkdir(exist_ok=True)
        
        # Save configuration in both formats
        config_json_path = self.output_dir / "configs" / "training_config.json"
        config_yaml_path = self.output_dir / "configs" / "training_config.yaml"
        
        with open(config_json_path, "w", encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        with open(config_yaml_path, "w", encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Output directory created: {self.output_dir}")
        logger.info(f"Configuration saved to: {config_json_path} and {config_yaml_path}")

    def _load_tokenizer(self):
        """Load and configure tokenizer with optimal settings."""
        logger.info(f"Loading tokenizer for model: {self.config['model_name']}")
        
        tokenizer_kwargs = {
            "trust_remote_code": True,
            "use_fast": self.config.get("use_fast_tokenizer", True),
            "padding_side": self.config.get("padding_side", "right")
        }
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config['model_name'], 
                **tokenizer_kwargs
            )
        except Exception as e:
            logger.warning(f"Failed to load fast tokenizer, falling back to slow: {e}")
            tokenizer_kwargs["use_fast"] = False
            tokenizer = AutoTokenizer.from_pretrained(
                self.config['model_name'], 
                **tokenizer_kwargs
            )
        
        # Configure special tokens
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                logger.info("Added [PAD] token to tokenizer")
        
        # Set padding side for generation tasks
        tokenizer.padding_side = self.config.get("padding_side", "right")
        
        logger.info(f"Tokenizer loaded successfully. Vocab size: {len(tokenizer)}")
        logger.info(f"Special tokens - PAD: {tokenizer.pad_token}, EOS: {tokenizer.eos_token}, BOS: {tokenizer.bos_token}")
        
        return tokenizer

    def _load_model(self):
        """Load model with advanced optimization and DGX B200 specific settings."""
        logger.info(f"Loading model: {self.config['model_name']}")
        
        # Determine optimal data type
        if self.config.get('mixed_precision') == 'bf16':
            torch_dtype = torch.bfloat16
        elif self.config.get('mixed_precision') == 'fp16':
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
            "device_map": None,  # Let trainer handle device placement
        }
        
        # Enable Flash Attention 2 for better performance on H100/B200
        if self.config.get("use_flash_attention_2", True):
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Enabled Flash Attention 2")
            except Exception as e:
                logger.warning(f"Flash Attention 2 not available: {e}")
        
        # Load model with error handling
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.config['model_name'], 
                **model_kwargs
            )
        except Exception as e:
            logger.error(f"Failed to load model with Flash Attention 2: {e}")
            # Fallback without Flash Attention 2
            model_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(
                self.config['model_name'], 
                **model_kwargs
            )
        
        # Configure gradient checkpointing for memory efficiency
        if self.config.get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable()
            if hasattr(model.config, 'use_cache'):
                model.config.use_cache = False
            logger.info("Enabled gradient checkpointing")
        
        # Apply LoRA if configured
        if self.config.get("use_lora", False):
            model = self._apply_lora(model)
        
        # Enable torch.compile for better performance (experimental)
        if self.config.get("torch_compile", False):
            try:
                model = torch.compile(model)
                logger.info("Enabled torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile not available: {e}")
        
        # Resize token embeddings if tokenizer was modified
        if len(self.tokenizer) != model.config.vocab_size:
            model.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"Resized token embeddings to {len(self.tokenizer)}")
        
        logger.info(f"Model loaded successfully. Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model

    def _apply_lora(self, model):
        """Apply LoRA (Low-Rank Adaptation) for efficient fine-tuning."""
        logger.info("Applying LoRA configuration")
        
        # Determine target modules based on model architecture
        target_modules = self.config.get("lora_target_modules")
        if target_modules is None:
            # Default target modules for common architectures
            if "llama" in self.config['model_name'].lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            elif "qwen" in self.config['model_name'].lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            else:
                # Generic transformer modules
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.get("lora_r", 16),
            lora_alpha=self.config.get("lora_alpha", 32),
            lora_dropout=self.config.get("lora_dropout", 0.1),
            target_modules=target_modules,
            bias=self.config.get("lora_bias", "none"),
            use_rslora=self.config.get("use_rslora", False),
            modules_to_save=self.config.get("lora_modules_to_save", None)
        )
        
        try:
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            
            # Log LoRA configuration
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"LoRA applied successfully")
            logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
            logger.info(f"Target modules: {target_modules}")
            
        except Exception as e:
            logger.error(f"Failed to apply LoRA: {e}")
            raise
        
        return model

    def _get_device(self):
        """Get the appropriate device for training."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            logger.warning("CUDA not available, using CPU")
        
        return device

    def prepare_dataset(self):
        """Load and preprocess dataset with advanced formatting and optimization."""
        dataset_name = self.config['dataset_name']
        logger.info(f"Loading and preparing dataset: {dataset_name}")
        
        try:
            # Load dataset with specified split
            dataset_config = self.config.get('dataset_config', None)
            dataset_split = self.config.get('dataset_split', 'train')
            
            if dataset_config:
                dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
            else:
                dataset = load_dataset(dataset_name, split=dataset_split)
            
            logger.info(f"Dataset loaded successfully. Size: {len(dataset):,} examples")
            
            # Apply dataset filtering if specified
            if self.config.get('max_dataset_size'):
                max_size = self.config['max_dataset_size']
                if len(dataset) > max_size:
                    dataset = dataset.select(range(max_size))
                    logger.info(f"Dataset truncated to {max_size:,} examples")
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
        
        def _preprocess(examples):
            """Preprocess examples with flexible formatting."""
            prompts = []
            
            # Get formatting template
            prompt_template = self.config.get('prompt_template', 'alpaca')
            
            for i in range(len(examples["instruction"])):
                instruction = examples["instruction"][i]
                input_text = examples.get("input", [""] * len(examples["instruction"]))[i] or ""
                output = examples["output"][i]
                
                # Format based on template
                if prompt_template == 'alpaca':
                    if input_text.strip():
                        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                    else:
                        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                elif prompt_template == 'vicuna':
                    if input_text.strip():
                        prompt = f"USER: {instruction}\n{input_text}\nASSISTANT: {output}"
                    else:
                        prompt = f"USER: {instruction}\nASSISTANT: {output}"
                elif prompt_template == 'chatml':
                    if input_text.strip():
                        prompt = f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
                    else:
                        prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
                else:
                    # Custom template or fallback
                    prompt = f"{instruction} {input_text} {output}".strip()
                
                prompts.append(prompt)
            
            # Tokenize with optimized settings
            tokenized = self.tokenizer(
                prompts,
                truncation=True,
                max_length=self.config['max_length'],
                padding="max_length",  # Ensure consistent lengths for batching
                return_tensors=None,
                add_special_tokens=True
            )
            
            # Create labels (copy of input_ids for causal LM)
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        # Process dataset with multiprocessing (disable if CUDA is already initialized)
        num_proc = min(os.cpu_count(), self.config.get('dataset_num_proc', os.cpu_count()))
        
        # Disable multiprocessing if CUDA is already initialized to avoid forking issues
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            num_proc = 1
            logger.info("CUDA already initialized, disabling multiprocessing for dataset preprocessing")
        
        try:
            processed_dataset = dataset.map(
                _preprocess,
                batched=True,
                remove_columns=dataset.column_names,
                num_proc=num_proc,
                desc="Preprocessing dataset"
            )
            
            logger.info(f"Dataset preprocessing completed with {num_proc} processes")
            logger.info(f"Processed dataset size: {len(processed_dataset):,} examples")
            
            # Log sample statistics
            if len(processed_dataset) > 0:
                sample_lengths = [len(item) for item in processed_dataset['input_ids'][:1000]]
                avg_length = sum(sample_lengths) / len(sample_lengths)
                max_length = max(sample_lengths)
                logger.info(f"Average sequence length: {avg_length:.1f}, Max: {max_length}")
            
            return processed_dataset
            
        except Exception as e:
            logger.error(f"Failed to preprocess dataset: {e}")
            raise

    def get_training_arguments(self) -> TrainingArguments:
        """Create optimized TrainingArguments for DGX B200."""
        return TrainingArguments(
            output_dir=str(self.output_dir / "checkpoints"),
            overwrite_output_dir=True,
            
            # Training hyperparameters
            num_train_epochs=self.config.get("num_epochs", 3),
            per_device_train_batch_size=self.config.get("batch_size_per_device", 1),
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 8),
            learning_rate=self.config.get("learning_rate", 2e-4),
            weight_decay=self.config.get("weight_decay", 0.1),
            adam_beta1=self.config.get("adam_beta1", 0.9),
            adam_beta2=self.config.get("adam_beta2", 0.95),
            adam_epsilon=self.config.get("adam_epsilon", 1e-8),
            max_grad_norm=self.config.get("max_grad_norm", 1.0),
            
            # Optimizer and scheduler
            optim=self.config.get("optim", "adamw_torch_fused"),
            lr_scheduler_type=self.config.get("lr_scheduler_type", "cosine"),
            warmup_ratio=self.config.get("warmup_ratio", 0.03),
            
            # Mixed precision
            bf16=self.config.get("mixed_precision") == "bf16",
            fp16=self.config.get("mixed_precision") == "fp16",
            
            # Logging and saving
            logging_steps=self.config.get("logging_steps", 10),
            save_steps=self.config.get("save_steps", 500),
            eval_steps=self.config.get("eval_steps", 500),
            save_total_limit=self.config.get("save_total_limit", 2),
            
            # Memory optimization
            dataloader_num_workers=self.config.get("dataloader_num_workers", 8),
            dataloader_pin_memory=self.config.get("dataloader_pin_memory", True),
            gradient_checkpointing=self.config.get("gradient_checkpointing", True),
            
            # Monitoring
            report_to=self.config.get("report_to", []),
            run_name=self.config.get("run_name"),
            
            # Resume training
            resume_from_checkpoint=self.config.get("resume_from_checkpoint"),
            
            # Evaluation (only enable if we have eval dataset)
            eval_strategy="no",  # Disable evaluation for now
            load_best_model_at_end=False,  # Disable since no evaluation
            
            # Advanced settings
            remove_unused_columns=False,
            label_names=["labels"],
        )

    def save_model(self, save_path: str, save_tokenizer: bool = True):
        """Save model and tokenizer with comprehensive metadata."""
        save_dir = self.output_dir / save_path
        save_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save model
            if hasattr(self.model, 'save_pretrained'):
                self.model.save_pretrained(
                    save_dir,
                    safe_serialization=True,
                    max_shard_size="5GB"
                )
                logger.info(f"Model saved using save_pretrained to {save_dir}")
            else:
                # Fallback for models without save_pretrained
                torch.save(self.model.state_dict(), save_dir / "pytorch_model.bin")
                logger.info(f"Model state dict saved to {save_dir}")
            
            # Save tokenizer
            if save_tokenizer:
                self.tokenizer.save_pretrained(save_dir)
                logger.info(f"Tokenizer saved to {save_dir}")
            
            # Save training metadata
            metadata = {
                "model_name": self.config['model_name'],
                "dataset_name": self.config['dataset_name'],
                "training_config": self.config,
                "pytorch_version": torch.__version__,
            }
            
            # Add model-specific info
            if hasattr(self.model, 'config'):
                metadata["model_config"] = self.model.config.to_dict()
            
            metadata_path = save_dir / "training_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"Training metadata saved to {metadata_path}")
            logger.info(f"Complete model checkpoint saved to {save_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint_dir = Path(checkpoint_path)
        
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            logger.info(f"Loading checkpoint from {checkpoint_dir}")
            
            # Load model
            if self.config.get("use_lora", False):
                # Load LoRA model
                self.model = PeftModel.from_pretrained(self.model, checkpoint_dir)
            else:
                # Load full model
                self.model = AutoModelForCausalLM.from_pretrained(
                    checkpoint_dir,
                    torch_dtype=torch.bfloat16 if self.config.get('mixed_precision') == 'bf16' else torch.float16,
                    trust_remote_code=True
                )
            
            logger.info(f"Model loaded successfully from {checkpoint_dir}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    @abstractmethod
    def train(self):
        """Abstract method for training, to be implemented by subclasses."""
        pass