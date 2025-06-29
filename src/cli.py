import os
import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

try:
    # Try relative imports first (works when run as module: -m src.cli)
    from trainers.single_gpu_trainer import SingleGPUTrainer
    from trainers.multi_gpu_trainer import MultiGPUTrainer
    from trainers.multi_node_trainer import MultiNodeTrainer
    from utils.monitoring import check_gpu_status, get_training_stats
except ImportError:
    # Fall back to absolute imports (works when run as script: src/cli.py)
    from src.trainers.single_gpu_trainer import SingleGPUTrainer
    from src.trainers.multi_gpu_trainer import MultiGPUTrainer
    from src.trainers.multi_node_trainer import MultiNodeTrainer
    from src.utils.monitoring import check_gpu_status, get_training_stats

app = typer.Typer(
    help="🚀 A modern GPU parallel processing LLM fine-tuning tool with multi-node DGX B200 support.",
    add_completion=False
)

# Global option for DeepSpeed local_rank
@app.callback()
def main(
    local_rank: Optional[int] = typer.Option(
        None, 
        "--local_rank", 
        help="Local rank for distributed training (set by DeepSpeed launcher)"
    )
):
    """Main callback to handle global options."""
    if local_rank is not None:
        import os
        os.environ["LOCAL_RANK"] = str(local_rank)
console = Console()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from JSON or YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        console.print(f"❌ Configuration file not found: {config_path}", style="bold red")
        raise typer.Exit(1)
    
    try:
        with open(config_file, 'r') as f:
            if config_file.suffix == '.json':
                return json.load(f)
            else:
                import yaml
                return yaml.safe_load(f)
    except Exception as e:
        console.print(f"❌ Error loading config: {e}", style="bold red")
        raise typer.Exit(1)


@app.command()
def train(
    mode: str = typer.Option(
        "single", 
        help="Training mode: single, multi, deepspeed, or multinode"
    ),
    config: str = typer.Option(
        "configs/training_config.json", 
        help="Path to training configuration file (JSON or YAML)"
    ),
    output_dir: Optional[str] = typer.Option(
        None, 
        help="Override output directory from config"
    ),
    model_name: Optional[str] = typer.Option(
        None, 
        help="Override model name from config"
    ),
    num_epochs: Optional[int] = typer.Option(
        None, 
        help="Override number of epochs from config"
    ),
    batch_size: Optional[int] = typer.Option(
        None, 
        help="Override batch size per device from config"
    ),
    learning_rate: Optional[float] = typer.Option(
        None, 
        help="Override learning rate from config"
    ),
    resume_from_checkpoint: Optional[str] = typer.Option(
        None, 
        help="Resume training from checkpoint directory"
    ),
    dry_run: bool = typer.Option(
        False, 
        help="Validate configuration without starting training"
    )
):
    """Fine-tune a language model using GPU parallel processing."""
    
    # Load and validate configuration
    config_data = load_config(config)
    
    # Override config with CLI arguments
    if output_dir:
        config_data["output_dir"] = output_dir
    if model_name:
        config_data["model_name"] = model_name
    if num_epochs:
        config_data["num_epochs"] = num_epochs
    if batch_size:
        config_data["batch_size_per_device"] = batch_size
    if learning_rate:
        config_data["learning_rate"] = learning_rate
    if resume_from_checkpoint:
        config_data["resume_from_checkpoint"] = resume_from_checkpoint
    
    # Display training configuration
    console.print(f"\n🚀 Starting training in [bold green]{mode}[/bold green] mode...")
    
    config_table = Table(title="Training Configuration")
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="white")
    
    key_configs = [
        ("Model", config_data.get("model_name", "Not specified")),
        ("Dataset", config_data.get("dataset_name", "Not specified")),
        ("Mode", mode),
        ("Epochs", str(config_data.get("num_epochs", "Not specified"))),
        ("Batch Size", str(config_data.get("batch_size_per_device", "Not specified"))),
        ("Learning Rate", str(config_data.get("learning_rate", "Not specified"))),
        ("Output Dir", config_data.get("output_dir", "Not specified")),
        ("Use LoRA", str(config_data.get("use_lora", False))),
    ]
    
    for param, value in key_configs:
        config_table.add_row(param, value)
    
    console.print(config_table)
    
    if dry_run:
        console.print("\n✅ Configuration validation completed (dry run mode)", style="bold green")
        return
    
    # Save updated config
    config_path = Path(config)
    temp_config_path = config_path.parent / f"temp_{config_path.name}"
    with open(temp_config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    try:
        # Initialize trainer based on mode
        if mode == "single":
            trainer = SingleGPUTrainer(str(temp_config_path))
        elif mode == "multi":
            trainer = MultiGPUTrainer(str(temp_config_path))
        elif mode in ["deepspeed", "multinode"]:
            trainer = MultiNodeTrainer(str(temp_config_path))
        else:
            console.print(
                f"❌ Invalid training mode '{mode}'. Choose from: single, multi, deepspeed, multinode", 
                style="bold red"
            )
            raise typer.Exit(1)
        
        # Start training
        trainer.train()
        console.print("\n✅ Training completed successfully!", style="bold green")
        
    except KeyboardInterrupt:
        console.print("\n⚠️  Training interrupted by user", style="bold yellow")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n❌ Training failed: {e}", style="bold red")
        logger.exception("Training error details:")
        raise typer.Exit(1)
    finally:
        # Clean up temporary config
        if temp_config_path.exists():
            temp_config_path.unlink()


@app.command()
def evaluate(
    model_path: str = typer.Argument(help="Path to the trained model"),
    config: str = typer.Option(
        "configs/training_config.json", 
        help="Path to training configuration file"
    ),
    output_dir: str = typer.Option(
        "./evaluation_results", 
        help="Directory to save evaluation results"
    ),
    batch_size: int = typer.Option(
        8, 
        help="Batch size for evaluation"
    )
):
    """Evaluate a trained model on test data."""
    console.print(f"🔍 Starting model evaluation...")
    console.print(f"Model: {model_path}")
    console.print(f"Output: {output_dir}")
    
    # TODO: Implement evaluation logic
    console.print("⚠️  Evaluation functionality coming soon!", style="bold yellow")


@app.command()
def monitor(
    output_dir: str = typer.Option(
        "./outputs", 
        help="Training output directory to monitor"
    ),
    refresh_interval: int = typer.Option(
        5, 
        help="Refresh interval in seconds"
    )
):
    """Monitor training progress and GPU usage."""
    console.print("📊 Starting training monitoring...")
    
    try:
        # Check GPU status
        gpu_info = check_gpu_status()
        if gpu_info:
            console.print(Panel(gpu_info, title="GPU Status", border_style="green"))
        
        # Get training stats if available
        stats = get_training_stats(output_dir)
        if stats:
            console.print(Panel(stats, title="Training Statistics", border_style="blue"))
        else:
            console.print(f"📁 Monitoring directory: {output_dir}")
            console.print("ℹ️  No training logs found yet")
            
    except Exception as e:
        console.print(f"❌ Monitoring error: {e}", style="bold red")


@app.command()
def info():
    """Display system information and available configurations."""
    console.print("🖥️  System Information", style="bold blue")
    
    # System info table
    info_table = Table()
    info_table.add_column("Component", style="cyan")
    info_table.add_column("Status", style="white")
    
    # Check Python version
    import sys
    info_table.add_row("Python Version", f"{sys.version.split()[0]}")
    
    # Check PyTorch
    try:
        import torch
        info_table.add_row("PyTorch Version", torch.__version__)
        info_table.add_row("CUDA Available", str(torch.cuda.is_available()))
        if torch.cuda.is_available():
            info_table.add_row("CUDA Version", torch.version.cuda)
            info_table.add_row("GPU Count", str(torch.cuda.device_count()))
    except ImportError:
        info_table.add_row("PyTorch", "Not installed")
    
    # Check other packages
    try:
        import transformers
        info_table.add_row("Transformers", transformers.__version__)
    except ImportError:
        info_table.add_row("Transformers", "Not installed")
    
    try:
        import accelerate
        info_table.add_row("Accelerate", accelerate.__version__)
    except ImportError:
        info_table.add_row("Accelerate", "Not installed")
    
    try:
        import deepspeed
        info_table.add_row("DeepSpeed", deepspeed.__version__)
    except ImportError:
        info_table.add_row("DeepSpeed", "Not installed")
    
    console.print(info_table)
    
    # Available configurations
    console.print("\n📁 Available Configurations", style="bold blue")
    config_dir = Path("configs")
    if config_dir.exists():
        config_files = list(config_dir.glob("*.json")) + list(config_dir.glob("*.yaml"))
        for config_file in sorted(config_files):
            console.print(f"  📄 {config_file.name}")
    else:
        console.print("  ⚠️  No configs directory found")


if __name__ == "__main__":
    app()