import time
import psutil
import torch
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.live import Live

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

logger = logging.getLogger(__name__)

def check_gpu_status() -> Optional[str]:
    """Check and return GPU status information."""
    if not torch.cuda.is_available():
        return "❌ CUDA not available"
    
    try:
        gpu_count = torch.cuda.device_count()
        gpu_info = []
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            if NVML_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    
                    memory_used = mem_info.used / 1024**3  # GB
                    memory_total = mem_info.total / 1024**3  # GB
                    memory_percent = (mem_info.used / mem_info.total) * 100
                    
                    gpu_info.append(
                        f"GPU {i}: {gpu_name}\n"
                        f"  Memory: {memory_used:.1f}GB / {memory_total:.1f}GB ({memory_percent:.1f}%)\n"
                        f"  Utilization: {utilization.gpu}%"
                    )
                except Exception as e:
                    gpu_info.append(f"GPU {i}: {gpu_name} (monitoring error: {e})")
            else:
                gpu_info.append(f"GPU {i}: {gpu_name} (detailed monitoring unavailable)")
        
        return "\n".join(gpu_info)
        
    except Exception as e:
        return f"❌ Error checking GPU status: {e}"

def get_training_stats(output_dir: str) -> Optional[str]:
    """Get training statistics from logs and checkpoints."""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        return None
    
    stats = []
    
    # Check for training metrics
    metrics_file = output_path / "training_metrics.json"
    if metrics_file.exists():
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            stats.append("📊 Training Metrics:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    stats.append(f"  {key}: {value:.4f}")
                else:
                    stats.append(f"  {key}: {value}")
        except Exception as e:
            stats.append(f"❌ Error reading metrics: {e}")
    
    # Check for checkpoints
    checkpoints_dir = output_path / "checkpoints"
    if checkpoints_dir.exists():
        checkpoints = list(checkpoints_dir.glob("checkpoint-*"))
        if checkpoints:
            stats.append(f"\n💾 Checkpoints: {len(checkpoints)} found")
            latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
            stats.append(f"  Latest: {latest_checkpoint.name}")
    
    # Check for logs
    logs_dir = output_path / "logs"
    if logs_dir.exists():
        log_files = list(logs_dir.glob("*.log"))
        if log_files:
            stats.append(f"\n📝 Log files: {len(log_files)} found")
    
    return "\n".join(stats) if stats else None

class GPUMonitor:
    """GPU and system monitoring with English interface."""
    
    def __init__(self):
        self.console = Console()
        
    def get_gpu_info(self):
        """Collect GPU information."""
        gpu_info = []
        
        if torch.cuda.is_available() and NVML_AVAILABLE:
            for i in range(torch.cuda.device_count()):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode('utf-8')
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    
                    gpu_info.append({
                        'id': i,
                        'name': name,
                        'memory_used': mem_info.used / 1024**3,  # GB
                        'memory_total': mem_info.total / 1024**3,  # GB
                        'memory_percent': (mem_info.used / mem_info.total) * 100,
                        'gpu_utilization': utilization.gpu,
                        'memory_utilization': utilization.memory
                    })
                except Exception as e:
                    logger.warning(f"Error getting info for GPU {i}: {e}")
                    gpu_info.append({
                        'id': i,
                        'name': f"GPU {i} (error)",
                        'memory_used': 0,
                        'memory_total': 0,
                        'memory_percent': 0,
                        'gpu_utilization': 0,
                        'memory_utilization': 0
                    })
        
        return gpu_info
    
    def get_system_info(self):
        """Collect system information."""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent
            }
        except Exception as e:
            logger.warning(f"Error getting system info: {e}")
            return {
                'cpu_percent': 0,
                'memory_percent': 0,
                'disk_percent': 0
            }
    
    def create_status_table(self):
        """Create status table."""
        table = Table(title="System Monitoring")
        
        # GPU information
        gpu_info = self.get_gpu_info()
        if gpu_info:
            table.add_column("GPU ID", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Memory Usage", style="yellow")
            table.add_column("GPU Usage", style="red")
            
            for gpu in gpu_info:
                table.add_row(
                    str(gpu['id']),
                    gpu['name'][:25],
                    f"{gpu['memory_percent']:.1f}% ({gpu['memory_used']:.1f}GB/{gpu['memory_total']:.1f}GB)",
                    f"{gpu['gpu_utilization']}%"
                )
        else:
            table.add_column("Info", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("GPU", "Not available")
        
        # System information
        sys_info = self.get_system_info()
        table.add_row("", "")  # Separator
        table.add_row("CPU", f"{sys_info['cpu_percent']:.1f}%")
        table.add_row("Memory", f"{sys_info['memory_percent']:.1f}%")
        table.add_row("Disk", f"{sys_info['disk_percent']:.1f}%")
        
        return table
    
    def start_monitoring(self, interval: int = 2):
        """Start real-time monitoring."""
        try:
            with Live(self.create_status_table(), refresh_per_second=0.5) as live:
                while True:
                    time.sleep(interval)
                    live.update(self.create_status_table())
        except KeyboardInterrupt:
            self.console.print("\nMonitoring stopped.", style="bold red")

def get_gpu_utilization() -> Dict[str, Any]:
    """Get detailed GPU utilization metrics for training monitoring."""
    if not torch.cuda.is_available() or not NVML_AVAILABLE:
        return {}
    
    try:
        metrics = {}
        gpu_count = torch.cuda.device_count()
        
        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # Temperature and power info
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
            except:
                temp = None
                power = None
            
            metrics[f"gpu_{i}"] = {
                "name": torch.cuda.get_device_name(i),
                "memory_used_gb": mem_info.used / 1024**3,
                "memory_total_gb": mem_info.total / 1024**3,
                "memory_utilization_percent": (mem_info.used / mem_info.total) * 100,
                "gpu_utilization_percent": utilization.gpu,
                "memory_bandwidth_utilization_percent": utilization.memory,
                "temperature_c": temp,
                "power_usage_w": power
            }
        
        return metrics
        
    except Exception as e:
        logger.warning(f"Error getting GPU utilization: {e}")
        return {}

class GPUTrainingMonitor:
    """Real-time GPU monitoring during training."""
    
    def __init__(self, log_interval_steps: int = 50):
        self.log_interval_steps = log_interval_steps
        self.metrics_history = []
        self.start_time = time.time()
        
    def log_metrics(self, step: int, loss: float, learning_rate: float, epoch: int):
        """Log training metrics with GPU utilization."""
        if step % self.log_interval_steps != 0:
            return
            
        # Get GPU metrics
        gpu_metrics = get_gpu_utilization()
        current_time = time.time()
        
        # Compile metrics
        metrics = {
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "learning_rate": learning_rate,
            "timestamp": current_time,
            "elapsed_time_min": (current_time - self.start_time) / 60,
            "gpu_metrics": gpu_metrics
        }
        
        self.metrics_history.append(metrics)
        
        # Log to console
        if gpu_metrics:
            primary_gpu = list(gpu_metrics.keys())[0]
            gpu_util = gpu_metrics[primary_gpu]["gpu_utilization_percent"]
            mem_util = gpu_metrics[primary_gpu]["memory_utilization_percent"]
            temp = gpu_metrics[primary_gpu]["temperature_c"]
            power = gpu_metrics[primary_gpu]["power_usage_w"]
            
            logger.info(
                f"Step {step:,} | Loss: {loss:.4f} | LR: {learning_rate:.2e} | "
                f"GPU: {gpu_util}% | Mem: {mem_util:.1f}% | Temp: {temp}°C | Power: {power:.0f}W"
            )
        else:
            logger.info(f"Step {step:,} | Loss: {loss:.4f} | LR: {learning_rate:.2e}")
    
    def save_metrics(self, output_path: Path):
        """Save metrics history to file."""
        metrics_file = output_path / "gpu_training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        logger.info(f"GPU training metrics saved to: {metrics_file}")

# For backwards compatibility and CLI usage
if __name__ == "__main__":
    monitor = GPUMonitor()
    monitor.start_monitoring()