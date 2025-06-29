# Basic configuration
python_version := "3.11"
project_name := "gpu-distributed-training"

# Initial project setup
setup:
    #!/usr/bin/env bash
    echo "🚀 Setting up GPU parallel/distributed processing training project..."
    
    # Check uv installation
    if ! command -v uv &> /dev/null; then
        echo "❌ uv is not installed. Proceeding with installation..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source $HOME/.cargo/env
    fi
    
    # Setup Python environment
    uv python install {{python_version}}
    uv sync --all-extras
    
    # Create configuration files
    just create-configs
    
    echo "✅ Setup completed!"

# Install/update dependencies
install:
    uv sync --all-extras
    uv pip install --no-build-isolation flash-attn==2.8.0.post2

install-dev:
    uv sync --all-extras

# Create configuration files
create-configs:
    #!/usr/bin/env bash
    echo "📁 Creating configuration files..."
    mkdir -p configs scripts examples
    
    # Create Accelerate configuration files
    just create-accelerate-configs
    just create-deepspeed-config
    just create-training-config
    just create-scripts
    
    echo "✅ Configuration files created!"

# Create Accelerate configuration files
create-accelerate-configs:
    #!/usr/bin/env bash
    
    # Single machine multi-GPU configuration
    cat > configs/accelerate_single.yaml << 'EOF'
    compute_environment: LOCAL_MACHINE
    distributed_type: MULTI_GPU
    downcast_bf16: 'no'
    gpu_ids: all
    machine_rank: 0
    main_training_function: main
    mixed_precision: fp16
    num_machines: 1
    num_processes: 8  # Adjust according to the number of GPUs
    rdzv_backend: static
    same_network: true
    use_cpu: false
    EOF
    
    # Multi-node configuration
    cat > configs/accelerate_multi.yaml << 'EOF'
    compute_environment: LOCAL_MACHINE
    deepspeed_config:
      deepspeed_config_file: configs/deepspeed_config.json
      zero3_init_flag: true
      zero3_save_16bit_model: true
    distributed_type: DEEPSPEED
    downcast_bf16: 'no'
    machine_rank: 0
    main_process_ip: localhost
    main_process_port: 29500
    main_training_function: main
    mixed_precision: fp16
    num_machines: 1
    num_processes: 8
    rdzv_backend: static
    same_network: true
    use_cpu: false
    EOF

# Create DeepSpeed configuration
create-deepspeed-config:
    #!/usr/bin/env bash
    cat > configs/deepspeed_config.json << 'EOF'
    {
    "train_batch_size": "auto",
    "gradient_accumulation_steps": "auto",
    "optimizer": {
        "type": "AdamW",
        "params": {
        "lr": "auto",
        "betas": "auto",
        "eps": 1e-8,
        "weight_decay": "auto"
        }
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
        "device": "cpu",
        "pin_memory": true
        },
        "offload_param": {
        "device": "cpu",
        "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "gather_16bit_weights_on_model_save": true
    },
    "gradient_clipping": 1.0,
    "bf16": {
            "enabled": true
        },
    "activation_checkpointing": {
        "partition_activations": true,
        "cpu_checkpointing": true,
        "number_checkpoints": 4
    },
    "wall_clock_breakdown": false
    }
    EOF

# Create training configuration
create-training-config:
    #!/usr/bin/env bash
    cat > configs/training_config.json << 'EOF'
    {
      "model_name": "Qwen/Qwen3-1.7B",
      "dataset_name": "tatsu-lab/alpaca",
      "max_length": 2048,
      "batch_size_per_device": 1,
      "gradient_accumulation_steps": 8,
      "learning_rate": 2e-4,
      "num_epochs": 3,
      "warmup_steps": 100,
      "save_steps": 500,
      "logging_steps": 10,
      "output_dir": "./outputs",
      "use_lora": true,
      "lora_r": 16,
      "lora_alpha": 32,
      "lora_dropout": 0.1,
      "use_wandb": false,
      "wandb_project": "llm-finetuning",
      "deepspeed_config": "configs/deepspeed_config.json"
    }
    EOF

# Create execution scripts
create-scripts:
    #!/usr/bin/env bash
    
    # Environment setup script
    cat > scripts/setup_environment.sh << 'EOF'
    #!/bin/bash
    # GPU environment setup
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export NCCL_SOCKET_IFNAME=^docker0,lo
    export NCCL_IB_DISABLE=0
    export NCCL_DEBUG=INFO
    export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
    
    # Memory optimization
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
    export OMP_NUM_THREADS=8
    EOF
    
    chmod +x scripts/*.sh

# Development environment setup
dev-setup:
    uv sync --all-extras
    uv run pre-commit install

# Code formatting
format:
    uv run black src/
    uv run isort src/

# Code linting
lint:
    uv run flake8 src/
    uv run mypy src/

# Run tests
test:
    uv run pytest tests/ -v

# Check GPU status
check-gpu:
    #!/usr/bin/env bash
    echo "🔍 Checking GPU status..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi
        echo ""
        uv run gpustat
    else
        echo "❌ NVIDIA GPU not found."
    fi

# Single GPU training (auto-detects hardware)
train-single GPU="0":
    #!/usr/bin/env bash
    source scripts/setup_environment.sh
    export CUDA_VISIBLE_DEVICES={{GPU}}
    echo "🚀 Starting single GPU training (GPU: {{GPU}})"
    uv run python -m src.cli train --config configs/training_config.json --mode single

# RTX 5080 optimized training (16GB VRAM)
train-rtx5080 GPU="0":
    #!/usr/bin/env bash
    source scripts/setup_environment.sh
    export CUDA_VISIBLE_DEVICES={{GPU}}
    echo "🚀 Starting RTX 5080 optimized training (GPU: {{GPU}})"
    echo "💾 Memory: Conservative settings for 16GB VRAM"
    uv run python -m src.cli train --config configs/training_config_rtx5080.json --mode single

# RTX 5080 balanced training (sweet spot between performance and memory)
train-rtx5080-balanced GPU="0":
    #!/usr/bin/env bash
    source scripts/setup_environment.sh
    export CUDA_VISIBLE_DEVICES={{GPU}}
    echo "🚀 Starting RTX 5080 balanced training (GPU: {{GPU}})"
    echo "🎯 Target: 60-70% GPU utilization with memory safety margin"
    uv run python -m src.cli train --config configs/training_config_rtx5080_balanced.json --mode single

# RTX 5080 optimized training (higher GPU utilization - risky!)
train-rtx5080-optimized GPU="0":
    #!/usr/bin/env bash
    source scripts/setup_environment.sh
    export CUDA_VISIBLE_DEVICES={{GPU}}
    echo "🚀 Starting RTX 5080 optimized training (GPU: {{GPU}})"
    echo "⚡ WARNING: Aggressive settings - may OOM during backward pass!"
    uv run python -m src.cli train --config configs/training_config_rtx5080_optimized.json --mode single

# DGX B200 optimized training (80GB+ VRAM)
train-dgx-b200 GPU="0":
    #!/usr/bin/env bash
    source scripts/setup_environment.sh
    export CUDA_VISIBLE_DEVICES={{GPU}}
    echo "🚀 Starting DGX B200 optimized training (GPU: {{GPU}})"
    echo "⚡ Performance: High-throughput settings for 80GB+ VRAM"
    uv run python -m src.cli train --config configs/training_config_dgx_b200.json --mode single

# Multi-GPU training (single node)
train-multi:
    #!/usr/bin/env bash
    source scripts/setup_environment.sh
    echo "🚀 Starting multi-GPU training (single node)"
    uv run accelerate launch --config_file configs/accelerate_single.yaml \
        -m src.cli train --config configs/training_config.json --mode multi

# DeepSpeed training (memory optimized)
train-deepspeed:
    #!/usr/bin/env bash
    source scripts/setup_environment.sh
    echo "🚀 Starting DeepSpeed training (memory optimized)"
    uv run accelerate launch --config_file configs/accelerate_multi.yaml \
        -m src.cli train --config configs/dee.json --mode deepspeed

# DGX B200 Multi-node training (master node)
train-multinode-master:
    #!/usr/bin/env bash
    echo "🚀 Starting DGX B200 multi-node training (master node)"
    echo "📄 Reading configuration from .env file"
    if [ -f ".env" ]; then
        export $(grep -v '^#' .env | xargs)
    elif [ -f ".env.local" ]; then
        export $(grep -v '^#' .env.local | xargs)
    fi
    export NODE_RANK=0
    echo "📊 Nodes: ${NUM_NODES:-2}, Master: ${MASTER_ADDR:-'not set'}"
    scripts/run_multinode_dgx.sh

# DGX B200 Multi-node training (worker node)
train-multinode-worker NODE_RANK="1":
    #!/usr/bin/env bash
    echo "🚀 Starting DGX B200 multi-node training (worker node {{NODE_RANK}})"
    echo "📄 Reading configuration from .env file"
    if [ -f ".env" ]; then
        export $(grep -v '^#' .env | xargs)
    elif [ -f ".env.local" ]; then
        export $(grep -v '^#' .env.local | xargs)
    fi
    export NODE_RANK={{NODE_RANK}}
    echo "📊 Nodes: ${NUM_NODES:-2}, Master: ${MASTER_ADDR:-'not set'}"
    scripts/run_multinode_dgx.sh

# Training monitoring
monitor:
    #!/usr/bin/env bash
    echo "📊 Training monitoring..."
    uv run python -m src.utils.monitoring

# Model evaluation
evaluate MODEL_PATH OUTPUT_DIR:
    uv run python -m src.cli evaluate --model_path {{MODEL_PATH}} --output_dir {{OUTPUT_DIR}}

# Create sample data
create-sample-data:
    #!/usr/bin/env bash
    cat > examples/sample_data.json << 'EOF'
    [
      {
        "instruction": "Translate the following sentence into Korean.",
        "input": "Hello, how are you today?",
        "output": "Hello, how are you today?"
      },
      {
        "instruction": "Explain how to sort a list in Python.",
        "input": "",
        "output": "There are several ways to sort a list in Python. The simplest methods are using the sort() method or the sorted() function."
      }
    ]
    EOF
    echo "✅ Sample data created: examples/sample_data.json"

# Clean project
clean:
    #!/usr/bin/env bash
    echo "🧹 Cleaning project..."
    rm -rf __pycache__ .pytest_cache .mypy_cache
    find . -name "*.pyc" -delete
    find . -name "*.pyo" -delete
    echo "✅ Cleanup completed!"

# Complete cleanup (also delete output files)
clean-all: clean
    #!/usr/bin/env bash
    echo "🧹 Complete cleanup in progress..."
    rm -rf outputs/ wandb/ .venv/
    echo "✅ Complete cleanup finished!"

# Help
help:
    @echo "🎯 GPU Parallel Processing Training Project Commands:"
    @echo ""
    @echo "📦 Setup Related:"
    @echo "  setup              - Initial project setup"
    @echo "  install            - Install dependencies"
    @echo "  install-dev        - Install including dev dependencies"
    @echo "  create-configs     - Create configuration files"
    @echo ""
    @echo "🚀 Training Related:"
    @echo "  train-single [GPU]              - Single GPU training (auto-config)"
    @echo "  train-rtx5080 [GPU]             - RTX 5080 conservative (safe)"
    @echo "  train-rtx5080-balanced [GPU]    - RTX 5080 balanced (recommended)"
    @echo "  train-rtx5080-optimized [GPU]   - RTX 5080 aggressive (may OOM)"
    @echo "  train-dgx-b200 [GPU]            - DGX B200 optimized (80GB+ VRAM)"
    @echo "  train-multi                     - Multi-GPU training (single node)"
    @echo "  train-deepspeed                 - DeepSpeed training (memory optimized)"
    @echo "  train-multinode-master          - DGX B200 multi-node (master node)"
    @echo "  train-multinode-worker [RANK]   - DGX B200 multi-node (worker node)"
    @echo ""
    @echo "🔍 Monitoring:"
    @echo "  check-gpu          - Check GPU status"
    @echo "  monitor            - Training monitoring"
    @echo ""
    @echo "🛠️  Development Tools:"
    @echo "  dev-setup          - Development environment setup"
    @echo "  format             - Code formatting"
    @echo "  lint               - Code linting"
    @echo "  test               - Run tests"
    @echo ""
    @echo "🧹 Cleanup:"
    @echo "  clean              - Clean cache files"
    @echo "  clean-all          - Complete cleanup"

# Default command (when running 'just')
default: help
