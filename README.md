# GPU Parallel/Distributed Processing LLM Fine-tuning Project

A modern GPU parallel processing LLM fine-tuning tool using uv and Justfile.

## 🚀 Quick Start

### 1. Project Setup
```bash
# Install Just (if needed)
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/.local/bin

# Project setup
just setup
```

### 2. Training Execution
```bash
# Single GPU training
just train-single

# Multi-GPU training (single node)
just train-multi

# DeepSpeed training (memory optimized)
just train-deepspeed

# DGX B200 Multi-node training (configure .env first)
just train-multinode-master          # Master node
just train-multinode-worker 1        # Worker node (rank 1)
just train-multinode-worker 2        # Worker node (rank 2)
```

### 3. Multi-Node Configuration
```bash
# Copy and customize environment file
cp .env .env.local

# Edit .env.local with your network settings
# Set MASTER_ADDR to your master node's InfiniBand IP
# Create hostfile with your node hostnames
cp configs/hostfile_template.txt configs/hostfile.txt
```

## 📁 Project Structure

```
gpu-distributed-training/
├── configs/           # Configuration files
├── src/              # Source code
├── scripts/          # Execution scripts
├── examples/         # Example data
├── Justfile          # Command management
└── pyproject.toml    # Project configuration
```

## 🛠️ Key Features

- **Various parallel processing support**: Single GPU, multi-GPU, multi-node
- **Memory optimization**: DeepSpeed ZeRO, LoRA, Gradient Checkpointing
- **Monitoring**: WandB, TensorBoard, real-time GPU monitoring
- **Flexible configuration**: JSON-based configuration, command-line interface
- **DGX B200 optimized**: Proven multi-node configuration for DGX B200 nodes

## 📊 Performance

**Configuration:**
- **DeepSpeed ZeRO Stage 1** for optimal performance
- **Flash Attention 2** with BF16 precision
- **InfiniBand networking** with proven NCCL settings