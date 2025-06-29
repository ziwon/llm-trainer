#!/bin/bash
# DGX B200 Multi-Node Training Script
# Based on proven 2-node DGX B200 configuration

set -e

# Load environment variables
if [ -f ".env" ]; then
    echo "📄 Loading configuration from .env file"
    export $(grep -v '^#' .env | xargs)
elif [ -f ".env.local" ]; then
    echo "📄 Loading configuration from .env.local file"
    export $(grep -v '^#' .env.local | xargs)
else
    echo "⚠️  No .env file found, using environment variables or defaults"
fi

# Configuration with defaults from .env
NUM_NODES=${NUM_NODES:-2}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29500}
CONFIG_FILE=${TRAINING_CONFIG:-"configs/training_config.json"}
HOSTFILE=${HOSTFILE:-"configs/hostfile.txt"}

# NCCL settings with defaults from .env
NCCL_DEBUG=${NCCL_DEBUG:-INFO}
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-"ibp24s0,ibp64s0,ibp79s0,ibp94s0,ibp154s0,ibp192s0,ibp206s0,ibp220s0"}
NCCL_IB_HCA=${NCCL_IB_HCA:-"mlx5_4,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_13,mlx5_14,mlx5_15"}
NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-PIX}
NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}

# Validate inputs
if [ -z "$MASTER_ADDR" ]; then
    echo "❌ Error: MASTER_ADDR not set"
    echo "   Set in .env file or export MASTER_ADDR=<your_master_ip>"
    echo "   Example: MASTER_ADDR=100.40.31.8"
    exit 1
fi

if [ -z "$NODE_RANK" ]; then
    echo "❌ Error: NODE_RANK not set"
    echo "   Set in .env file or export NODE_RANK=<node_rank>"
    echo "   Master node: NODE_RANK=0"
    echo "   Worker nodes: NODE_RANK=1,2,3..."
    exit 1
fi

if [ ! -f "$HOSTFILE" ]; then
    echo "❌ Error: Hostfile not found: $HOSTFILE"
    echo "   Create hostfile based on configs/hostfile_template.txt"
    exit 1
fi

# Setup environment
source scripts/setup_4node_dgx.sh

# Generate accelerate config from environment variables
echo "🔧 Generating accelerate configuration..."
scripts/generate_accelerate_config.sh

echo ""
echo "🚀 Starting DGX B200 Multi-Node Training"
echo "📊 Configuration:"
echo "   Nodes: $NUM_NODES"
echo "   GPUs per node: $NUM_GPUS_PER_NODE"
echo "   Total GPUs: $((NUM_NODES * NUM_GPUS_PER_NODE))"
echo "   Master: $MASTER_ADDR:$MASTER_PORT"
echo "   Node rank: $NODE_RANK"
echo "   Config: $CONFIG_FILE"
echo "   Hostfile: $HOSTFILE"
echo ""

# Run training with environment-based DGX B200 settings
NCCL_DEBUG="$NCCL_DEBUG" \
NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
NCCL_IB_HCA="$NCCL_IB_HCA" \
NCCL_NET_GDR_LEVEL="$NCCL_NET_GDR_LEVEL" \
NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
uv run deepspeed \
  --num_nodes=$NUM_NODES \
  --num_gpus=$NUM_GPUS_PER_NODE \
  --node_rank=$NODE_RANK \
  --no_ssh \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  --hostfile=$HOSTFILE \
  src/cli.py train --config $CONFIG_FILE --mode deepspeed

echo "✅ Training completed!"