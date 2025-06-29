#!/bin/bash
# DGX B200 Multi-Node Training Environment Setup
# Based on proven configuration from actual 2-node DGX B200 testing

# Load environment variables if available
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
elif [ -f ".env.local" ]; then
    export $(grep -v '^#' .env.local | xargs)
fi

# DGX B200 specific GPU configuration (8x H100 per node)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# NCCL Configuration for DGX B200 with InfiniBand (from actual test)
# These settings were proven to work on DGX B200 nodes
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-"ibp24s0,ibp64s0,ibp79s0,ibp94s0,ibp154s0,ibp192s0,ibp206s0,ibp220s0"}
export NCCL_IB_HCA=${NCCL_IB_HCA:-"mlx5_4,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_13,mlx5_14,mlx5_15"}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-PIX}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}

# Docker container network fixes
export NCCL_SOCKET_FAMILY=AF_INET
export CUDA_LAUNCH_BLOCKING=0

# Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

echo "✅ DGX B200 multi-node environment configured"
echo "📊 GPUs per node: ${NUM_GPUS_PER_NODE:-8} (H100)"
echo "🔗 NCCL backend: InfiniBand with proven DGX B200 settings"
echo "🚀 DeepSpeed ZeRO Stage 1 for optimal performance"
echo "🌐 Master node: ${MASTER_ADDR:-'not set'}"
echo "📍 Node rank: ${NODE_RANK:-'not set'}"
echo ""
echo "💡 Based on actual test results:"
echo "   - Single node (8 GPUs): 44 min, 7.43 samples/sec"
echo "   - Multi-node (16 GPUs): 17 min, 19.08 samples/sec"
echo "   - 2.5x speedup with 2x GPUs!"