#!/bin/bash
# Generate Accelerate configuration from environment variables

# Load environment variables
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
elif [ -f ".env.local" ]; then
    export $(grep -v '^#' .env.local | xargs)
fi

# Set defaults
NUM_NODES=${NUM_NODES:-2}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29500}
NODE_RANK=${NODE_RANK:-0}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"configs/deepspeed_4node_dgx.json"}

# Calculate total processes
TOTAL_PROCESSES=$((NUM_NODES * NUM_GPUS_PER_NODE))

# Generate accelerate config
cat > configs/accelerate_multinode_generated.yaml << EOF
compute_environment: LOCAL_MACHINE
deepspeed_config:
  deepspeed_config_file: ${DEEPSPEED_CONFIG}
  zero3_init_flag: true
  zero3_save_16bit_model: true
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: ${NODE_RANK}
main_process_ip: ${MASTER_ADDR}
main_process_port: ${MASTER_PORT}
main_training_function: main
mixed_precision: bf16
num_machines: ${NUM_NODES}
num_processes: ${TOTAL_PROCESSES}
rdzv_backend: static
same_network: true
use_cpu: false
EOF

echo "✅ Generated accelerate config: configs/accelerate_multinode_generated.yaml"
echo "📊 Configuration:"
echo "   Nodes: ${NUM_NODES}"
echo "   GPUs per node: ${NUM_GPUS_PER_NODE}"
echo "   Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "   Node rank: ${NODE_RANK}"