
set -euo pipefail
 
CONFIG=${1:-training/config/qwen_train.yaml}
 
# Detect GPU count from nvidia-smi
if command -v nvidia-smi &>/dev/null; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
else
    echo "WARNING: nvidia-smi not found, defaulting to 1 GPU"
    NUM_GPUS=1
fi
 
# Read which backend is set in the config (for the log message only)
BACKEND=$(python3 -c "
import yaml, sys
cfg = yaml.safe_load(open('$CONFIG'))
print(cfg.get('hardware', {}).get('distributed_backend', 'ddp'))
" 2>/dev/null || echo "ddp")
 
ZERO_STAGE=$(python3 -c "
import yaml, sys
cfg = yaml.safe_load(open('$CONFIG'))
print(cfg.get('hardware', {}).get('zero_stage', 2))
" 2>/dev/null || echo "2")
 
echo "──────────────────────────────────────────"
echo " Backend  : $BACKEND"
if [ "$BACKEND" = "deepspeed" ]; then
    echo " ZeRO     : stage $ZERO_STAGE"
fi
echo " GPUs     : $NUM_GPUS"
echo " Config   : $CONFIG"
echo "──────────────────────────────────────────"
 
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29501 \
    training/train.py \
    --config "$CONFIG"
 