#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if (($# > 0)); then
  SEEDS=("$@")
else
  SEEDS=(1 2 3 4 5 6 7 8 9 10)
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
elif command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
else
  echo "Could not detect GPUs. Set CUDA_VISIBLE_DEVICES or install nvidia-smi." >&2
  exit 1
fi

GPU_COUNT="${#GPU_IDS[@]}"
if ((GPU_COUNT == 0)); then
  echo "No GPUs detected." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON:-python}"
WANDB_TAGS="${WANDB_TAGS:-[loqa,selfplay]}"
PIDS=()

wait_for_batch() {
  local failed=0
  local pid
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  PIDS=()
  if ((failed)); then
    echo "At least one seed failed." >&2
    exit 1
  fi
}

echo "Detected ${GPU_COUNT} GPU(s): ${GPU_IDS[*]}"
echo "Running ${#SEEDS[@]} LOQA self-play seed(s): ${SEEDS[*]}"

for i in "${!SEEDS[@]}"; do
  slot=$((i % GPU_COUNT))
  if ((i > 0 && slot == 0)); then
    wait_for_batch
  fi

  seed="${SEEDS[$i]}"
  gpu="${GPU_IDS[$slot]}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
    echo "Starting seed ${seed} on GPU ${gpu}"
    "${PYTHON_BIN}" coin_train.py \
      hp=loqa \
      wandb.state=enabled \
      "wandb.tags=${WANDB_TAGS}" \
      "hp.seed=${seed}"
  ) &
  PIDS+=("$!")
done

wait_for_batch
echo "All LOQA self-play seeds finished."
