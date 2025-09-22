#!/bin/bash
export FASTVIDEO_ATTENTION_BACKEND=MONARCH_ATTN

# Configs
MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DATA_DIR=/checkpoint-fsx/beidchen-sandbox/video/wan-syn/test/
VALIDATION_DATASET_FILE=examples/training/finetune/Wan2.1-VSA/Wan-Syn-Data/validation_64.json
# VALIDATION_DATASET_FILE=examples/training/finetune/wan_t2v_1.3B/crush_smol/validation.json

# Training arguments
training_args=(
  --tracker_project_name fastwan
  --wandb_run_name wan_1.3b_t2v_monarch_max_constrain
  --output_dir "checkpoints/wan_1.3b_t2v_finetune_monarch_max_constrain"
  --max_train_steps 4000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 20
  --num_height 448
  --num_width 832
  --num_frames 77
  --enable_gradient_checkpointing_type "full" # if OOM enable this
)

# Parallel arguments
parallel_args=(
  --num_gpus 64
  --sp_size 1
  --tp_size 1
  --hsdp_replicate_dim 64
  --hsdp_shard_dim 1
)

# Model arguments
model_args=(
  --model_path $MODEL_PATH
  --pretrained_model_name_or_path $MODEL_PATH
)

# Dataset arguments
dataset_args=(
  --data_path "$DATA_DIR"
  --dataloader_num_workers 4
)

# Validation arguments
validation_args=(
  --log_validation
  --validation_dataset_file $VALIDATION_DATASET_FILE
  --validation_steps 200
  --validation_sampling_steps "50"
  --validation_guidance_scale "5.0"
)

# Optimizer arguments
optimizer_args=(
  --learning_rate 1e-6
  --mixed_precision "bf16"
  --checkpointing_steps 1000
  --weight_decay 0.01
  --max_grad_norm 1.0
)

# Miscellaneous arguments
miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 3
  --training_cfg_rate 0.1
  --dit_precision "fp32"
  --ema_start_step 0
  --flow_shift 1
  --seed 1000
)

# cp -r /checkpoint-fsx/beidchen-sandbox/video/hub /workspace
export HF_HOME="/workspace"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun \
--nproc_per_node 8 \
--rdzv-conf="timeout=3600,read_timeout=3600,join_timeout=3600" \
    fastvideo/training/wan_training_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}"
