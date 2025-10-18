DATA_DIR=/checkpoint-fsx/beidchen-sandbox/video/wan-syn/test/
VALIDATION_DIR=examples/training/finetune/Wan2.1-VSA/Wan-Syn-Data/validation_64.json
NUM_GPUS=8
export FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN
export TOKENIZERS_PARALLELISM=false

export HF_HOME="/workspace"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Train generator with VSA
# Make sure that num_latent_t is a multiple of sp_size
torchrun \
--nproc_per_node 8 \
--rdzv-conf="timeout=7200,read_timeout=7200,join_timeout=7200" \
    fastvideo/training/wan_distillation_pipeline.py \
    --model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --inference_mode False\
    --wandb_run_name wan_1.3b_t2v_vsa_dmd\
    --pretrained_model_name_or_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --data_path "$DATA_DIR" \
    --validation_dataset_file  "$VALIDATION_DIR" \
    --train_batch_size 1 \
    --num_latent_t 20 \
    --sp_size 1 \
    --tp_size 1 \
    --num_gpus 64 \
    --hsdp_replicate_dim 64 \
    --hsdp-shard-dim 1 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 1 \
    --max_train_steps 4000 \
    --learning_rate 1e-5 \
    --mixed_precision "bf16" \
    --checkpointing_steps 400 \
    --validation_steps 100 \
    --validation_sampling_steps "3" \
    --log_validation \
    --checkpoints_total_limit 3 \
    --ema_start_step 0 \
    --training_cfg_rate 0.0 \
    --output_dir "checkpoints/wan_1.3b_t2v_vsa_dmd" \
    --tracker_project_name fastwan \
    --num_height 448 \
    --num_width 832 \
    --num_frames 77 \
    --flow_shift 8 \
    --validation_guidance_scale "6.0" \
    --master_weight_type "fp32" \
    --dit_precision "fp32" \
    --vae_precision "bf16" \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --generator_update_interval 5 \
    --dmd_denoising_steps '1000,757,522' \
    --min_timestep_ratio 0.02 \
    --max_timestep_ratio 0.98 \
    --real_score_guidance_scale 3.5 \
    --seed 1024 \
    --VSA_decay_rate 0.03 \
    --VSA_decay_interval_steps 50 \
    --VSA_sparsity 0.95 \
    --enable_gradient_checkpointing_type "full" \
    --training_state_checkpointing_steps 500 \
    --weight_only_checkpointing_steps 200

