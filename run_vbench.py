import os
import wandb
from dataclasses import asdict
import torch.distributed as dist
from fastvideo import VideoGenerator

negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

def get_dist_info():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    return rank, world_size, local_rank

def init_dist(backend="nccl"):
    # torchrun sets MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE for env://
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size

def main():
    global_rank, world_size = init_dist(backend="gloo")
    _, _, local_rank = get_dist_info()
    # global_rank, world_size, local_rank = get_dist_info()
    print("reached from rank", global_rank, "local rank", local_rank, "world size", world_size)

    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"
    run_name = "vsa_wan1.3b_0.9sparse"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    # Create a video generator with a pre-trained model
    generator = VideoGenerator.from_pretrained(
        "/workspace/vsa_checkpoint",
        num_gpus=1,
        VSA_sparsity=0.9,
        dit_cpu_offload=False,
        use_fsdp_inference=False,
        text_encoder_cpu_offload=False,
        image_encoder_cpu_offload=False,
        vae_cpu_offload=False,
    )

    if global_rank == 0:
        wandb.init(
            config=asdict(generator.fastvideo_args),
            name=run_name,
            mode="online",
            project="self_forcing",
            dir="/workspace/wandb"
        )
    
    os.makedirs("/workspace/vbench_videos", exist_ok=True)

    with open("assets/all_dimension_extended.txt", "r") as f:
        prompts = [line.strip() for line in f.readlines()]

    with open("assets/all_dimension.txt", "r") as f:
        names = [line.strip() for line in f.readlines()]

    bsz_per_gpu = len(prompts) // world_size
    print(f"rank {global_rank} processing {global_rank * bsz_per_gpu} to {(global_rank + 1) * bsz_per_gpu}")
    if global_rank < len(prompts) % world_size:
        print(f"rank {global_rank} processing extra prompt")
    local_prompts = prompts[global_rank * bsz_per_gpu: (global_rank + 1) * bsz_per_gpu]
    local_names = names[global_rank * bsz_per_gpu: (global_rank + 1) * bsz_per_gpu]
    if global_rank < len(prompts) % world_size:
        local_prompts.append(prompts[-global_rank - 1])
        local_names.append(names[-global_rank - 1])
    for sample_num in range(5):
        for prompt, name in zip(local_prompts, local_names):
            print(f"rank {global_rank} generating sample {sample_num} for prompt: {prompt}")
            path = f"/workspace/vbench_videos/{name}-{sample_num}.mp4"
            generator.generate_video(
                prompt,
                negative_prompt=negative_prompt,
                output_path=path,
                save_video=True,
                num_inference_steps=50,
                guidance_scale=5.0,
                num_frames=81,
                height=448,
                width=832,
                fps=16,
                seed=42 + global_rank * 5 + sample_num,
            )
    
    dist.barrier()

    if local_rank == 0:
        print(f"uploading data from rank {global_rank}")
        os.system("ls -l /workspace/vbench_videos | wc -l")
        os.system(f"aws s3 cp /workspace/vbench_videos s3://agi-mm-training-shared-us-east-2/beidchen/data/{run_name}_vbench_videos/ --region us-east-2 --recursive")
        print(f"rank {global_rank} finished")

    dist.barrier()

if __name__ == '__main__':
    main()
