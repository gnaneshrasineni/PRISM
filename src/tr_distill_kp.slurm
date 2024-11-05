#!/bin/bash
#SBATCH --time=11:59:00
#SBATCH --nodes=1
#SBATCH -o ../logs/slurm-%j.out-%N
#SBATCH -e ../logs/slurm-%j.err-%N
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --account=soc-gpu-kp
#SBATCH --partition=soc-gpu-kp
#SBATCH --gres=gpu:p100:2
#SBATCH --mail-user=rasinenignanesh@gmail.com

conda activate /scratch/general/vast/u1470943/envs/prism
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
module load cuda/12.1.0

module load cudnn/8.9.7.29-12-gpu

set +x
srun /scratch/general/vast/u1470943/envs/prism/bin/torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 train.py  --data colon --data_dir /scratch/general/vast/u1470943/Research/data/datasets/Task10_colon --save_name prism_3point_distill_kldiv_ultra --multiple_outputs --dynamic --use_box --use_scribble --efficient_scribble --multiple_outputs --refine --use_distillation --checkpoint_distiller /scratch/general/vast/u1470943/Research/distillation/PRISM/src/implementation/colon/prism_plain_colon/best.pth.tar --ddp

srun /scratch/general/vast/u1470943/envs/prism/bin/torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 train.py --data colon --data_dir /scratch/general/vast/u1470943/Research/data/datasets/Task10_colon --save_name prism_3point_distill_kldiv_ultra_plus --multiple_outputs --dynamic --use_box --use_scribble --efficient_scribble --multiple_outputs --refine --use_distillation --checkpoint_distiller /scratch/general/vast/u1470943/Research/distillation/PRISM/src/implementation/colon/prism_ultra_plus_colon/best.pth.tar --ddp
