#!/bin/bash
#SBATCH --time=11:59:00
#SBATCH --nodes=1
#SBATCH -o ../logs/slurm-%j.out-%N
#SBATCH -e ../logs/slurm-%j.err-%N
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=25G
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --gres=gpu:a6000:1
#SBATCH --mail-user=rasinenignanesh@gmail.com

conda activate /scratch/general/vast/u1470943/envs/prism

module load cuda/12.1.0

module load cudnn/8.9.7.29-12-gpu

set +x
/scratch/general/vast/u1470943/envs/prism/bin/python train.py --data colon --data_dir /scratch/general/vast/u1470943/Research/data/datasets/Task10_colon --save_name prism_3point_distill_kldiv_plain --multiple_outputs --dynamic --use_box --multiple_outputs --refine --use_distillation --checkpoint_distiller /scratch/general/vast/u1470943/Research/distillation/PRISM/src/implementation/colon/prism_plain_colon/best.pth.tar --resume