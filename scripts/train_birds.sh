#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=General_Usage
#SBATCH --gres=gpu:4
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=3-08:00:00

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}


export TMPDIR=/disk/scratch/${STUDENT_ID}
export TMP=/disk/scratch/${STUDENT_ID}

mkdir -p ${TMP}/dataset
export DATASET_DIR=${TMP}/dataset

rsync -uav dataset/CUB_200_2011/ ${DATASET_DIR}/CUB_200_2011/

# Activate the relevant virtual environment:

source /home/${STUDENT_ID}/miniconda3/bin/activate mlp

python train.py \
    --img_root ${DATASET_DIR}/CUB_200_2011/images \
    --caption_root ${DATASET_DIR}/CUB_200_2011/cub_icml \
    --trainclasses_file trainvalclasses.txt \
    --save_filename_G ./birds/G.pth \
    --save_filename_D ./birds/D.pth \
    --save_filename_stats ./birds/ \
    --lambda_cond_loss 10 \
    --lambda_recon_loss 0.2
    --instance_noise 0.1
