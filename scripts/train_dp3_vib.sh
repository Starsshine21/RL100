#!/bin/bash
# RL-100 IL Stage Training Script with VIB
# Examples:
# bash scripts/train_dp3_vib.sh adroit_hammer 0115 0 0
# bash scripts/train_dp3_vib.sh dexart_laptop 0115 0 0
# bash scripts/train_dp3_vib.sh metaworld_dial-turn 0115 0 0


DEBUG=False
save_ckpt=True

alg_name="dp3_vib"
task_name=${1}
config_name=${alg_name}
addition_info=${2}
seed=${3}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"


gpu_id=${4}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    wandb_mode=offline
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=offline
    echo -e "\033[33mTrain mode\033[0m"
fi

cd 3D-Diffusion-Policy


export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
python train.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt}



