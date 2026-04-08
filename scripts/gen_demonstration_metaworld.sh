# bash scripts/gen_demonstration_metaworld.sh peg-insert-side [sparse|dense]



cd third_party/Metaworld

task_name=${1}
reward_type=${2:-sparse}
task_config="../../3D-Diffusion-Policy/diffusion_policy_3d/config/task/metaworld_${task_name}.yaml"

if [ ! -f "${task_config}" ]; then
    echo "Task config not found: ${task_config}" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0
python gen_demonstration_expert.py --env_name=${task_name} \
            --task_config "${task_config}" \
            --root_dir "../../3D-Diffusion-Policy/data/" \
            --reward_type ${reward_type}
