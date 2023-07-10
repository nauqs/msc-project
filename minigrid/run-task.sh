#!/bin/bash

# Input Parameters
TASK_ID=${SGE_TASK_ID}
EXP_FILENAME=${EXP_FILENAME}
CSV_FILE="/cluster/project2/tithonus/msc-project/minigrid/$EXP_FILENAME"
seed=${SEED}

echo "TASK_ID is: $TASK_ID"
echo "SEED is: $seed"

# Read the specific line from the csv file
LINE=$(sed "${TASK_ID}q;d" $CSV_FILE)

# Parse the configuration
IFS=, read -r env_id time_cost action_cost final_reward_penalty fully_obs wandb total_timesteps num_envs num_steps ent_coef <<< "$LINE"

# Determine the --cuda parameter based on the RESOURCE environment variable
if [ "$RESOURCE" = "gpu" ]; then
    cuda_option="--cuda true"
else
    cuda_option="--cuda false"
fi

# Create the experiment name
timestamp=$(date "+%m%d-%H%M%S.%3N")
exp_name="${timestamp}_${TASK_ID}_${seed}"

# Execute the command
echo "Executing task with parameters: env_id=$env_id, time_cost=$time_cost, action_cost=$action_cost, final_reward_penalty=$final_reward_penalty, fully_obs=$fully_obs, wandb=$wandb, seed=$seed, total_timesteps=$total_timesteps, num_envs=$num_envs, num_steps=$num_steps, ent_coef=$ent_coef, exp_name=$exp_name, $cuda_option"
cd /cluster/project2/tithonus/msc-project/minigrid
free -g
/cluster/project2/tithonus/miniconda3/bin/conda run -n minigrid python -u train.py --env-id $env_id --time-cost $time_cost --action-cost $action_cost --final_reward_penalty $final_reward_penalty --fully-obs $fully_obs --wandb $wandb --seed $seed --total-timesteps $total_timesteps --num-envs $num_envs --num-steps $num_steps --ent-coef $ent_coef --exp-name $exp_name $cuda_option
