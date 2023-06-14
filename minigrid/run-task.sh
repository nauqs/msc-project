#!/bin/bash

# Input Parameters
TASK_ID=${SGE_TASK_ID}
CSV_FILE="/cluster/project2/tithonus/msc-project/minigrid/experiments.csv"

echo "TASK_ID is: $TASK_ID"

# Read the specific line from the csv file
LINE=$(sed "${TASK_ID}q;d" $CSV_FILE)

# Parse the configuration
IFS=, read -r env_id fully_obs wandb total_timesteps num_envs num_steps ent_coef <<< "$LINE"

# Determine the --cuda parameter based on the RESOURCE environment variable
if [ "$RESOURCE" = "gpu" ]; then
    cuda_option="--cuda true"
else
    cuda_option="--cuda false"
fi

# Execute the command
echo "Executing task with parameters: env_id=$env_id, fully_obs=$fully_obs, wandb=$wandb, total_timesteps=$total_timesteps, num_envs=$num_envs, num_steps=$num_steps, ent_coef=$ent_coef, $cuda_option"
cd /cluster/project2/tithonus/msc-project/minigrid
free -g
/cluster/project2/tithonus/miniconda3/bin/conda run -n minigrid python -u train.py --env-id $env_id --fully-obs $fully_obs --wandb $wandb --total-timesteps $total_timesteps --num-envs $num_envs --num-steps $num_steps --ent-coef $ent_coef $cuda_option
