#!/bin/bash

resource=$1  # This should be 'cpu' or 'gpu'
EXP_FILENAME=${2:-"experiments.csv"}
csv_file="/cluster/project2/tithonus/msc-project/minigrid/$EXP_FILENAME"
output_directory="/cluster/project2/tithonus/logs"
SEED_RANGE=$3  # This should be a range of seeds, like "1-100"

# Extract start and end of the seed range
IFS='-' read -r -a RANGE <<< "$SEED_RANGE"
START_SEED=${RANGE[0]}
END_SEED=${RANGE[1]}

# Calculate the number of seeds
NUM_SEEDS=$((END_SEED - START_SEED + 1))

# Count number of lines in csv file (excluding header)
num_tasks=$(($(wc -l < "$csv_file") - 1))

# Determine the resource to be used
if [ "$resource" = "cpu" ]; then
    resource_options="-l tmem=16G,h_vmem=16G,h_rt=1:00:00"
else
    resource_options="-l tmem=16G,h_vmem=16G,h_rt=1:00:00,gpu=true"
    RESOURCE="gpu"
fi

# Submit array job
qsub -t 1-$((num_tasks * NUM_SEEDS)) ${resource_options} -S /bin/bash -v RESOURCE=${RESOURCE},EXP_FILENAME=${EXP_FILENAME},START_SEED=${START_SEED},NUM_SEEDS=${NUM_SEEDS},NUM_TASKS=${num_tasks} -wd /cluster/project2/tithonus/ -j y -N train-minigrid -o ${output_directory} run-task.sh

echo "Array job submitted with $((num_tasks * NUM_SEEDS)) tasks"
