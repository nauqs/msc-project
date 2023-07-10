#!/bin/bash

resource=$1  # This should be 'cpu' or 'gpu'
EXP_FILENAME=${2:-"experiments.csv"}
csv_file="/cluster/project2/tithonus/msc-project/minigrid/$EXP_FILENAME"
output_directory="/cluster/project2/tithonus/logs"

# Count number of lines in csv file (excluding header)
num_tasks=$(($(wc -l < "$csv_file") - 1))

# Determine the resource to be used
if [ "$resource" = "cpu" ]; then
    resource_options="-l tmem=16G,h_vmem=16G,h_rt=4:00:00"
else
    resource_options="-l tmem=16G,h_vmem=16G,h_rt=4:00:00,gpu=true"
    RESOURCE="gpu"
fi

# Submit array job
qsub -t 2-$((num_tasks + 1)) ${resource_options} -S /bin/bash -v RESOURCE=${RESOURCE},EXP_FILENAME=${EXP_FILENAME} -wd /cluster/project2/tithonus/ -j y -N train-minigrid -o ${output_directory} run-task.sh

echo "Array job submitted with $num_tasks tasks"
