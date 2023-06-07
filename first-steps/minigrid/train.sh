#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -l h_rt=1:59:00
#$ -l gpu=true

### to exclude small gpus:
### gpu_type=!(gtx1080ti|titanx)
### gpu_type=!(gtx1080ti|rtx2080ti|titanxp|titanx)

#$ -S /bin/bash
#$ -wd /cluster/project2/tithonus/
#$ -j y # merge stdout and stderr
#$ -N train-minigrid

cd /cluster/project2/tithonus/msc-project/first-steps/minigrid
free -g
echo "Assigned device: $CUDA_VISIBLE_DEVICES"

timestamp=$(date +"%m%d_%H%M%S")

mkdir -p logs
echo "Running train.py"
/cluster/project2/tithonus/anaconda3/bin/conda run -n minigrid python -u train.py --env_size 8x8  --max_timesteps 1000000 --timestamp "$timestamp" &>> "logs/logs_$timestamp.txt"