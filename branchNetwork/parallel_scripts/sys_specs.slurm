#!/bin/bash
#SBATCH --job-name=SysSpecs         # Job name
#SBATCH --nodes=1                   # Ensure we are running on a single node
#SBATCH --exclusive
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --time=00:10:00                # Time limit hrs:min:sec
#SBATCH --output=sys_specs_%j.out   # Standard output and error log (job ID will replace %j)
#SBATCH --partition=gpu
#SBATCH --account=Tau

echo "Gathering system specifications..."
echo "Gathering system specifications on the node:"
hostname


# Print CPU info
echo "CPU Information:"
lscpu

# Load the CUDA module
echo "Loading CUDA module..."
module load cuda/11.5.1

# Check if NVIDIA GPUs are available and print GPU info
echo "GPU Information:"
if command -v nvidia-smi &> /dev/null
then
    nvidia-smi
else
    echo "No NVIDIA GPUs detected."
fi

echo "Job completed."
