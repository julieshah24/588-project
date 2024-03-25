import os
import tempfile
import time
from tqdm import tqdm

HYPERPARAMETER_DIR =  "/nfs/turbo/justincj-turbo/kaulg/spam/hp_search"

learning_rates = [1e-4,1e-5,1e-6,1e-7]

for lr in tqdm(learning_rates):
    hp_folder_name = (
        "no_train_eval_wd_0_lr_" + str(lr).replace('.','_') + '/'
    )
    name = 'spam_roberta_lr_' + str(lr)
    output_path = os.path.join(HYPERPARAMETER_DIR, hp_folder_name)
    slurm_log_dir = os.path.join(output_path, "program_output.log")
    print(output_path)
    os.makedirs(output_path, exist_ok=True)

    SBATCH_STRING = """#!/bin/sh
#SBATCH --account=justincj1 
#SBATCH --partition=spgpu
#SBATCH --job-name={name}
#SBATCH --output={slurm_log_dir}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=45GB

export PATH=/nfs/turbo/justincj-turbo/kaulg/miniconda3/envs/spam/bin:$PATH

conda init

conda activate spam 

cd /nfs/turbo/justincj-turbo/kaulg/spam

python train_model.py --learning_rate {lr}
    

"""

    SBATCH_STRING = SBATCH_STRING.format(
        lr=lr,
        slurm_log_dir=slurm_log_dir,
        name=name,
    )
    print(SBATCH_STRING)
    dirpath = tempfile.mkdtemp()
    with open(os.path.join(dirpath, "scr.sh"), "w") as tmpfile:
        tmpfile.write(SBATCH_STRING)
    os.system(f"sbatch {os.path.join(dirpath, 'scr.sh')}")
    print(f"Launched from {os.path.join(dirpath, 'scr.sh')}")
    time.sleep(0.01)
