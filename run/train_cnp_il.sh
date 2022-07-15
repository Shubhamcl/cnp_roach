#!/bin/bash

# * BC phase of training the IL agent `L_A(AP)` on the NoCrash benchmark.
# Addition: CNP, new trainer and new model
# FIX: create small dataset and update in the run command
# Add params for cfg policy entry point and train config entry point
train_cnp_il () {
python -u train_il.py reset_step=true \
agent.cilrs.wb_run_path=null agent.cilrs.wb_ckpt_step=null \
wb_project="il_nocrash_ap" wb_group="train0" 'wb_name="L_A(AP)"' \
dagger_datasets=["/scratch/lustre/home/shju7830/data/bc_small/"] \
'agent.cilrs.env_wrapper.kwargs.input_states=[speed]' \
agent.cilrs.policy.kwargs.number_of_branches=6 \
agent.cilrs.training.kwargs.branch_weights=[1.0,1.0,1.0,1.0,1.0,1.0] \
agent.cilrs.env_wrapper.kwargs.action_distribution=null \
agent.cilrs.rl_run_path=null agent.cilrs.rl_ckpt_step=null \
agent.cilrs.training.kwargs.action_kl=false \
agent.cilrs.env_wrapper.kwargs.value_as_supervision=false \
agent.cilrs.training.kwargs.value_weight=0.0 \
agent.cilrs.env_wrapper.kwargs.dim_features_supervision=0 \
agent.cilrs.training.kwargs.features_weight=0.0 \
agent.cilrs.training.kwargs.batch_size=16 \
agent.cilrs.policy.entry_point="agents.cilrs.models.cilrs_model:CoILICRA_CNP" \
agent.cilrs.training.entry_point="agents.cilrs.models.trainer:CNP_trainer" \
cache_dir=${CACHE_DIR}
}

source ~/miniconda3/bin/activate
conda activate carla

NODE_ROOT=~/tmp_data
mkdir -p "${NODE_ROOT}"
CACHE_DIR=$(mktemp -d --tmpdir="${NODE_ROOT}")

echo "CACHE_DIR: ${CACHE_DIR}"

train_cnp_il

echo "Python finished!!"
rm -rf "${CACHE_DIR}"
echo "Bash script done!!"
echo finished at: `date`
exit 0;

